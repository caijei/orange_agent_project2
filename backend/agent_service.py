"""
OrangeAgent v2 — 升级点：
  1. 多轮对话记忆（每个 session 独立历史）
  2. 并行工具执行（asyncio.gather）
  3. 工具失败自动重试
  4. 流式输出：工具调用阶段实时推送状态，最终回答真流式
  5. 上下文裁剪（防止历史过长撑爆 context window）
  6. 推理步骤透明化（可选，供前端展示 "思考过程"）

这个文件可以理解为项目里的“智能体大脑”：
  - TOOLS：告诉大模型“你有哪些工具可以用、每个工具需要什么参数”。
  - ToolExecutor：真正执行工具，比如查知识库、查天气、联网搜索、图片诊断。
  - ConversationMemory：保存每个用户会话的上下文，让系统能记住前面说过什么。
  - OrangeAgent：组织完整问答流程，让模型先判断要不要调用工具，再根据工具结果生成回答。
"""

# 标准库：处理异步、JSON、日志、环境变量、时间、路径、类型提示等基础功能。
import asyncio
import json
import logging
import os
import time
import datetime
import requests
from urllib.parse import quote
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Optional

# 第三方库：
# - dashscope：调用通义千问视觉模型，用于图片诊断。
# - dotenv：读取 .env 配置文件。
# - OpenAI：这里使用 OpenAI 兼容接口调用本地 Ollama 或其它兼容模型。
# - TavilyClient：联网搜索工具。
from dashscope import MultiModalConversation
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

# 项目内服务：RAGService 负责本地知识库检索和基于知识库的问答。
from rag_service import RAGService

# 加载 backend/.env 中的环境变量，比如模型地址、API Key 等。
load_dotenv(Path(__file__).with_name(".env"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------

# TOOLS 是“工具说明书”，不是工具真正执行的代码。
# 大模型看到这份列表后，会根据用户问题自动选择是否调用某个工具。
# 真正执行工具的逻辑在下面的 ToolExecutor 类中。
TOOLS = [
    # 工具 1：本地知识库检索。适合回答稳定的脐橙种植知识。
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "从本地脐橙种植知识库中检索稳定技术知识。"
                "内部已集成 CRAG：会先检索本地知识库并评估质量；"
                "如果检索质量不足，且当前允许联网，则自动联网补充。"
                "适用：病虫害防治、施肥修剪、品种特性、采后处理等农业技术问题。"
                "实时天气、最新价格、近期新闻也可以先查知识库，再由 CRAG 判断是否联网补充。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词，建议简洁中文短语"},
                },
                "required": ["query"],
            },
        },
    },
    # 工具 2：联网搜索。适合查实时变化的信息，比如价格、政策、新闻。
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "联网搜索实时信息：天气、价格、政策、市场行情、近期新闻。"
                "稳定的农业技术知识请优先用 search_knowledge_base。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索词，建议加地域或时间限定词"},
                },
                "required": ["query"],
            },
        },
    },
    # 工具 3：图片诊断。适合用户上传叶片、果实、树体图片时调用。
    {
        "type": "function",
        "function": {
            "name": "diagnose_image",
            "description": (
                "对脐橙叶片/果实/树体图片进行病虫害诊断。"
                "返回：现象判断、可能原因、处理建议、是否需要线下复核。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "图片的本地绝对路径"},
                },
                "required": ["image_path"],
            },
        },
    },
    # 工具 4：施肥计算。适合用户给出面积、目标产量、肥料类型后做简单估算。
    {
        "type": "function",
        "function": {
            "name": "calculate_fertilizer",
            "description": (
                "根据果园面积和目标产量计算施肥用量。"
                "当用户询问'XX 亩需要施多少肥'类问题时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area_mu": {"type": "number", "description": "果园面积（亩）"},
                    "target_yield_kg": {"type": "number", "description": "目标产量（公斤/亩），不提供则用经验值 2500"},
                    "fertilizer_type": {
                        "type": "string",
                        "enum": ["氮肥", "磷肥", "钾肥", "复合肥", "有机肥"],
                        "description": "肥料类型",
                    },
                },
                "required": ["area_mu", "fertilizer_type"],
            },
        },
    },
    # 工具 5：追问用户。信息不足时，Agent 不硬答，而是先问清楚。
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "当用户描述模糊、信息不足以准确回答时，向用户提出追问。"
                "适用场景：\n"
                "- 症状描述不清（如只说'树有问题'）\n"
                "- 缺少关键信息（面积、树龄、品种、发病部位等）\n"
                "- 问题有多种可能原因需要区分\n"
                "不适用：问题已经足够清晰时不要用此工具，直接查知识库。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "向用户提出的追问内容，要简洁具体，一次最多问2个问题",
                    },
                    "reason": {
                        "type": "string",
                        "description": "需要追问的原因，简短说明（用于内部日志）",
                    },
                },
                "required": ["question"],
            },
        },
    },
    # 工具 6：天气查询。天气属于实时信息，单独做成工具比普通联网搜索更准确。
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "查询指定城市的实时天气。"
                "适用：用户询问某地今天、现在、当前天气、气温、下雨、降雨、湿度等问题。"
                "如果用户问天气，优先使用本工具，不要使用 search_web。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：赣州、上海、北京、信丰、Tokyo",
                    },
                },
                "required": ["city"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# 施肥计算（无需外部 API 的本地工具，展示 Agent 可扩展性）
# ---------------------------------------------------------------------------

# 每种肥料的经验用量基准。
# base 表示在目标产量 2500kg/亩 时，每亩建议使用量。
# 后面 _calculate_fertilizer 会根据目标产量按比例放大或缩小。
FERTILIZER_RATES = {
    "氮肥": {"base": 0.6, "unit": "kg/亩"},       # 纯氮
    "磷肥": {"base": 0.3, "unit": "kg/亩"},       # P2O5
    "钾肥": {"base": 0.8, "unit": "kg/亩"},       # K2O
    "复合肥": {"base": 2.5, "unit": "kg/亩"},
    "有机肥": {"base": 50.0, "unit": "kg/亩"},
}

import requests
from urllib.parse import quote


def get_weather(city: str) -> str:
    """
    查询指定城市天气信息，并返回较详细的自然语言结果。
    包含：
    1. 当前天气
    2. 今日最高/最低温
    3. 降雨概率/降水量
    4. 风速、湿度、能见度
    5. 未来3天预报
    6. 面向脐橙种植的简单农事提醒
    """
    # 去掉用户输入前后的空格，避免 “ 赣州 ” 这种输入导致查询失败。
    city = city.strip()
    if not city:
        return "错误：城市名称不能为空。"

    # wttr.in 是一个免费的天气查询接口。
    # quote(city) 用来处理中文城市名，避免中文直接放进 URL 后出错。
    url = f"https://wttr.in/{quote(city)}"

    try:
        # 请求 JSON 格式天气数据，timeout 防止接口一直无响应卡住后端。
        response = requests.get(
            url,
            params={
                "format": "j1",
                "lang": "zh",
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()

        # 当前天气：接口返回的是列表，这里取第 1 条当前天气记录。
        current = data["current_condition"][0]

        weather_desc = current["weatherDesc"][0]["value"]
        temp_c = current.get("temp_C", "未知")
        feels_like_c = current.get("FeelsLikeC", "未知")
        humidity = current.get("humidity", "未知")
        wind_kmph = current.get("windspeedKmph", "未知")
        wind_dir = current.get("winddir16Point", "未知")
        precip_mm = current.get("precipMM", "未知")
        pressure = current.get("pressure", "未知")
        visibility = current.get("visibility", "未知")
        uv_index = current.get("uvIndex", "未知")

        # 今日天气：包括当天最高温、最低温、日照、小时级天气等。
        today = data["weather"][0]
        today_date = today.get("date", "今天")
        max_temp = today.get("maxtempC", "未知")
        min_temp = today.get("mintempC", "未知")
        avg_temp = today.get("avgtempC", "未知")
        total_snow_cm = today.get("totalSnow_cm", "0")
        sun_hour = today.get("sunHour", "未知")
        uv_today = today.get("uvIndex", uv_index)

        # 今日逐小时信息，取中间时段作为当天白天情况的参考。
        hourly_list = today.get("hourly", [])
        noon_hour = hourly_list[len(hourly_list) // 2] if hourly_list else {}

        chance_of_rain = noon_hour.get("chanceofrain", "未知")
        chance_of_snow = noon_hour.get("chanceofsnow", "未知")
        cloud_cover = noon_hour.get("cloudcover", "未知")

        # 未来 3 天预报：整理成文本列表，方便后续模型直接引用。
        forecast_lines = []
        for day in data.get("weather", [])[:3]:
            date = day.get("date", "未知日期")
            day_max = day.get("maxtempC", "未知")
            day_min = day.get("mintempC", "未知")
            day_avg = day.get("avgtempC", "未知")
            day_uv = day.get("uvIndex", "未知")

            hourly = day.get("hourly", [])
            mid = hourly[len(hourly) // 2] if hourly else {}
            desc = "未知"
            if mid.get("weatherDesc"):
                desc = mid["weatherDesc"][0].get("value", "未知")

            rain_chance = mid.get("chanceofrain", "未知")
            rain_mm = mid.get("precipMM", "未知")
            wind = mid.get("windspeedKmph", "未知")
            hum = mid.get("humidity", "未知")

            forecast_lines.append(
                f"- {date}：{desc}，{day_min}～{day_max}℃，"
                f"平均 {day_avg}℃，降雨概率 {rain_chance}%，"
                f"预计降水 {rain_mm} mm，湿度 {hum}%，风速 {wind} km/h，紫外线指数 {day_uv}"
            )

        # 简单农事提醒：根据温度、湿度、风速、降水等指标给出种植管理提示。
        # 这里不是复杂模型推理，只是一些基础规则判断。
        farming_tips = []

        try:
            temp_value = float(temp_c)
            humidity_value = float(humidity)
            wind_value = float(wind_kmph)
            precip_value = float(precip_mm)
            uv_value = float(uv_index)
        except ValueError:
            temp_value = humidity_value = wind_value = precip_value = uv_value = None

        if precip_value is not None:
            if precip_value > 5:
                farming_tips.append("当前降水较明显，果园应注意排水，避免积水导致根系缺氧。")
            elif precip_value == 0:
                farming_tips.append("当前无明显降水，如近期持续干旱，可结合土壤墒情适当灌溉。")

        if humidity_value is not None and humidity_value >= 85:
            farming_tips.append("空气湿度较高，病害发生风险可能增加，应注意炭疽病、溃疡病等病害巡查。")

        if temp_value is not None:
            if temp_value >= 35:
                farming_tips.append("气温较高，应注意脐橙日灼风险，可采取树盘覆盖、合理灌水、改善树冠通风等措施。")
            elif temp_value <= 5:
                farming_tips.append("气温较低，应关注低温冻害风险，幼树和弱树可加强防寒保护。")

        if wind_value is not None and wind_value >= 30:
            farming_tips.append("风速较大，不建议喷药或叶面施肥，避免药液漂移和效果下降。")

        if uv_value is not None and uv_value >= 7:
            farming_tips.append("紫外线较强，果实裸露部位日灼风险增加，应注意树冠管理和果面保护。")

        if not farming_tips:
            farming_tips.append("当前天气条件未发现明显高风险因素，仍建议结合果园土壤湿度、树势和病虫害情况综合管理。")

        # 把天气数据组织成一段完整文本，作为工具结果返回给大模型。
        # 大模型拿到这段结果后，再用更自然的语言回复用户。
        result = f"""【{city}天气查询结果】

一、当前天气
- 天气状况：{weather_desc}
- 当前气温：{temp_c}℃
- 体感温度：{feels_like_c}℃
- 空气湿度：{humidity}%
- 风向风速：{wind_dir}，{wind_kmph} km/h
- 当前降水量：{precip_mm} mm
- 气压：{pressure} hPa
- 能见度：{visibility} km
- 紫外线指数：{uv_index}

二、今日天气概况
- 日期：{today_date}
- 最高气温：{max_temp}℃
- 最低气温：{min_temp}℃
- 平均气温：{avg_temp}℃
- 降雨概率：{chance_of_rain}%
- 降雪概率：{chance_of_snow}%
- 云量：{cloud_cover}%
- 日照时长：{sun_hour} 小时
- 今日紫外线指数：{uv_today}
- 降雪量：{total_snow_cm} cm

三、未来3天预报
{chr(10).join(forecast_lines)}

四、脐橙种植管理提醒
{chr(10).join(f"- {tip}" for tip in farming_tips)}

说明：以上天气数据来自 wttr.in，适合作为实时参考；具体农事操作还应结合果园实际土壤湿度、树龄、树势和病虫害情况判断。
"""

        return result

    except requests.exceptions.RequestException as e:
        return f"错误：查询天气时遇到网络问题 - {e}"

    except (KeyError, IndexError, ValueError, TypeError) as e:
        return f"错误：解析天气数据失败，可能是城市名称无效或接口返回格式变化 - {e}"

def _calculate_fertilizer(area_mu: float, fertilizer_type: str, target_yield_kg: float = 2500) -> str:
    # 根据肥料类型找到对应的经验基准用量。
    rate_info = FERTILIZER_RATES.get(fertilizer_type)
    if not rate_info:
        return f"未知肥料类型：{fertilizer_type}"

    # 以 2500kg/亩 为标准产量。
    # 例如目标产量是 5000kg/亩，yield_factor 就是 2，用量也按 2 倍估算。
    yield_factor = target_yield_kg / 2500  # 以 2500kg/亩 为基准
    per_mu = rate_info["base"] * yield_factor
    total = per_mu * area_mu
    return (
        f"施肥计算结果：\n"
        f"  肥料类型：{fertilizer_type}\n"
        f"  果园面积：{area_mu} 亩\n"
        f"  目标产量：{target_yield_kg} kg/亩\n"
        f"  建议用量：{per_mu:.1f} {rate_info['unit']}，共 {total:.1f} kg\n"
        f"注意：以上为经验参考值，具体用量请结合土壤检测报告调整。"
    )

# ---------------------------------------------------------------------------
# 工具执行层
# ---------------------------------------------------------------------------

class ToolExecutor:
    """
    工具执行器。

    前面的 TOOLS 只是给大模型看的“工具说明书”，
    这个类才是后端真正执行工具的地方。

    简单理解：
      大模型决定：我要调用 search_knowledge_base。
      ToolExecutor 负责：真的去查知识库，并把结果拿回来。
    """

    # 每个工具失败后最多重试 2 次，减少偶发网络错误带来的影响。
    MAX_RETRIES = 2

    def __init__(self, rag_service: RAGService):
        # 本地知识库服务，用于 search_knowledge_base。
        self.rag_service = rag_service

        # Tavily 是联网搜索服务；如果 .env 没有配置 TAVILY_API_KEY，就禁用联网搜索。
        self.tavily_client = (
            TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            if os.getenv("TAVILY_API_KEY") else None
        )

        # 通义千问视觉模型的 API Key，用于 diagnose_image。
        self.vl_api_key = os.getenv("DASHSCOPE_API_KEY")

    def _run_once(self, tool_name: str, tool_args: dict) -> str:
        """
        执行一次指定工具。

        tool_name 是大模型选择的工具名；
        tool_args 是大模型生成的工具参数。

        这里用 if/elif 做“工具路由”：
        根据 tool_name 不同，把请求分发给不同的真实函数或服务。
        """
        if tool_name == "search_knowledge_base":
            # _allow_web 由 chat_stream 根据 search_mode 注入：
            # local 模式为 False；auto / web 模式为 True。
            allow_web = tool_args.get("_allow_web", True)

            # 调用 RAGService，从本地向量知识库中检索脐橙相关资料。
            # 如果 allow_web=True，RAGService 内部可能在本地结果不足时联网补充。
            result = self.rag_service.ask_question(
                tool_args["query"],
                allow_web=allow_web,
            )
            return result.get("answer") or "知识库中未找到相关信息。"
        elif tool_name == "get_weather":
            # 天气问题直接走专门的天气函数，而不是普通联网搜索。
            city = tool_args.get("city", "").strip()
            return get_weather(city)

        elif tool_name == "search_web":
            # local 模式下禁止联网，避免违背用户选择的“仅本地知识库”模式。
            if tool_args.get("_allow_web") is False:
                return "当前为仅本地知识库模式，已禁止联网搜索。"
            if not self.tavily_client:
                return "未配置联网搜索（缺少 TAVILY_API_KEY）。"

            # 调用 Tavily 搜索，并限制最多返回 3 条结果，避免上下文太长。
            result = self.tavily_client.search(
                query=tool_args["query"],
                search_depth="basic",
                max_results=3,
                include_answer="basic",
            )
            parts = []
            if ans := result.get("answer"):
                parts.append(f"摘要：{ans}")
            for item in result.get("results", [])[:3]:
                content = (item.get("content") or "")[:200]
                parts.append(f"{item.get('title', '')}\n{content}\n来源：{item.get('url', '')}")
            return "\n\n".join(parts) or "未检索到有用结果。"

        elif tool_name == "diagnose_image":
            # 图片路径可能来自前端或用户输入，先去掉多余引号和空格。
            path = tool_args["image_path"].strip(' "\'')
            if not os.path.exists(path):
                return f"找不到图片：{path}"
            abs_path = os.path.abspath(path)

            # 调用视觉大模型，让它根据图片判断病虫害现象和处理建议。
            resp = MultiModalConversation.call(
                model="qwen-vl-plus",
                api_key=self.vl_api_key,
                messages=[{
                    "role": "user",
                    "content": [
                        {"image": f"file://{abs_path}"},
                        {"text": '你是专业脐橙病虫害诊断专家。请按"现象判断、可能原因、处理建议、是否需要线下复核"四部分给出结论。'},
                    ],
                }],
            )
            if resp.status_code != 200:
                raise RuntimeError(f"图片诊断失败：{resp.code} - {resp.message}")
            content = resp.output.choices[0].message.content
            return "".join(i.get("text", "") for i in content) if isinstance(content, list) else str(content)

        elif tool_name == "calculate_fertilizer":
            # 本地计算工具，不需要联网，也不需要大模型参与计算。
            return _calculate_fertilizer(
                area_mu=tool_args["area_mu"],
                fertilizer_type=tool_args["fertilizer_type"],
                target_yield_kg=tool_args.get("target_yield_kg", 2500),
            )
        elif tool_name == "ask_user":
            # ask_user 不做任何外部调用，直接返回特殊标记
            # chat_stream 检测到这个标记后会终止循环、把追问发给前端
            return f"__ASK_USER__:{tool_args['question']}"

        # 如果大模型返回了 TOOLS 里没有定义的工具名，就返回错误提示。
        return f"未知工具：{tool_name}"

    async def run_async(self, tool_name: str, tool_args: dict) -> str:
        """
        带重试的异步工具执行。

        大部分工具本身是同步函数，比如 requests 请求、知识库查询等。
        asyncio.to_thread 会把这些同步函数放到线程里执行，
        这样不会阻塞整个异步聊天流程。
        """
        last_err = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return await asyncio.to_thread(self._run_once, tool_name, tool_args)
            except Exception as exc:
                last_err = exc
                if attempt < self.MAX_RETRIES:
                    logger.warning(f"工具 {tool_name} 第 {attempt+1} 次失败，重试中：{exc}")
                    # 简单退避：第 1 次等 1 秒，第 2 次等 2 秒。
                    await asyncio.sleep(1.0 * (attempt + 1))
        return f"工具 {tool_name} 执行失败（已重试 {self.MAX_RETRIES} 次）：{last_err}"


# ---------------------------------------------------------------------------
# 对话历史管理（每个 session 独立，自动裁剪）
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    保留最近 N 轮对话 + 系统提示。
    防止 context window 爆炸。

    每个 session_id 都会有自己独立的 ConversationMemory。
    这样不同用户或不同浏览器会话之间不会互相串话。
    """
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        # 每条记录是一个 message dict
        self._history: deque = deque()

    def add(self, message: dict):
        # 把用户消息、助手消息、工具结果都按顺序放进历史。
        self._history.append(message)
        # 一轮 = user + assistant，超出则从最早的非 system 消息删
        while len(self._history) > self.max_turns * 3:  # *3 因为有 tool 消息
            self._history.popleft()

    def get_messages(self, system_prompt: str) -> list:
        # 每次请求模型时，都把 system_prompt 放在最前面。
        # system_prompt 负责告诉模型它是谁、有什么工具、必须遵守什么规则。
        return [{"role": "system", "content": system_prompt}] + list(self._history)

    def clear(self):
        # 清空当前会话历史，相当于开启一段新的对话。
        self._history.clear()


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

# SYSTEM_PROMPT 是给大模型的“角色说明 + 工作规则”。
# 它会影响模型是否调用工具、优先调用哪个工具、回答风格是什么。
# 注意：这里不是直接显示给用户看的内容，而是每次请求模型时作为 system 消息发送。
SYSTEM_PROMPT = """你是专业的脐橙种植智能助手，服务于果农和农技人员。

可用工具：
- search_knowledge_base：查询本地知识库（病虫害、施肥、修剪等稳定技术）
- get_weather：查询指定城市的实时天气、气温、湿度、降雨等信息
- search_web：联网获取实时信息（价格、政策、新闻、市场行情）
- diagnose_image：对图片进行病虫害诊断
- calculate_fertilizer：根据面积和产量计算施肥用量
- ask_user：用户描述模糊或缺少关键信息时，主动向用户追问

重要约束：
- 回答任何脐橙种植相关问题时，必须先调用 search_knowledge_base
- 用户询问天气、气温、降雨、湿度时，优先调用 get_weather，不要调用 search_web
- 禁止直接凭自身知识回答专业农业问题，必须以知识库结果为准
- 如果知识库没有相关内容，明确告知用户"知识库暂无此信息"

工作原则：
1. 能本地解决的不联网，优先知识库
2. 天气问题使用 get_weather
3. 价格、政策、新闻、市场行情使用 search_web
4. 多个工具可以同时调用（并行执行，速度更快）
5. 工具结果有限时，诚实说明，不编造信息
6. 图片诊断结论仅供参考，建议结合实地复核
7. 回答简洁、可执行，避免套话
8. 记住用户在本次对话中说过的信息（如面积、品种等），避免重复询问
9. 天气查询结果必须使用简洁列表展示，不要使用 Markdown 表格。
"""


# ---------------------------------------------------------------------------
# OrangeAgent v2
# ---------------------------------------------------------------------------

class OrangeAgent:
    """
    脐橙智能问答 Agent 的主类。

    这个类负责把几件事串起来：
      1. 接收用户问题；
      2. 把问题和历史上下文交给大模型；
      3. 让大模型自动判断是否需要调用工具；
      4. 执行工具并把结果再交给大模型；
      5. 输出最终回答。
    """

    # 最多允许模型进行 6 轮“思考 -> 调工具 -> 看结果”的循环。
    # 防止模型一直调用工具，导致请求停不下来。
    MAX_ROUNDS = 6

    def __init__(self):
        # 从环境变量读取模型服务地址。
        # 默认地址是 Ollama 的 OpenAI 兼容接口。
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

        # 使用哪个大模型，由 .env 中的 OLLAMA_MODEL 决定。
        # 没配置时默认用 qwen2.5:3b。
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

        # 初始化本地知识库问答服务。
        self.rag_service = RAGService()

        # 创建大模型客户端。
        # 虽然类名叫 OpenAI，但这里可以连接任何兼容 OpenAI 接口的模型服务。
        self.llm = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.ollama_base_url,
        )

        # 工具执行器：负责真正执行知识库、联网搜索、天气、图片诊断等工具。
        self.executor = ToolExecutor(self.rag_service)

        # session_id → ConversationMemory
        # 用字典保存多个会话的历史记录。
        self._memories: dict[str, ConversationMemory] = {}

    def get_memory(self, session_id: str) -> ConversationMemory:
        # 如果这个 session 之前没有历史，就新建一个。
        if session_id not in self._memories:
            self._memories[session_id] = ConversationMemory(max_turns=10)
        return self._memories[session_id]

    def clear_memory(self, session_id: str):
        # 清空指定 session 的历史记录。
        if session_id in self._memories:
            self._memories[session_id].clear()

    # ------------------------------------------------------------------
    # 核心：异步流式 ReAct 循环（带并行工具执行）
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        user_message: str,
        session_id: str = "default",
        search_mode: str = "auto",
    ) -> AsyncIterator[str]:
        """
        流式聊天主流程。

        这是 Agent 最核心的方法，可以把它理解为一个 ReAct 循环：
          Reason：大模型先理解用户问题，判断下一步要做什么；
          Act：如果需要工具，就生成 tool_calls；
          Observe：后端执行工具，把工具结果返回给模型；
          Answer：模型结合工具结果生成最终回答。

        search_mode 控制搜索范围：
          - auto：优先本地知识库，必要时允许联网补充；
          - local：只用本地知识库，不允许联网；
          - web：更倾向使用联网搜索。
        """
        # 根据 session_id 拿到本次会话的历史记录。
        memory = self.get_memory(session_id)

        # 先把用户本轮问题存入记忆，后面发给模型时会带上。
        memory.add({"role": "user", "content": user_message})

        # ✅ 动态注入当前日期，每次请求都是准确的
        # 这样用户问“明天”“后天”时，模型能按真实日期理解。
        today = datetime.date.today().strftime("%Y年%m月%d日")
        system = SYSTEM_PROMPT + f"\n\n【当前日期】今天是 {today}，请以此为基准判断明天/后天等相对时间。"

        # 根据前端选择的搜索模式，给模型追加额外规则。
        if search_mode == "web":
            system += "\n\n[当前模式：优先联网搜索]"
        elif search_mode == "local":
            system += "\n\n[当前模式：仅使用本地知识库，禁止调用 search_web]"

        # messages 是最终发给大模型的完整上下文：
        # system_prompt + 历史对话 + 当前用户问题。
        messages = memory.get_messages(system)

        for round_idx in range(self.MAX_ROUNDS):
            # ---- 请求 LLM ----
            # 这里是“工具路由”的第一步：
            # 把 TOOLS 传给大模型，并设置 tool_choice="auto"。
            # 这样模型会自动判断：
            #   - 不需要工具：直接返回自然语言回答；
            #   - 需要工具：返回 tool_calls，里面包含工具名和参数。
            response = await asyncio.to_thread(
                self.llm.chat.completions.create,
                model=self.ollama_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
            msg = response.choices[0].message

            # ---- 不调工具 → 最终答案，真流式输出 ----
            if not msg.tool_calls:
                # 如果模型没有要求调用工具，说明它已经准备好最终回答。
                final_text = msg.content or "抱歉，我暂时无法回答这个问题。"
                # 存入记忆
                memory.add({"role": "assistant", "content": final_text})
                # 流式输出
                # 这里把完整回答切成小块 yield 给前端，让用户看到逐步输出效果。
                chunk_size = 8
                for i in range(0, len(final_text), chunk_size):
                    yield final_text[i: i + chunk_size]
                    await asyncio.sleep(0.05)
                return

            # ---- 有工具调用 ----
            # 1. 推送状态给前端
            # 如果模型决定调用工具，先告诉前端“正在做什么”，提升等待体验。
            tool_names_cn = {
                "search_knowledge_base": "📚 查询知识库",
                "search_web": "🌐 联网搜索",
                "get_weather": "🌦️ 查询天气",
                "diagnose_image": "🔬 图片诊断",
                "calculate_fertilizer": "🧮 施肥计算",
            }
            status_parts = [tool_names_cn.get(tc.function.name, tc.function.name) for tc in msg.tool_calls]

            yield f"__STATUS__:正在执行：{' | '.join(status_parts)}"
            await asyncio.sleep(0)

            # 2. 把 LLM 决策加入消息历史
            # OpenAI 工具调用协议要求：
            # assistant 的 tool_calls 消息后面，必须紧跟对应的 tool 结果消息。
            # LLM 有时会在 tool_calls 里同时带 content（部分预生成文本）
            # 清空它，防止下一轮从残缺文本续写导致"早出字"
            clean_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": msg.tool_calls,
            }
            messages.append(clean_msg)
            memory.add(clean_msg)

            # 3. 并行执行所有工具
            # 有时模型会一次调用多个工具，比如同时查知识库和查天气。
            # 这里先收集所有任务，再用 asyncio.gather 并行执行。
            # 根据 search_mode 注入 _allow_web：
            # - local：CRAG 和 search_web 都不允许联网
            # - auto/web：CRAG 在 LOW/MEDIUM 时允许联网补充
            tasks = []
            for tc in msg.tool_calls:
                # tc.function.arguments 是模型生成的 JSON 字符串，需要转成 Python 字典。
                args = json.loads(tc.function.arguments or "{}")

                # _allow_web 是后端额外加的控制参数，不是用户输入的。
                # local 模式下为 False，其它模式下为 True。
                args["_allow_web"] = (search_mode != "local")

                # 把具体工具交给 ToolExecutor 执行。
                tasks.append(self.executor.run_async(tc.function.name, args))

            # 等待所有工具执行完毕，results 顺序和 tasks 顺序一致。
            results = await asyncio.gather(*tasks)

            # ✅ 检查是否有 ask_user 结果
            # ask_user 是特殊工具：它表示信息不足，需要先追问用户，而不是继续生成答案。
            ask_user_question = None
            for result in results:
                if isinstance(result, str) and result.startswith("__ASK_USER__:"):
                    ask_user_question = result.replace("__ASK_USER__:", "").strip()
                    break

            # 无论是否有 ask_user，先把所有 tool 结果存入历史
            # 保证 assistant(tool_calls) 后面紧跟完整 tool 结果，避免 400 报错
            for tc, result in zip(msg.tool_calls, results):
                # 每个工具结果都必须带上 tool_call_id，
                # 这样模型才能知道这个结果对应刚才哪个工具调用。
                tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
                messages.append(tool_msg)
                memory.add(tool_msg)

            if ask_user_question:
                # 如果工具结果是追问，就直接把追问发给前端，然后结束本轮。
                # 等用户补充信息后，下次请求会继续进入这个流程。
                memory.add({"role": "assistant", "content": ask_user_question})
                yield f"__ASK_USER__:{ask_user_question}"
                return

            # 如果不是 ask_user，本轮循环不会 return。
            # 下一轮会把工具结果再次发给大模型，让模型基于工具结果继续思考或生成最终答案。

        # 超出最大轮次
        # 正常情况下不会走到这里；如果模型一直调用工具不结束，就用这句兜底。
        yield "\n[超出最大推理轮次，请重新提问]"

    # ------------------------------------------------------------------
    # 同步接口（兼容旧调用）
    # ------------------------------------------------------------------

    def chat(self, user_message: str, session_id: str = "default", search_mode: str = "auto") -> str:
        # 有些旧代码可能只想拿到完整字符串，不想处理异步流。
        # 这个方法会内部调用 chat_stream，把所有分片拼成完整回答再返回。
        async def _run():
            parts = []
            async for chunk in self.chat_stream(user_message, session_id=session_id, search_mode=search_mode):
                parts.append(chunk)
            return "".join(parts)
        return asyncio.run(_run())
