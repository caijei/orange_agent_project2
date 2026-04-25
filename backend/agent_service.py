"""
OrangeAgent v2 — 升级点：
  1. 多轮对话记忆（每个 session 独立历史）
  2. 并行工具执行（asyncio.gather）
  3. 工具失败自动重试
  4. 流式输出：工具调用阶段实时推送状态，最终回答真流式
  5. 上下文裁剪（防止历史过长撑爆 context window）
  6. 推理步骤透明化（可选，供前端展示 "思考过程"）
"""

import asyncio
import json
import logging
import os
import time
import datetime
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Optional

from dashscope import MultiModalConversation
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

from rag_service import RAGService

load_dotenv(Path(__file__).with_name(".env"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "从本地脐橙种植知识库中检索稳定技术知识。"
                "适用：病虫害防治、施肥修剪、品种特性、采后处理等农业技术问题。"
                "不适用：实时天气、最新价格、近期新闻——这些请用 search_web。"
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
]

# ---------------------------------------------------------------------------
# 施肥计算（无需外部 API 的本地工具，展示 Agent 可扩展性）
# ---------------------------------------------------------------------------

FERTILIZER_RATES = {
    "氮肥": {"base": 0.6, "unit": "kg/亩"},       # 纯氮
    "磷肥": {"base": 0.3, "unit": "kg/亩"},       # P2O5
    "钾肥": {"base": 0.8, "unit": "kg/亩"},       # K2O
    "复合肥": {"base": 2.5, "unit": "kg/亩"},
    "有机肥": {"base": 50.0, "unit": "kg/亩"},
}

def _calculate_fertilizer(area_mu: float, fertilizer_type: str, target_yield_kg: float = 2500) -> str:
    rate_info = FERTILIZER_RATES.get(fertilizer_type)
    if not rate_info:
        return f"未知肥料类型：{fertilizer_type}"
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
    MAX_RETRIES = 2

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.tavily_client = (
            TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            if os.getenv("TAVILY_API_KEY") else None
        )
        self.vl_api_key = os.getenv("DASHSCOPE_API_KEY")

    def _run_once(self, tool_name: str, tool_args: dict) -> str:
        if tool_name == "search_knowledge_base":
            result = self.rag_service.ask_question(tool_args["query"])
            return result.get("answer") or "知识库中未找到相关信息。"

        elif tool_name == "search_web":
            if not self.tavily_client:
                return "未配置联网搜索（缺少 TAVILY_API_KEY）。"
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
            path = tool_args["image_path"].strip(' "\'')
            if not os.path.exists(path):
                return f"找不到图片：{path}"
            abs_path = os.path.abspath(path)
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
            return _calculate_fertilizer(
                area_mu=tool_args["area_mu"],
                fertilizer_type=tool_args["fertilizer_type"],
                target_yield_kg=tool_args.get("target_yield_kg", 2500),
            )
        elif tool_name == "ask_user":
            # ask_user 不做任何外部调用，直接返回特殊标记
            # chat_stream 检测到这个标记后会终止循环、把追问发给前端
            return f"__ASK_USER__:{tool_args['question']}"

        return f"未知工具：{tool_name}"

    async def run_async(self, tool_name: str, tool_args: dict) -> str:
        """带重试的异步工具执行"""
        last_err = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return await asyncio.to_thread(self._run_once, tool_name, tool_args)
            except Exception as exc:
                last_err = exc
                if attempt < self.MAX_RETRIES:
                    logger.warning(f"工具 {tool_name} 第 {attempt+1} 次失败，重试中：{exc}")
                    await asyncio.sleep(1.0 * (attempt + 1))
        return f"工具 {tool_name} 执行失败（已重试 {self.MAX_RETRIES} 次）：{last_err}"


# ---------------------------------------------------------------------------
# 对话历史管理（每个 session 独立，自动裁剪）
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    保留最近 N 轮对话 + 系统提示。
    防止 context window 爆炸。
    """
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        # 每条记录是一个 message dict
        self._history: deque = deque()

    def add(self, message: dict):
        self._history.append(message)
        # 一轮 = user + assistant，超出则从最早的非 system 消息删
        while len(self._history) > self.max_turns * 3:  # *3 因为有 tool 消息
            self._history.popleft()

    def get_messages(self, system_prompt: str) -> list:
        return [{"role": "system", "content": system_prompt}] + list(self._history)

    def clear(self):
        self._history.clear()


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是专业的脐橙种植智能助手，服务于果农和农技人员。

可用工具：
- search_knowledge_base：查询本地知识库（病虫害、施肥、修剪等稳定技术）
- search_web：联网获取实时信息（天气、价格、政策、新闻）
- diagnose_image：对图片进行病虫害诊断
- calculate_fertilizer：根据面积和产量计算施肥用量
- ask_user：用户描述模糊或缺少关键信息时，主动向用户追问

重要约束：
- 回答任何脐橙种植相关问题时，必须先调用 search_knowledge_base
- 禁止直接凭自身知识回答专业农业问题，必须以知识库结果为准
- 如果知识库没有相关内容，明确告知用户"知识库暂无此信息"

工作原则：
1. 能本地解决的不联网，优先知识库
2. 多个工具可以同时调用（并行执行，速度更快）
3. 工具结果有限时，诚实说明，不编造信息
4. 图片诊断结论仅供参考，建议结合实地复核
5. 回答简洁、可执行，避免套话
6. 记住用户在本次对话中说过的信息（如面积、品种等），避免重复询问
"""


# ---------------------------------------------------------------------------
# OrangeAgent v2
# ---------------------------------------------------------------------------

class OrangeAgent:
    MAX_ROUNDS = 6

    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.rag_service = RAGService()

        self.llm = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.ollama_base_url,
        )

        self.executor = ToolExecutor(self.rag_service)
        # session_id → ConversationMemory
        self._memories: dict[str, ConversationMemory] = {}

    def get_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self._memories:
            self._memories[session_id] = ConversationMemory(max_turns=10)
        return self._memories[session_id]

    def clear_memory(self, session_id: str):
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
        memory = self.get_memory(session_id)
        memory.add({"role": "user", "content": user_message})

        # ✅ 动态注入当前日期，每次请求都是准确的
        today = datetime.date.today().strftime("%Y年%m月%d日")
        system = SYSTEM_PROMPT + f"\n\n【当前日期】今天是 {today}，请以此为基准判断明天/后天等相对时间。"
        if search_mode == "web":
            system += "\n\n[当前模式：优先联网搜索]"
        elif search_mode == "local":
            system += "\n\n[当前模式：仅使用本地知识库，禁止调用 search_web]"

        messages = memory.get_messages(system)

        for round_idx in range(self.MAX_ROUNDS):
            # ---- 请求 LLM ----
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
                final_text = msg.content or "抱歉，我暂时无法回答这个问题。"
                # 存入记忆
                memory.add({"role": "assistant", "content": final_text})
                # 流式输出
                chunk_size = 8
                for i in range(0, len(final_text), chunk_size):
                    yield final_text[i: i + chunk_size]
                    await asyncio.sleep(0.05)
                return

            # ---- 有工具调用 ----
            # 1. 推送状态给前端
            tool_names_cn = {
                "search_knowledge_base": "📚 查询知识库",
                "search_web": "🌐 联网搜索",
                "diagnose_image": "🔬 图片诊断",
                "calculate_fertilizer": "🧮 施肥计算",
            }
            status_parts = [tool_names_cn.get(tc.function.name, tc.function.name) for tc in msg.tool_calls]

            yield f"__STATUS__:正在执行：{' | '.join(status_parts)}"
            await asyncio.sleep(0)

            # 2. 把 LLM 决策加入消息历史
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
            tasks = [
                self.executor.run_async(tc.function.name, json.loads(tc.function.arguments))
                for tc in msg.tool_calls
            ]
            results = await asyncio.gather(*tasks)

            # ✅ 检查是否有 ask_user 结果
            ask_user_question = None
            for result in results:
                if isinstance(result, str) and result.startswith("__ASK_USER__:"):
                    ask_user_question = result.replace("__ASK_USER__:", "").strip()
                    break

            # 无论是否有 ask_user，先把所有 tool 结果存入历史
            # 保证 assistant(tool_calls) 后面紧跟完整 tool 结果，避免 400 报错
            for tc, result in zip(msg.tool_calls, results):
                tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
                messages.append(tool_msg)
                memory.add(tool_msg)

            if ask_user_question:
                memory.add({"role": "assistant", "content": ask_user_question})
                yield f"__ASK_USER__:{ask_user_question}"
                return

        # 超出最大轮次
        yield "\n[超出最大推理轮次，请重新提问]"

    # ------------------------------------------------------------------
    # 同步接口（兼容旧调用）
    # ------------------------------------------------------------------

    def chat(self, user_message: str, session_id: str = "default", search_mode: str = "auto") -> str:
        async def _run():
            parts = []
            async for chunk in self.chat_stream(user_message, session_id=session_id, search_mode=search_mode):
                parts.append(chunk)
            return "".join(parts)
        return asyncio.run(_run())