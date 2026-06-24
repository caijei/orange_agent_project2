"""
agent_service.py 真实 Agent 调用链鲁棒性测评

这个脚本测试的是前端实际使用的 Agent：
  api_server.py -> agent_service.OrangeAgent.chat_stream()

它会收集流式输出中的：
  - __STATUS__: 正在执行的工具
  - __ASK_USER__: Agent 主动追问
  - 最终回答文本

并生成：
  - CSV 明细表
  - HTML 可视化报告
  - 可选 JSON 结果

运行示例：
  python evaluate_agent_service_robustness.py --backend "D:\\实验\\orange_agent_project\\backend"
  python evaluate_agent_service_robustness.py --backend "D:\\实验\\orange_agent_project\\backend" --search-mode local
  python evaluate_agent_service_robustness.py --backend "D:\\实验\\orange_agent_project\\backend" --output agent_service_results.json
"""

import argparse
import asyncio
import csv
import html
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROBUSTNESS_DATASET = [
    # 口语化表达
    {
        "query": "我家橙子叶子发黄咋办",
        "case_type": "口语化",
        "expected_tools": ["search_knowledge_base", "ask_user"],
        "answer_keywords": ["黄", "叶", "原因", "防治", "管理", "症状"],
        "expected_behavior": "能理解口语化表达，并给出叶片发黄的可能原因和处理建议。",
    },
    {
        "query": "橙子树叶子卷起来了是不是虫子咬的",
        "case_type": "口语化",
        "expected_tools": ["search_knowledge_base", "ask_user"],
        "answer_keywords": ["卷叶", "虫", "症状", "检查", "防治"],
        "expected_behavior": "能把橙子树映射到脐橙栽培问题，并给出排查思路。",
    },
    {
        "query": "我家脐橙果子上长黑点点了咋弄",
        "case_type": "口语化",
        "expected_tools": ["search_knowledge_base", "ask_user"],
        "answer_keywords": ["黑点", "病", "果实", "防治", "管理"],
        "expected_behavior": "能理解口语描述，围绕果实黑点给出可能病害和管理建议。",
    },
    {
        "query": "树上好多小虫子，叶子也黏黏的，怎么办",
        "case_type": "口语化",
        "expected_tools": ["search_knowledge_base", "ask_user"],
        "answer_keywords": ["虫", "蚜", "蚧", "煤烟", "防治", "叶"],
        "expected_behavior": "能从黏叶、小虫等口语线索联想到刺吸式害虫或煤烟病。",
    },
    {
        "query": "橙子最近掉果有点厉害，正常不",
        "case_type": "口语化",
        "expected_tools": ["search_knowledge_base", "ask_user"],
        "answer_keywords": ["落果", "掉果", "原因", "水肥", "管理"],
        "expected_behavior": "能识别为落果/生理落果问题，并给出排查和管理建议。",
    },

    # 错别字、同音字、非标准病虫害名称
    {
        "query": "脐橙黄龙并怎么治",
        "case_type": "错别字",
        "expected_tools": ["search_knowledge_base"],
        "answer_keywords": ["黄龙病", "木虱", "防治", "病株", "检疫"],
        "expected_behavior": "能容忍“黄龙并”错别字，识别为黄龙病问题。",
    },
    {
        "query": "脐橙溃杨病会不会传染",
        "case_type": "错别字",
        "expected_tools": ["search_knowledge_base"],
        "answer_keywords": ["溃疡病", "传播", "风雨", "伤口", "防治"],
        "expected_behavior": "能容忍“溃杨病”别字，识别为溃疡病传播问题。",
    },
    {
        "query": "脐橙碳蛆病叶子上有什么表现",
        "case_type": "错别字",
        "expected_tools": ["search_knowledge_base"],
        "answer_keywords": ["炭疽病", "叶", "病斑", "症状", "防治"],
        "expected_behavior": "能把“碳蛆病”纠正为炭疽病，并回答症状。",
    },
    {
        "query": "红知蛛打什么药比较好",
        "case_type": "错别字",
        "expected_tools": ["search_knowledge_base"],
        "answer_keywords": ["红蜘蛛", "螨", "防治", "药", "叶片"],
        "expected_behavior": "能识别“红知蛛”为红蜘蛛，并给出防治建议。",
    },
    {
        "query": "潜叶鹅危害新梢怎么处理",
        "case_type": "错别字",
        "expected_tools": ["search_knowledge_base"],
        "answer_keywords": ["潜叶蛾", "新梢", "防治", "幼虫", "药"],
        "expected_behavior": "能识别“潜叶鹅”为潜叶蛾，围绕新梢危害回答。",
    },

    # 省略上下文、指代不明、信息不足
    {
        "query": "那现在要不要打药",
        "case_type": "省略上下文",
        "expected_tools": ["ask_user", "search_knowledge_base"],
        "answer_keywords": ["症状", "天气", "病虫害", "需要", "建议", "补充"],
        "expected_behavior": "问题信息不足时，应结合上下文或提示需要补充病虫害、天气、发病程度等信息。",
    },
    {
        "query": "这个病要怎么处理",
        "case_type": "省略上下文",
        "expected_tools": ["ask_user", "search_knowledge_base"],
        "answer_keywords": ["症状", "图片", "部位", "补充", "判断"],
        "expected_behavior": "缺少具体病害名称时，不应武断诊断，应提示补充症状或图片。",
    },
    {
        "query": "这样还能不能留果",
        "case_type": "省略上下文",
        "expected_tools": ["ask_user", "search_knowledge_base"],
        "answer_keywords": ["留果", "疏果", "树势", "果量", "判断"],
        "expected_behavior": "能识别为疏果/留果问题，并提示需要树势、结果量等信息。",
    },
    {
        "query": "现在喷这个可以吗",
        "case_type": "省略上下文",
        "expected_tools": ["ask_user", "search_knowledge_base", "get_weather"],
        "answer_keywords": ["药", "天气", "浓度", "时期", "说明", "补充"],
        "expected_behavior": "缺少药剂、天气、对象时，应谨慎回答并提示补充关键信息。",
    },
    {
        "query": "还要不要再施一次",
        "case_type": "省略上下文",
        "expected_tools": ["ask_user", "search_knowledge_base"],
        "answer_keywords": ["施肥", "树龄", "树势", "时期", "用量"],
        "expected_behavior": "能识别为施肥追问，提示需要树龄、时期、上次施肥情况。",
    },

    # 混合问题：实时因素 + 农技知识
    {
        "query": "赣州明天下雨，能不能防木虱",
        "case_type": "混合问题",
        "expected_tools": ["get_weather", "search_web", "search_knowledge_base"],
        "answer_keywords": ["赣州", "下雨", "木虱", "防治", "打药"],
        "expected_behavior": "能识别天气实时性，同时回答木虱防治是否适合打药。",
    },
    {
        "query": "信丰这周降温，幼树要不要盖膜",
        "case_type": "混合问题",
        "expected_tools": ["get_weather", "search_web", "search_knowledge_base"],
        "answer_keywords": ["信丰", "降温", "幼树", "防寒", "保温"],
        "expected_behavior": "能识别天气实时性，同时给出幼树防寒建议。",
    },
    {
        "query": "明天有雨，今天喷溃疡病药来得及吗",
        "case_type": "混合问题",
        "expected_tools": ["get_weather", "search_web", "search_knowledge_base"],
        "answer_keywords": ["雨", "溃疡病", "喷药", "药效", "天气"],
        "expected_behavior": "能结合天气和病害防治，提醒避开降雨或考虑药效保持时间。",
    },
    {
        "query": "最近温度高，红蜘蛛是不是更容易爆发",
        "case_type": "混合问题",
        "expected_tools": ["get_weather", "search_web", "search_knowledge_base"],
        "answer_keywords": ["温度", "红蜘蛛", "高温", "干旱", "防治"],
        "expected_behavior": "能识别天气因素和虫害发生条件的结合问题。",
    },
    {
        "query": "下周要采果了，这几天还能不能打保鲜药",
        "case_type": "混合问题",
        "expected_tools": ["search_knowledge_base", "search_web"],
        "answer_keywords": ["采果", "安全间隔期", "药", "采收", "说明"],
        "expected_behavior": "能识别采前用药风险，并提醒安全间隔期和规范用药。",
    },

    # 无关或弱相关问题
    {
        "query": "帮我写首歌",
        "case_type": "无关问题",
        "expected_tools": [],
        "answer_keywords": ["可以", "脐橙", "农业", "种植", "歌"],
        "expected_behavior": "能处理非农技问题，最好礼貌回应或引导回脐橙场景。",
    },
    {
        "query": "帮我写一篇关于春天的作文",
        "case_type": "无关问题",
        "expected_tools": [],
        "answer_keywords": ["可以", "脐橙", "农业", "写", "作文"],
        "expected_behavior": "能处理泛化写作需求，或引导用户说明是否需要脐橙主题。",
    },
    {
        "query": "你会不会做数学题",
        "case_type": "无关问题",
        "expected_tools": [],
        "answer_keywords": ["可以", "会", "问题", "脐橙", "帮助"],
        "expected_behavior": "能作为闲聊/能力说明处理，不应误走联网搜索。",
    },
    {
        "query": "给我讲个笑话",
        "case_type": "无关问题",
        "expected_tools": [],
        "answer_keywords": ["可以", "脐橙", "笑话", "种植", "农业"],
        "expected_behavior": "能轻量回应或转成脐橙相关表达，不应编造农技事实。",
    },
    {
        "query": "帮我取一个果园名字",
        "case_type": "无关问题",
        "expected_tools": [],
        "answer_keywords": ["果园", "名字", "脐橙", "可以", "建议"],
        "expected_behavior": "虽非农技问答，但与果园弱相关，可正常创意回答。",
    },

    # 超出本地知识库或强实时问题
    {
        "query": "今年某地脐橙收购价多少",
        "case_type": "超出知识库",
        "expected_tools": ["search_web"],
        "answer_keywords": ["价格", "收购价", "今年", "地区", "实时"],
        "expected_behavior": "能识别价格属于实时信息，应走联网搜索或提示需要联网/具体地区。",
    },
    {
        "query": "今天赣南脐橙批发价是多少",
        "case_type": "超出知识库",
        "expected_tools": ["search_web"],
        "answer_keywords": ["今天", "批发价", "价格", "赣南", "实时"],
        "expected_behavior": "价格是实时市场信息，应走联网搜索或提示需要实时数据。",
    },
    {
        "query": "2026年脐橙出口政策有什么变化",
        "case_type": "超出知识库",
        "expected_tools": ["search_web"],
        "answer_keywords": ["2026", "政策", "出口", "变化", "最新"],
        "expected_behavior": "政策变化属于实时信息，应走联网搜索。",
    },
    {
        "query": "最近哪个电商平台卖脐橙比较火",
        "case_type": "超出知识库",
        "expected_tools": ["search_web"],
        "answer_keywords": ["最近", "电商", "平台", "销量", "市场"],
        "expected_behavior": "电商销售热度属于动态市场信息，应走联网搜索。",
    },
    {
        "query": "今年赣州脐橙节什么时候开始",
        "case_type": "超出知识库",
        "expected_tools": ["search_web"],
        "answer_keywords": ["今年", "赣州", "脐橙节", "时间", "活动"],
        "expected_behavior": "活动时间属于最新信息，应走联网搜索。",
    },
]


STATUS_TOOL_MAP = {
    "查询知识库": "search_knowledge_base",
    "联网搜索": "search_web",
    "图片诊断": "diagnose_image",
    "施肥计算": "calculate_fertilizer",
    "追问用户": "ask_user",
    "get_weather": "get_weather",
    "search_knowledge_base": "search_knowledge_base",
    "search_web": "search_web",
    "diagnose_image": "diagnose_image",
    "calculate_fertilizer": "calculate_fertilizer",
    "ask_user": "ask_user",
}


@dataclass
class AgentServiceResult:
    index: int
    query: str
    case_type: str
    expected_tools: list[str]
    used_tools: list[str]
    tool_hit: bool
    answer_hit: bool
    robustness_pass: bool
    latency_s: float
    answer_preview: str
    status_events: list[str]
    ask_user: str = ""
    error: str = ""


def load_agent(backend_dir: str):
    backend_path = Path(backend_dir).resolve()
    if not backend_path.exists():
        raise FileNotFoundError(f"backend 目录不存在: {backend_path}")

    sys.path.insert(0, str(backend_path))
    from agent_service import OrangeAgent

    return OrangeAgent()


def contains_any(text: str, keywords: list[str]) -> bool:
    if not text:
        return False
    return any(keyword in text for keyword in keywords)


def parse_status_tools(status_text: str) -> list[str]:
    tools = []
    for label, tool_name in STATUS_TOOL_MAP.items():
        if label in status_text and tool_name not in tools:
            tools.append(tool_name)
    return tools


async def run_one(agent: Any, query: str, session_id: str, search_mode: str) -> dict[str, Any]:
    chunks = []
    status_events = []
    used_tools = []
    ask_user = ""

    async for chunk in agent.chat_stream(query, session_id=session_id, search_mode=search_mode):
        if chunk.startswith("__STATUS__:"):
            status = chunk.replace("__STATUS__:", "", 1).strip()
            status_events.append(status)
            for tool in parse_status_tools(status):
                if tool not in used_tools:
                    used_tools.append(tool)
            continue

        if chunk.startswith("__ASK_USER__:"):
            ask_user = chunk.replace("__ASK_USER__:", "", 1).strip()
            if "ask_user" not in used_tools:
                used_tools.append("ask_user")
            chunks.append(ask_user)
            continue

        chunks.append(chunk)

    return {
        "answer": "".join(chunks),
        "used_tools": used_tools,
        "status_events": status_events,
        "ask_user": ask_user,
    }


async def evaluate_async(backend_dir: str, search_mode: str) -> list[AgentServiceResult]:
    agent = load_agent(backend_dir)
    results = []

    for i, sample in enumerate(ROBUSTNESS_DATASET, start=1):
        start = time.time()
        session_id = f"robustness_eval_{i:02d}_{int(start)}"
        answer = ""
        used_tools = []
        status_events = []
        ask_user = ""
        error = ""

        try:
            output = await run_one(agent, sample["query"], session_id=session_id, search_mode=search_mode)
            answer = output["answer"]
            used_tools = output["used_tools"]
            status_events = output["status_events"]
            ask_user = output["ask_user"]
        except Exception as exc:
            error = str(exc)

        latency = round(time.time() - start, 4)
        expected_tools = sample["expected_tools"]

        if expected_tools:
            tool_hit = any(tool in used_tools for tool in expected_tools)
        else:
            tool_hit = not used_tools

        answer_hit = contains_any(answer, sample["answer_keywords"])

        results.append(
            AgentServiceResult(
                index=i,
                query=sample["query"],
                case_type=sample["case_type"],
                expected_tools=expected_tools,
                used_tools=used_tools,
                tool_hit=tool_hit,
                answer_hit=answer_hit,
                robustness_pass=tool_hit and answer_hit,
                latency_s=latency,
                answer_preview=answer.replace("\n", " ")[:220],
                status_events=status_events,
                ask_user=ask_user,
                error=error,
            )
        )

    return results


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _summary(results: list[AgentServiceResult]) -> dict[str, float | int]:
    total = len(results)
    tool_hits = sum(1 for r in results if r.tool_hit)
    answer_hits = sum(1 for r in results if r.answer_hit)
    pass_hits = sum(1 for r in results if r.robustness_pass)
    avg_latency = sum(r.latency_s for r in results) / max(total, 1)
    return {
        "total": total,
        "tool_hits": tool_hits,
        "answer_hits": answer_hits,
        "pass_hits": pass_hits,
        "tool_accuracy": tool_hits / max(total, 1),
        "answer_accuracy": answer_hits / max(total, 1),
        "pass_rate": pass_hits / max(total, 1),
        "avg_latency": avg_latency,
    }


def _category_rows(results: list[AgentServiceResult]) -> list[dict[str, Any]]:
    by_type = defaultdict(list)
    for result in results:
        by_type[result.case_type].append(result)

    rows = []
    for case_type, items in by_type.items():
        total = len(items)
        tool_hits = sum(1 for r in items if r.tool_hit)
        answer_hits = sum(1 for r in items if r.answer_hit)
        pass_hits = sum(1 for r in items if r.robustness_pass)
        rows.append({
            "case_type": case_type,
            "total": total,
            "tool_accuracy": tool_hits / max(total, 1),
            "answer_accuracy": answer_hits / max(total, 1),
            "pass_rate": pass_hits / max(total, 1),
            "avg_latency": sum(r.latency_s for r in items) / max(total, 1),
        })
    return rows


def print_report(results: list[AgentServiceResult], search_mode: str):
    summary = _summary(results)
    print("\n========== agent_service.py 鲁棒性测评 ==========")
    print(f"调用链: agent_service.OrangeAgent.chat_stream()")
    print(f"search_mode: {search_mode}")
    print(f"样本数: {summary['total']}")
    print(f"工具命中率: {_pct(summary['tool_accuracy'])}")
    print(f"答案关键词命中率: {_pct(summary['answer_accuracy'])}")
    print(f"鲁棒性通过率: {_pct(summary['pass_rate'])}")
    print(f"平均耗时: {summary['avg_latency']:.2f}s")

    print("\n========== 分类表现 ==========")
    for row in _category_rows(results):
        print(
            f"{row['case_type']:<8} "
            f"工具={_pct(row['tool_accuracy'])} "
            f"答案={_pct(row['answer_accuracy'])} "
            f"通过={_pct(row['pass_rate'])} "
            f"耗时={row['avg_latency']:.2f}s"
        )

    print("\n========== 明细 ==========")
    for r in results:
        mark = "✓" if r.robustness_pass else "✗"
        expected = "/".join(r.expected_tools) if r.expected_tools else "无工具"
        used = "/".join(r.used_tools) if r.used_tools else "无工具"
        print(
            f"[{mark}] #{r.index:02d} {r.case_type:<6} "
            f"期望={expected:<36} 实际={used:<36} "
            f"工具={'✓' if r.tool_hit else '✗'} "
            f"答案={'✓' if r.answer_hit else '✗'} "
            f"耗时={r.latency_s:.2f}s"
        )
        print(f"    问题: {r.query}")
        if r.answer_preview:
            print(f"    回答: {r.answer_preview}")
        if r.status_events:
            print(f"    状态: {' | '.join(r.status_events)}")
        if r.error:
            print(f"    错误: {r.error}")


def save_json(results: list[AgentServiceResult], output_path: str):
    summary = _summary(results)
    payload = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 结果已保存: {Path(output_path).resolve()}")


def export_csv_report(results: list[AgentServiceResult], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "序号", "问题类型", "问题", "期望工具", "实际工具",
            "工具命中", "答案命中", "是否通过", "耗时(s)",
            "状态事件", "追问内容", "回答摘要", "错误",
        ])
        for r in results:
            writer.writerow([
                r.index,
                r.case_type,
                r.query,
                "/".join(r.expected_tools) if r.expected_tools else "无工具",
                "/".join(r.used_tools) if r.used_tools else "无工具",
                "是" if r.tool_hit else "否",
                "是" if r.answer_hit else "否",
                "是" if r.robustness_pass else "否",
                r.latency_s,
                " | ".join(r.status_events),
                r.ask_user,
                r.answer_preview,
                r.error,
            ])


def export_html_report(results: list[AgentServiceResult], html_path: Path, search_mode: str):
    html_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _summary(results)
    category_rows = _category_rows(results)
    max_total = max([row["total"] for row in category_rows] or [1])

    category_chart = []
    for row in category_rows:
        pass_width = row["pass_rate"] * 100
        tool_width = row["tool_accuracy"] * 100
        label = html.escape(str(row["case_type"]))
        category_chart.append(f"""
        <div class="chart-row">
          <div class="chart-label">{label}</div>
          <div class="bar-wrap">
            <div class="bar tool" style="width:{tool_width:.1f}%"></div>
            <div class="bar pass" style="width:{pass_width:.1f}%"></div>
          </div>
          <div class="chart-value">{_pct(row["pass_rate"])}</div>
        </div>
        """)

    type_distribution = []
    for row in category_rows:
        width = row["total"] / max_total * 100
        label = html.escape(str(row["case_type"]))
        type_distribution.append(f"""
        <div class="dist-row">
          <span>{label}</span>
          <div class="dist-bar"><i style="width:{width:.1f}%"></i></div>
          <b>{row["total"]}</b>
        </div>
        """)

    detail_rows = []
    for r in results:
        status = "pass" if r.robustness_pass else "fail"
        expected = "/".join(r.expected_tools) if r.expected_tools else "无工具"
        used = "/".join(r.used_tools) if r.used_tools else "无工具"
        detail_rows.append(f"""
        <tr class="{status}">
          <td>{r.index}</td>
          <td>{html.escape(r.case_type)}</td>
          <td class="question">{html.escape(r.query)}</td>
          <td>{html.escape(expected)}</td>
          <td>{html.escape(used)}</td>
          <td>{'命中' if r.tool_hit else '未命中'}</td>
          <td>{'命中' if r.answer_hit else '未命中'}</td>
          <td>{'通过' if r.robustness_pass else '未通过'}</td>
          <td>{r.latency_s:.2f}</td>
          <td class="answer">{html.escape(r.answer_preview)}</td>
        </tr>
        """)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>agent_service.py 鲁棒性测评报告</title>
  <style>
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      color: #243027;
      background: #f6f8f4;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-end;
      border-bottom: 3px solid #6d9f3f;
      padding-bottom: 16px;
      margin-bottom: 22px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    .meta {{ color: #61705d; font-size: 13px; text-align: right; line-height: 1.8; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #dfe7da;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 8px 22px rgba(43, 62, 36, 0.06);
    }}
    .card span {{ display: block; color: #5c6d57; font-size: 13px; margin-bottom: 10px; }}
    .card strong {{ display: block; font-size: 30px; color: #2f6f3e; line-height: 1.1; }}
    .card small {{ display: block; color: #7a8875; margin-top: 8px; }}
    .grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-bottom: 20px;
    }}
    .panel {{
      background: #fff;
      border: 1px solid #dfe7da;
      border-radius: 8px;
      padding: 18px;
    }}
    h2 {{ margin: 0 0 16px; font-size: 18px; }}
    .legend {{ display: flex; gap: 16px; color: #66755f; font-size: 12px; margin-bottom: 12px; }}
    .legend i {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      margin-right: 5px;
      vertical-align: -2px;
    }}
    .chart-row {{
      display: grid;
      grid-template-columns: 86px 1fr 70px;
      gap: 12px;
      align-items: center;
      margin: 11px 0;
    }}
    .chart-label {{ font-size: 14px; color: #42513e; }}
    .bar-wrap {{
      height: 20px;
      border-radius: 6px;
      background: #eef3eb;
      position: relative;
      overflow: hidden;
    }}
    .bar {{ position: absolute; top: 0; bottom: 0; left: 0; border-radius: 6px; }}
    .bar.tool {{ background: rgba(76, 132, 177, 0.35); }}
    .bar.pass {{
      background: rgba(93, 156, 73, 0.82);
      height: 10px;
      top: 5px;
      bottom: auto;
      z-index: 2;
    }}
    .chart-value {{ text-align: right; color: #2f6f3e; font-weight: 700; }}
    .dist-row {{
      display: grid;
      grid-template-columns: 76px 1fr 28px;
      gap: 10px;
      align-items: center;
      margin: 12px 0;
      font-size: 14px;
    }}
    .dist-bar {{ height: 16px; background: #eef3eb; border-radius: 5px; overflow: hidden; }}
    .dist-bar i {{ display: block; height: 100%; background: #d9a441; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid #dfe7da;
      border-radius: 8px;
      overflow: hidden;
      font-size: 13px;
    }}
    th {{
      background: #edf4e9;
      color: #31422d;
      text-align: left;
      padding: 10px 9px;
      border-bottom: 1px solid #d6e1d1;
      white-space: nowrap;
    }}
    td {{ padding: 9px; border-bottom: 1px solid #edf1ea; vertical-align: top; }}
    tr.pass td:first-child {{ border-left: 4px solid #5d9c49; }}
    tr.fail td:first-child {{ border-left: 4px solid #c65d4b; }}
    .question {{ min-width: 170px; font-weight: 600; }}
    .answer {{ max-width: 330px; color: #53624f; }}
    @media (max-width: 860px) {{
      .cards, .grid {{ grid-template-columns: 1fr; }}
      .header {{ display: block; }}
      .meta {{ text-align: left; margin-top: 10px; }}
      .page {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="header">
      <div>
        <h1>agent_service.py 鲁棒性测评报告</h1>
        <div>调用链：api_server.py -> agent_service.OrangeAgent.chat_stream()</div>
      </div>
      <div class="meta">
        <div>生成时间：{html.escape(now)}</div>
        <div>search_mode：{html.escape(search_mode)}</div>
      </div>
    </section>

    <section class="cards">
      <div class="card"><span>样本数</span><strong>{summary["total"]}</strong><small>测试问题总量</small></div>
      <div class="card"><span>工具命中率</span><strong>{_pct(summary["tool_accuracy"])}</strong><small>{summary["tool_hits"]}/{summary["total"]}</small></div>
      <div class="card"><span>答案关键词命中率</span><strong>{_pct(summary["answer_accuracy"])}</strong><small>{summary["answer_hits"]}/{summary["total"]}</small></div>
      <div class="card"><span>鲁棒性通过率</span><strong>{_pct(summary["pass_rate"])}</strong><small>{summary["pass_hits"]}/{summary["total"]}</small></div>
      <div class="card"><span>平均耗时</span><strong>{summary["avg_latency"]:.2f}s</strong><small>单条样本平均</small></div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>分类通过率</h2>
        <div class="legend">
          <span><i style="background:#5d9c49"></i>鲁棒性通过率</span>
          <span><i style="background:rgba(76,132,177,.45)"></i>工具命中率</span>
        </div>
        {''.join(category_chart)}
      </div>
      <div class="panel">
        <h2>样本类型分布</h2>
        {''.join(type_distribution)}
      </div>
    </section>

    <section>
      <h2>样本明细表</h2>
      <table>
        <thead>
          <tr>
            <th>#</th><th>类型</th><th>问题</th><th>期望工具</th><th>实际工具</th>
            <th>工具</th><th>答案</th><th>结论</th><th>耗时</th><th>回答摘要</th>
          </tr>
        </thead>
        <tbody>{''.join(detail_rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")


def export_reports(results: list[AgentServiceResult], report_dir: str, search_mode: str):
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "agent_service_robustness_results.csv"
    html_path = output_dir / "agent_service_robustness_report.html"

    export_csv_report(results, csv_path)
    export_html_report(results, html_path, search_mode=search_mode)

    print(f"\nCSV 表格已生成: {csv_path.resolve()}")
    print(f"HTML 可视化报告已生成: {html_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="agent_service.py 真实 Agent 鲁棒性测评")
    parser.add_argument(
        "--backend",
        default=os.getenv("ORANGE_BACKEND_DIR", r"D:\实验\orange_agent_project\backend"),
        help="项目 backend 目录路径",
    )
    parser.add_argument(
        "--search-mode",
        choices=["auto", "web", "local"],
        default="auto",
        help="与前端一致的 search_mode 参数",
    )
    parser.add_argument(
        "--output",
        default="",
        help="可选：保存 JSON 结果文件路径",
    )
    parser.add_argument(
        "--report-dir",
        default="agent_service_robustness_report",
        help="可选：生成 CSV 表格和 HTML 可视化报告的目录；传入空字符串可关闭",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = asyncio.run(evaluate_async(args.backend, search_mode=args.search_mode))
    print_report(results, search_mode=args.search_mode)

    if args.output:
        save_json(results, args.output)

    if args.report_dir:
        export_reports(results, args.report_dir, search_mode=args.search_mode)


if __name__ == "__main__":
    main()
