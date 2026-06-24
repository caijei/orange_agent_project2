"""
Agent 工具路由准确率测评脚本

用途：
  测试用户问题是否被 Router 正确分配到对应工具：
    - knowledge_qa
    - web_search
    - memory_set
    - memory_query
    - chat
    - command

运行示例：
  python evaluate_agent_routing.py --backend "D:\\实验\\orange_agent_project\\backend"
  python evaluate_agent_routing.py --backend "D:\\实验\\orange_agent_project\\backend" --rules-only
  python evaluate_agent_routing.py --backend "D:\\实验\\orange_agent_project\\backend" --output routing_results.json

说明：
  默认调用 Router.route()，更接近真实系统路由。
  如果本地 LLM/API 不稳定，可加 --rules-only，只测规则路由；规则未命中时默认 knowledge_qa。
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


ROUTING_DATASET = [
    # 知识库问答：稳定农业知识
    {
        "query": "脐橙黄龙病的主要症状是什么？",
        "expected_tool": "knowledge_qa",
        "category": "知识库问答",
    },
    {
        "query": "脐橙木虱怎么防治？",
        "expected_tool": "knowledge_qa",
        "category": "知识库问答",
    },
    {
        "query": "赣南脐橙适合什么土壤种植？",
        "expected_tool": "knowledge_qa",
        "category": "知识库问答",
    },
    {
        "query": "脐橙采后应该怎么储藏保鲜？",
        "expected_tool": "knowledge_qa",
        "category": "知识库问答",
    },
    {
        "query": "幼树第一年怎么施肥？",
        "expected_tool": "knowledge_qa",
        "category": "知识库问答",
    },

    # 实时信息：天气、价格、政策、市场
    {
        "query": "赣州今天适合给脐橙打药吗？",
        "expected_tool": "web_search",
        "category": "实时信息",
    },
    {
        "query": "今天赣州天气怎么样？",
        "expected_tool": "web_search",
        "category": "实时信息",
    },
    {
        "query": "今年脐橙收购价格是多少？",
        "expected_tool": "web_search",
        "category": "实时信息",
    },
    {
        "query": "最近有没有脐橙种植补贴政策？",
        "expected_tool": "web_search",
        "category": "实时信息",
    },
    {
        "query": "现在赣南脐橙市场行情怎么样？",
        "expected_tool": "web_search",
        "category": "实时信息",
    },

    # 记忆写入
    {
        "query": "记住我在赣州种纽荷尔脐橙。",
        "expected_tool": "memory_set",
        "category": "记忆写入",
    },
    {
        "query": "我是李明，我家果园有20亩。",
        "expected_tool": "memory_set",
        "category": "记忆写入",
    },
    {
        "query": "以后回答我先给结论，再说原因。",
        "expected_tool": "memory_set",
        "category": "记忆写入",
    },

    # 记忆查询
    {
        "query": "你还记得我在哪里种脐橙吗？",
        "expected_tool": "memory_query",
        "category": "记忆查询",
    },
    {
        "query": "我之前告诉过你我种的是什么品种吗？",
        "expected_tool": "memory_query",
        "category": "记忆查询",
    },
    {
        "query": "你记住了我的哪些信息？",
        "expected_tool": "memory_query",
        "category": "记忆查询",
    },

    # 闲聊
    {
        "query": "你好",
        "expected_tool": "chat",
        "category": "闲聊",
    },
    {
        "query": "谢谢你",
        "expected_tool": "chat",
        "category": "闲聊",
    },
    {
        "query": "你是谁？",
        "expected_tool": "chat",
        "category": "闲聊",
    },

    # 命令
    {
        "query": "清空记忆",
        "expected_tool": "command",
        "category": "命令",
    },
    {
        "query": "清空历史",
        "expected_tool": "command",
        "category": "命令",
    },
    {
        "query": "退出",
        "expected_tool": "command",
        "category": "命令",
    },

    # 容易混淆的真实用户表达
    {
        "query": "我这边明天下雨，还能不能防治木虱？",
        "expected_tool": "web_search",
        "category": "混合问题",
    },
    {
        "query": "叶子发黄是不是黄龙病？",
        "expected_tool": "knowledge_qa",
        "category": "混合问题",
    },
    {
        "query": "我在信丰，给我按当地情况回答。",
        "expected_tool": "memory_set",
        "category": "混合问题",
    },
]


@dataclass
class RoutingResult:
    index: int
    query: str
    category: str
    expected_tool: str
    predicted_tool: str
    correct: bool
    latency_s: float
    error: str = ""


def load_router(backend_dir: str):
    backend_path = Path(backend_dir).resolve()
    if not backend_path.exists():
        raise FileNotFoundError(f"backend 目录不存在: {backend_path}")

    sys.path.insert(0, str(backend_path))

    from rag_service import Router, VectorKnowledgeBase

    kb = VectorKnowledgeBase()
    return Router(kb)


def predict_tool(router, query: str, rules_only: bool) -> str:
    if rules_only:
        rule_result: Optional[str] = router._route_by_rules(query)
        return rule_result or "knowledge_qa"
    return router.route(query)


def evaluate(router, dataset: list[dict], rules_only: bool) -> list[RoutingResult]:
    results = []

    for i, sample in enumerate(dataset, start=1):
        start = time.time()
        predicted = ""
        error = ""

        try:
            predicted = predict_tool(router, sample["query"], rules_only)
        except Exception as exc:
            predicted = "ERROR"
            error = str(exc)

        latency = round(time.time() - start, 4)
        expected = sample["expected_tool"]

        results.append(
            RoutingResult(
                index=i,
                query=sample["query"],
                category=sample["category"],
                expected_tool=expected,
                predicted_tool=predicted,
                correct=(predicted == expected),
                latency_s=latency,
                error=error,
            )
        )

    return results


def print_report(results: list[RoutingResult], rules_only: bool):
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total else 0

    print("\n========== Agent 工具路由准确率测评 ==========")
    print(f"模式: {'仅规则路由' if rules_only else '真实 Router.route()'}")
    print(f"样本数: {total}")
    print(f"正确数: {correct}")
    print(f"总体准确率: {accuracy:.2%}")

    by_category = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)

    print("\n========== 分类准确率 ==========")
    for category, items in by_category.items():
        hit = sum(1 for r in items if r.correct)
        print(f"{category:<10} {hit:>2}/{len(items):<2}  {hit / len(items):.2%}")

    expected_counter = Counter(r.expected_tool for r in results)
    predicted_counter = Counter(r.predicted_tool for r in results)

    print("\n========== 工具分布 ==========")
    all_tools = sorted(set(expected_counter) | set(predicted_counter))
    for tool in all_tools:
        print(f"{tool:<14} expected={expected_counter[tool]:>2}  predicted={predicted_counter[tool]:>2}")

    print("\n========== 明细 ==========")
    for r in results:
        mark = "✓" if r.correct else "✗"
        print(
            f"[{mark}] #{r.index:02d} {r.category} | "
            f"期望={r.expected_tool:<12} 预测={r.predicted_tool:<12} "
            f"耗时={r.latency_s:.4f}s | {r.query}"
        )
        if r.error:
            print(f"    错误: {r.error}")

    failures = [r for r in results if not r.correct]
    if failures:
        print("\n========== 错误样本 ==========")
        for r in failures:
            print(f"#{r.index:02d} 期望 {r.expected_tool}，实际 {r.predicted_tool}：{r.query}")


def save_results(results: list[RoutingResult], output_path: str):
    payload = {
        "total": len(results),
        "correct": sum(1 for r in results if r.correct),
        "accuracy": round(sum(1 for r in results if r.correct) / max(len(results), 1), 4),
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Agent 工具路由准确率测评")
    parser.add_argument(
        "--backend",
        default=os.getenv("ORANGE_BACKEND_DIR", r"D:\实验\orange_agent_project\backend"),
        help="项目 backend 目录路径",
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="只测规则路由；规则未命中时按 knowledge_qa 处理",
    )
    parser.add_argument(
        "--output",
        default="",
        help="可选：保存 JSON 结果文件路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    router = load_router(args.backend)
    results = evaluate(router, ROUTING_DATASET, rules_only=args.rules_only)
    print_report(results, rules_only=args.rules_only)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
