"""
test_rag_ablation.py — RAG 增强方法消融测试
================================================
控制变量，逐步叠加各增强方法，对比召回率和准确率。

测试配置：
  C0  Baseline          直接向量检索
  C3  +HyDE+MQE         融合增强检索
  C6  Full(CRAG)        完整流程（含 CRAG 质量评估 + 按需联网）

评估指标：
  - Recall@k      top-k 文档中命中相关关键词的比例
  - Precision@k   返回文档中相关文档占比
  - MRR           首个相关文档排名的倒数均值
  - Answer Hit    生成答案是否包含期望关键词

运行方式：
  python test_rag_ablation.py
  python test_rag_ablation.py --k 5 --output results.json

依赖：需要和 rag_service.py 在同一目录，且 .env 已配置好。
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ── 把项目根目录加入 sys.path，确保能 import rag_service ──────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_service import VectorKnowledgeBase, KnowledgeQATool, MemoryStore


# =====================================================================
# 测试数据集
# 每条样本包含：
#   question         用户提问
#   doc_keywords     至少一个相关文档中应包含的关键词组（任一组命中即算相关）
#   answer_keywords  期望答案中应包含的关键词（至少命中一个）
#   note             备注（可选）
# =====================================================================

TEST_DATASET = [
    {
        "question": "脐橙黄龙病的主要症状是什么？",
        "doc_keywords": [["黄龙病", "症状"], ["黄化", "斑驳"]],
        "answer_keywords": ["黄化", "斑驳", "症状", "黄龙病"],
        "note": "病害识别基础题",
    },
    {
        "question": "脐橙木虱怎么防治？",
        "doc_keywords": [["木虱", "防治"], ["木虱", "药"]],
        "answer_keywords": ["木虱", "防治", "药"],
        "note": "虫害防治",
    },
    {
        "question": "脐橙什么时候施肥最好？",
        "doc_keywords": [["施肥", "时间"], ["施肥", "季节"], ["花前肥", "壮果肥"]],
        "answer_keywords": ["施肥", "月", "肥"],
        "note": "水肥管理",
    },
    {
        "question": "脐橙炭疽病如何识别和防治？",
        "doc_keywords": [["炭疽病", "防治"], ["炭疽", "症状"]],
        "answer_keywords": ["炭疽", "症状", "防治"],
        "note": "病害防治",
    },
    {
        "question": "赣南脐橙主要种植在哪些地区？",
        "doc_keywords": [["赣南", "种植"], ["赣州", "产区"], ["赣南", "脐橙"]],
        "answer_keywords": ["赣南", "赣州", "地区"],
        "note": "产区地理",
    },
    {
        "question": "脐橙采后如何保鲜储存？",
        "doc_keywords": [["保鲜", "储存"], ["采后", "贮藏"], ["冷藏", "脐橙"]],
        "answer_keywords": ["保鲜", "储存", "温度", "贮藏"],
        "note": "采后处理",
    },
    {
        "question": "脐橙溃疡病的传播途径是什么？",
        "doc_keywords": [["溃疡病", "传播"], ["溃疡", "途径"]],
        "answer_keywords": ["溃疡病", "传播", "途径"],
        "note": "病害传播",
    },
    {
        "question": "脐橙疏果的原则和时间？",
        "doc_keywords": [["疏果", "时间"], ["疏果", "原则"], ["疏果", "留果"]],
        "answer_keywords": ["疏果", "留果", "时间"],
        "note": "果实管理",
    },
    {
        "question": "如何预防脐橙冻害？",
        "doc_keywords": [["冻害", "预防"], ["霜冻", "防护"], ["冬季", "保温"]],
        "answer_keywords": ["冻害", "预防", "低温", "保温"],
        "note": "极端天气管理",
    },

    # ── 病害（续）────────────────────────────────────────────
    {
        "question": "脐橙脚腐病的发病条件和防治方法？",
        "doc_keywords": [["脚腐病", "防治"], ["脚腐", "症状"], ["根颈", "腐烂"]],
        "answer_keywords": ["脚腐病", "防治", "根颈", "腐烂"],
        "note": "根部病害",
    },
    {
        "question": "脐橙疮痂病是什么，怎么防治？",
        "doc_keywords": [["疮痂病", "防治"], ["疮痂", "症状"]],
        "answer_keywords": ["疮痂病", "症状", "防治"],
        "note": "真菌病害",
    },
    {
        "question": "脐橙煤烟病如何发生和防治？",
        "doc_keywords": [["煤烟病", "防治"], ["煤烟", "蚧壳虫"], ["煤污", "防治"]],
        "answer_keywords": ["煤烟病", "防治", "蚧", "煤污"],
        "note": "附生性病害",
    },
    {
        "question": "脐橙流胶病的症状和防治措施？",
        "doc_keywords": [["流胶病", "症状"], ["流胶", "防治"]],
        "answer_keywords": ["流胶病", "症状", "防治"],
        "note": "枝干病害",
    },
    {
        "question": "脐橙根腐病的防治方法？",
        "doc_keywords": [["根腐病", "防治"], ["根腐", "症状"]],
        "answer_keywords": ["根腐", "防治", "症状"],
        "note": "根部病害",
    },

    # ── 虫害（续）────────────────────────────────────────────
    {
        "question": "脐橙红蜘蛛怎么防治？",
        "doc_keywords": [["红蜘蛛", "防治"], ["螨", "防治"]],
        "answer_keywords": ["红蜘蛛", "防治", "螨", "药"],
        "note": "螨类害虫",
    },
    {
        "question": "脐橙蚧壳虫的种类和防治方法？",
        "doc_keywords": [["蚧壳虫", "防治"], ["介壳虫", "防治"], ["蚧", "种类"]],
        "answer_keywords": ["蚧壳虫", "防治", "介壳虫"],
        "note": "刺吸式害虫",
    },
    {
        "question": "脐橙潜叶蛾怎么防治，打药时机是什么？",
        "doc_keywords": [["潜叶蛾", "防治"], ["潜叶蛾", "打药"], ["潜叶蛾", "时机"]],
        "answer_keywords": ["潜叶蛾", "防治", "打药", "新梢"],
        "note": "叶部害虫",
    },
    {
        "question": "脐橙花蕾蛆的危害和防治？",
        "doc_keywords": [["花蕾蛆", "防治"], ["花蕾蛆", "危害"]],
        "answer_keywords": ["花蕾蛆", "防治", "危害"],
        "note": "花期害虫",
    },
    {
        "question": "脐橙天牛如何识别和防治？",
        "doc_keywords": [["天牛", "防治"], ["天牛", "危害"]],
        "answer_keywords": ["天牛", "防治", "枝干"],
        "note": "蛀干害虫",
    },
    {
        "question": "脐橙蚜虫对脐橙有什么危害，如何防治？",
        "doc_keywords": [["蚜虫", "防治"], ["蚜虫", "危害"]],
        "answer_keywords": ["蚜虫", "防治", "危害"],
        "note": "刺吸式害虫",
    },

    # ── 水肥管理（续）─────────────────────────────────────────
    {
        "question": "脐橙幼树如何施肥？",
        "doc_keywords": [["幼树", "施肥"], ["幼树", "肥料"]],
        "answer_keywords": ["幼树", "施肥", "氮", "肥料"],
        "note": "幼树管理",
    },
    {
        "question": "脐橙叶面喷肥用什么肥料，怎么喷？",
        "doc_keywords": [["叶面肥", "喷施"], ["叶面施肥", "脐橙"]],
        "answer_keywords": ["叶面肥", "喷施", "浓度"],
        "note": "叶面施肥",
    },
    {
        "question": "脐橙灌溉方式有哪些，如何合理灌水？",
        "doc_keywords": [["灌溉", "脐橙"], ["灌水", "方式"], ["滴灌", "脐橙"]],
        "answer_keywords": ["灌溉", "灌水", "方式", "滴灌"],
        "note": "水分管理",
    },
    {
        "question": "脐橙缺锌有什么表现，如何补救？",
        "doc_keywords": [["缺锌", "脐橙"], ["锌", "症状"]],
        "answer_keywords": ["缺锌", "症状", "补锌", "硫酸锌"],
        "note": "微量元素缺乏",
    },
    {
        "question": "脐橙缺镁叶片有什么症状？",
        "doc_keywords": [["缺镁", "脐橙"], ["镁", "叶片", "症状"]],
        "answer_keywords": ["缺镁", "叶片", "症状", "镁"],
        "note": "缺素症",
    },

    # ── 栽培管理────────────────────────────────────────────
    {
        "question": "脐橙建园时如何选地和整地？",
        "doc_keywords": [["建园", "选地"], ["建园", "整地"], ["脐橙园", "土地"]],
        "answer_keywords": ["建园", "选地", "整地", "坡度"],
        "note": "建园规划",
    },
    {
        "question": "脐橙的种植密度和株行距是多少？",
        "doc_keywords": [["种植密度", "脐橙"], ["株行距", "脐橙"], ["密植", "脐橙"]],
        "answer_keywords": ["密度", "株行距", "亩", "株"],
        "note": "种植规格",
    },
    {
        "question": "脐橙砧木有哪些，各有什么优缺点？",
        "doc_keywords": [["砧木", "脐橙"], ["枳壳", "砧木"], ["砧木", "优缺点"]],
        "answer_keywords": ["砧木", "枳", "优缺点"],
        "note": "砧木选择",
    },
    {
        "question": "脐橙的整形修剪原则和方法？",
        "doc_keywords": [["整形", "修剪"], ["修剪", "脐橙"], ["树形", "脐橙"]],
        "answer_keywords": ["整形", "修剪", "树形", "枝"],
        "note": "整形修剪",
    },
    {
        "question": "脐橙开花期需要注意什么管理措施？",
        "doc_keywords": [["花期", "管理"], ["开花", "脐橙"], ["保花", "脐橙"]],
        "answer_keywords": ["花期", "保花", "管理", "开花"],
        "note": "花期管理",
    },
    {
        "question": "脐橙如何促进花芽分化？",
        "doc_keywords": [["花芽分化", "脐橙"], ["促花", "脐橙"], ["花芽", "促进"]],
        "answer_keywords": ["花芽分化", "促花", "控梢"],
        "note": "花芽管理",
    },
    {
        "question": "脐橙壮果期如何管理？",
        "doc_keywords": [["壮果", "管理"], ["壮果期", "脐橙"]],
        "answer_keywords": ["壮果", "管理", "施肥", "灌水"],
        "note": "果实膨大期管理",
    },

    # ── 采收与采后处理────────────────────────────────────────
    {
        "question": "脐橙的成熟期和最佳采收时间是什么时候？",
        "doc_keywords": [["采收", "时间"], ["成熟", "脐橙"], ["采摘", "时间"]],
        "answer_keywords": ["采收", "成熟", "月", "采摘"],
        "note": "采收时期",
    },
    {
        "question": "脐橙采收时有哪些注意事项？",
        "doc_keywords": [["采收", "注意"], ["采摘", "方法"], ["采果", "操作"]],
        "answer_keywords": ["采收", "采摘", "注意", "操作"],
        "note": "采收操作",
    },
    {
        "question": "脐橙的商品化处理包括哪些步骤？",
        "doc_keywords": [["商品化", "处理"], ["分级", "包装"], ["清洗", "打蜡"]],
        "answer_keywords": ["清洗", "分级", "包装", "商品化"],
        "note": "商品化处理",
    },

    # ── 品种────────────────────────────────────────────────
    {
        "question": "赣南脐橙有哪些主要品种？",
        "doc_keywords": [["脐橙", "品种"], ["赣南", "品种"], ["纽荷尔", "脐橙"]],
        "answer_keywords": ["品种", "纽荷尔", "赣南"],
        "note": "品种介绍",
    },
    {
        "question": "纽荷尔脐橙的特点是什么？",
        "doc_keywords": [["纽荷尔", "特点"], ["纽荷尔", "性状"]],
        "answer_keywords": ["纽荷尔", "特点", "品质", "产量"],
        "note": "主栽品种",
    },
    {
        "question": "脐橙早熟品种有哪些推荐？",
        "doc_keywords": [["早熟", "脐橙", "品种"], ["早熟", "品种"]],
        "answer_keywords": ["早熟", "品种", "脐橙"],
        "note": "早熟品种",
    },

    # ── 土壤与环境────────────────────────────────────────────
    {
        "question": "脐橙适合什么样的土壤条件？",
        "doc_keywords": [["土壤", "脐橙"], ["土壤", "pH"], ["土壤", "要求"]],
        "answer_keywords": ["土壤", "pH", "有机质", "酸性"],
        "note": "土壤要求",
    },
    {
        "question": "脐橙对温度和气候有什么要求？",
        "doc_keywords": [["温度", "脐橙"], ["气候", "脐橙"], ["积温", "脐橙"]],
        "answer_keywords": ["温度", "气候", "积温", "℃"],
        "note": "气候适应性",
    },
    {
        "question": "脐橙园如何改良土壤，提高土壤有机质？",
        "doc_keywords": [["土壤改良", "脐橙"], ["有机质", "提高"], ["绿肥", "脐橙"]],
        "answer_keywords": ["土壤改良", "有机质", "绿肥", "堆肥"],
        "note": "土壤改良",
    },
    {
        "question": "脐橙园的除草方法有哪些？",
        "doc_keywords": [["除草", "脐橙"], ["杂草", "管理"], ["覆盖", "除草"]],
        "answer_keywords": ["除草", "杂草", "覆盖", "管理"],
        "note": "园地管理",
    },
]


# =====================================================================
# 评估工具函数
# =====================================================================

def docs_contain_keywords(docs: list, keyword_groups: List[List[str]]) -> Tuple[bool, int]:
    """
    判断文档列表中是否有文档命中了关键词组（任一组中的所有关键词都出现在同一文档中）。
    返回：(是否命中, 首个命中文档的索引，未命中时为 -1)
    """
    for idx, doc in enumerate(docs):
        content = doc.page_content
        for group in keyword_groups:
            if all(kw in content for kw in group):
                return True, idx
    return False, -1


def docs_count_relevant(docs: list, keyword_groups: List[List[str]]) -> int:
    """
    统计文档列表中命中关键词组的文档数（用于计算 Precision@k）。
    每篇文档只要命中任意一个关键词组即算相关。
    """
    count = 0
    for doc in docs:
        content = doc.page_content
        for group in keyword_groups:
            if all(kw in content for kw in group):
                count += 1
                break  # 同一篇文档不重复计数
    return count


def answer_contains_keywords(answer: str, keywords: List[str]) -> bool:
    """答案中是否包含至少一个期望关键词"""
    return any(kw in answer for kw in keywords)


def compute_mrr(hit_index: int) -> float:
    """根据首个命中文档位置计算 MRR（未命中时为 0）"""
    if hit_index < 0:
        return 0.0
    return 1.0 / (hit_index + 1)


def compute_f1(precision: float, recall: float) -> float:
    """F1 调和均值"""
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


# =====================================================================
# 各检索配置的包装函数
# =====================================================================

def retrieve_baseline(kb: VectorKnowledgeBase, question: str, k: int) -> list:
    """C0: 直接向量检索"""
    return kb.search_docs(question, k=k)


def retrieve_enhanced(kb: VectorKnowledgeBase, question: str, k: int) -> list:
    """C3: HyDE + MQE 融合"""
    return kb.search_enhanced(question, k=k)


def retrieve_crag(kb: VectorKnowledgeBase, question: str, k: int) -> list:
    """
    C6: 增强检索 + Rerank + CRAG 质量分级过滤。
    - GOOD：直接返回 rerank 后的文档
    - MEDIUM：返回 rerank 后的文档（联网补充留给生成阶段）
    - LOW：rerank 分数不达标，但仍返回本地文档用于召回评估
      （评估的是本地检索能力，不能因为 CRAG 要走联网就把文档置空）
    始终以本地 rerank 文档作为召回评估依据，与 full 模式保持一致。
    """
    try:
        candidates = kb.search_enhanced(question, k=k)
    except Exception:
        candidates = kb.search_docs(question, k=k)

    ranked_items = kb.rerank_docs_with_scores(question, candidates, top_k=k)
    # 不论 CRAG 等级如何，召回评估始终使用本地 rerank 文档
    return [item["doc"] for item in ranked_items]


# =====================================================================
# 单条样本评估
# =====================================================================

# =====================================================================
# LLM-as-Judge：让 LLM 从三个维度对答案打分
# =====================================================================

@dataclass
class JudgeScores:
    faithfulness: float   # 忠实度：答案是否忠于检索文档，无编造（1-5）
    relevance: float      # 相关性：答案是否切题（1-5）
    completeness: float   # 完整性：答案是否覆盖问题要点（1-5）
    avg: float            # 三项均值


def llm_judge(kb: VectorKnowledgeBase, question: str, context: str, answer: str) -> JudgeScores:
    """
    调用 LLM 对一条 RAG 答案从三个维度打分，每项 1-5 分。
    输出严格 JSON，解析失败时三项均为 0。
    """
    prompt = f"""你是一个专业的 RAG 系统答案质量评估员。
请根据【问题】【参考文档】【生成答案】，对答案从以下三个维度打分，每项 1-5 分整数：

1. 忠实度（faithfulness）：答案内容是否完全来自参考文档，有无凭空编造。
   5=完全忠实无编造，4=基本忠实有极少推断，3=部分内容超出文档，2=明显编造，1=严重编造

2. 相关性（relevance）：答案是否正面回答了问题，有无跑题。
   5=完全切题，4=基本切题，3=部分切题，2=大部分跑题，1=完全跑题

3. 完整性（completeness）：答案是否覆盖了问题的所有核心要点。
   5=要点完整，4=覆盖主要要点，3=覆盖一半要点，2=只覆盖少量要点，1=几乎未覆盖

只输出如下 JSON，不要输出任何解释和 markdown：
{{"faithfulness": 分数, "relevance": 分数, "completeness": 分数}}

【问题】
{question}

【参考文档】
{context[:1000]}

【生成答案】
{answer[:500]}

JSON："""

    try:
        raw = kb.call_llm(prompt, temperature=0.0).strip()
        # 去掉可能的 ```json ... ``` 包裹
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        data = json.loads(raw)
        f = float(data.get("faithfulness", 0))
        r = float(data.get("relevance", 0))
        c = float(data.get("completeness", 0))
        # 限制在 1-5 范围内
        f, r, c = [max(1.0, min(5.0, x)) for x in (f, r, c)]
        return JudgeScores(faithfulness=f, relevance=r, completeness=c, avg=round((f + r + c) / 3, 3))
    except Exception as e:
        print(f"         [Judge] 解析失败: {e}  原始输出: {raw[:80] if 'raw' in dir() else '无'}")
        return JudgeScores(faithfulness=0.0, relevance=0.0, completeness=0.0, avg=0.0)


@dataclass
class SampleResult:
    question: str
    config_name: str
    recall_hit: bool        # 是否命中相关文档（Recall@k 分子）
    hit_index: int          # 首个命中文档位置（-1=未命中）
    mrr: float              # MRR
    precision: float        # Precision@k：相关文档数 / k
    answer_hit: bool        # 答案关键词命中（旧指标，保留兼容）
    answer: str             # LLM 生成的答案（截取前 200 字）
    latency_s: float        # 检索+生成总耗时（秒）
    judge: Optional[JudgeScores] = None   # LLM-as-Judge 评分（仅 full 模式）
    error: Optional[str] = None


def evaluate_sample_retrieval_only(
    kb: VectorKnowledgeBase,
    sample: dict,
    retrieve_fn,
    config_name: str,
    k: int,
) -> SampleResult:
    """
    仅评估检索质量（不调用 LLM 生成答案），速度更快，适合快速对比。
    answer_hit 固定为 False，answer 为空。
    """
    question = sample["question"]
    doc_keywords = sample["doc_keywords"]

    t0 = time.time()
    try:
        docs = retrieve_fn(kb, question, k)
        latency = time.time() - t0

        recall_hit, hit_idx = docs_contain_keywords(docs, doc_keywords)
        mrr = compute_mrr(hit_idx)
        relevant_count = docs_count_relevant(docs, doc_keywords)
        precision = round(relevant_count / max(len(docs), 1), 4)

        return SampleResult(
            question=question,
            config_name=config_name,
            recall_hit=recall_hit,
            hit_index=hit_idx,
            mrr=mrr,
            precision=precision,
            answer_hit=False,
            answer="",
            latency_s=round(latency, 3),
        )
    except Exception as e:
        return SampleResult(
            question=question,
            config_name=config_name,
            recall_hit=False,
            hit_index=-1,
            mrr=0.0,
            precision=0.0,
            answer_hit=False,
            answer="",
            latency_s=round(time.time() - t0, 3),
            error=str(e),
        )


def evaluate_sample_full(
    kb: VectorKnowledgeBase,
    memory_store: MemoryStore,
    sample: dict,
    retrieve_fn,
    config_name: str,
    k: int,
) -> SampleResult:
    """
    完整评估：检索 + LLM 生成答案 + LLM-as-Judge 三维打分。
    """
    question = sample["question"]
    doc_keywords = sample["doc_keywords"]
    answer_keywords = sample["answer_keywords"]

    t0 = time.time()
    try:
        docs = retrieve_fn(kb, question, k)
        recall_hit, hit_idx = docs_contain_keywords(docs, doc_keywords)
        mrr = compute_mrr(hit_idx)
        relevant_count = docs_count_relevant(docs, doc_keywords)
        precision = round(relevant_count / max(len(docs), 1), 4)

        # 拼 Prompt，调 LLM 生成答案
        qa_tool = KnowledgeQATool(kb, memory_store)
        context = qa_tool._format_context(docs)
        prompt = qa_tool._build_qa_prompt(question, f"【本地知识库资料】\n{context}")
        answer_text = kb.call_llm(prompt, temperature=0.1)

        ans_hit = answer_contains_keywords(answer_text, answer_keywords)

        # LLM-as-Judge 打分
        judge = llm_judge(kb, question, context, answer_text)

        latency = time.time() - t0

        return SampleResult(
            question=question,
            config_name=config_name,
            recall_hit=recall_hit,
            hit_index=hit_idx,
            mrr=mrr,
            precision=precision,
            answer_hit=ans_hit,
            answer=answer_text[:200],
            latency_s=round(latency, 3),
            judge=judge,
        )
    except Exception as e:
        return SampleResult(
            question=question,
            config_name=config_name,
            recall_hit=False,
            hit_index=-1,
            mrr=0.0,
            precision=0.0,
            answer_hit=False,
            answer="",
            latency_s=round(time.time() - t0, 3),
            error=str(e),
        )


# =====================================================================
# CRAG 完整流程评估（独立，因为 CRAG 内部已含检索+生成）
# =====================================================================

def evaluate_sample_crag(
    kb: VectorKnowledgeBase,
    memory_store: MemoryStore,
    sample: dict,
    k: int,
    allow_web: bool = False,
) -> SampleResult:
    """
    C6_CRAG_Full: 使用 KnowledgeQATool.run() 完整 CRAG 流程。

    关键修正：
    - CRAG 在 LOW 分级时会丢弃本地文档改走联网，导致 source_documents=[]，
      召回评估全部为 0，无法反映本地检索能力。
    - 因此召回评估固定使用 retrieve_crag() 拿到的 rerank 本地文档，
      不受 CRAG grade 分支影响；生成质量评估仍使用完整 CRAG 流程的答案。
    """
    question = sample["question"]
    doc_keywords = sample["doc_keywords"]
    answer_keywords = sample["answer_keywords"]

    t0 = time.time()
    try:
        # ── 召回评估：始终用本地 rerank 文档 ─────────────────────
        eval_docs = retrieve_crag(kb, question, k)
        recall_hit, hit_idx = docs_contain_keywords(eval_docs, doc_keywords)
        mrr = compute_mrr(hit_idx)
        relevant_count = docs_count_relevant(eval_docs, doc_keywords)
        precision = round(relevant_count / max(len(eval_docs), 1), 4)

        # ── 生成评估：走完整 CRAG 流程（含问题改写 + 质量分级 + 按需联网）
        qa_tool = KnowledgeQATool(kb, memory_store)
        result = qa_tool.run(query=question, allow_web=allow_web)

        answer_text = result.get("answer", "")
        ans_hit = answer_contains_keywords(answer_text, answer_keywords)
        crag_grade = result.get("crag_grade", "UNKNOWN")
        best_score = result.get("retrieval_best_score", 0.0)

        # ── Judge 打分：用本地 rerank 文档作为参考上下文 ────────────
        tmp_tool = KnowledgeQATool(kb, memory_store)
        context = tmp_tool._format_context(eval_docs)
        judge = llm_judge(kb, question, context, answer_text)

        latency = time.time() - t0

        print(
            f"         [CRAG] grade={crag_grade}  score={best_score:.3f}  "
            f"召回={'✓' if recall_hit else '✗'}  答案={'✓' if ans_hit else '✗'}"
        )

        return SampleResult(
            question=question,
            config_name="C6_CRAG_Full",
            recall_hit=recall_hit,
            hit_index=hit_idx,
            mrr=mrr,
            precision=precision,
            answer_hit=ans_hit,
            answer=answer_text[:200],
            latency_s=round(latency, 3),
            judge=judge,
        )
    except Exception as e:
        return SampleResult(
            question=question,
            config_name="C6_CRAG_Full",
            recall_hit=False,
            hit_index=-1,
            mrr=0.0,
            precision=0.0,
            answer_hit=False,
            answer="",
            latency_s=round(time.time() - t0, 3),
            error=str(e),
        )


# =====================================================================
# 汇总统计
# =====================================================================

@dataclass
class ConfigSummary:
    config_name: str
    total: int
    recall_hits: int
    recall_rate: float      # Recall@k
    avg_precision: float    # Precision@k 均值
    f1: float               # F1（Precision 与 Recall 调和均值）
    avg_mrr: float          # Mean Reciprocal Rank
    answer_hits: int
    answer_accuracy: float  # Answer Hit Rate（关键词）
    avg_faithfulness: float # Judge：忠实度均值
    avg_relevance: float    # Judge：相关性均值
    avg_completeness: float # Judge：完整性均值
    avg_judge: float        # Judge：三项总均值（平均质量得分）
    avg_latency_s: float    # 平均响应时间（秒）
    errors: int


def summarize(results: List[SampleResult]) -> ConfigSummary:
    total = len(results)
    if total == 0:
        raise ValueError("结果列表为空")

    recall_hits = sum(1 for r in results if r.recall_hit)
    answer_hits = sum(1 for r in results if r.answer_hit)
    errors = sum(1 for r in results if r.error)
    avg_mrr = sum(r.mrr for r in results) / total
    avg_lat = sum(r.latency_s for r in results) / total
    avg_prec = round(sum(r.precision for r in results) / total, 4)
    recall = round(recall_hits / total, 4)
    f1 = compute_f1(avg_prec, recall)

    # Judge 分数：只统计有 judge 结果的样本
    judged = [r for r in results if r.judge is not None]
    if judged:
        avg_faith = sum(r.judge.faithfulness for r in judged) / len(judged)
        avg_rel   = sum(r.judge.relevance    for r in judged) / len(judged)
        avg_comp  = sum(r.judge.completeness for r in judged) / len(judged)
        avg_jdg   = sum(r.judge.avg          for r in judged) / len(judged)
    else:
        avg_faith = avg_rel = avg_comp = avg_jdg = 0.0

    return ConfigSummary(
        config_name=results[0].config_name,
        total=total,
        recall_hits=recall_hits,
        recall_rate=recall,
        avg_precision=avg_prec,
        f1=f1,
        avg_mrr=round(avg_mrr, 4),
        answer_hits=answer_hits,
        answer_accuracy=round(answer_hits / total, 4),
        avg_faithfulness=round(avg_faith, 3),
        avg_relevance=round(avg_rel, 3),
        avg_completeness=round(avg_comp, 3),
        avg_judge=round(avg_jdg, 3),
        avg_latency_s=round(avg_lat, 3),
        errors=errors,
    )


# =====================================================================
# 打印结果表格
# =====================================================================

def print_results_table(summaries: List[ConfigSummary], mode: str):
    has_judge = mode == "full"

    print("\n" + "=" * 120)
    print(f"  RAG 消融测试结果  |  评估模式: {'仅检索' if mode == 'retrieval' else '检索+生成+Judge'}")
    print("=" * 120)

    # 表头
    if has_judge:
        header = (
            f"{'配置':<26} {'Precision':>10} {'Recall@k':>10} {'F1':>7} {'MRR':>7} "
            f"{'Faith':>6} {'Relev':>6} {'Compl':>6} {'Judge↑':>7} "
            f"{'Lat(s)':>7} {'Err':>4}"
        )
    else:
        header = (
            f"{'配置':<26} {'Precision':>10} {'Recall@k':>10} {'F1':>7} {'MRR':>7} "
            f"{'Lat(s)':>7} {'Err':>4}"
        )
    print(header)
    print("-" * 120)

    for s in summaries:
        if has_judge:
            print(
                f"{s.config_name:<26} "
                f"{s.avg_precision:>10.1%}  "
                f"{s.recall_rate:>8.1%}  "
                f"{s.f1:>6.4f}  "
                f"{s.avg_mrr:>6.4f}  "
                f"{s.avg_faithfulness:>5.2f}  "
                f"{s.avg_relevance:>5.2f}  "
                f"{s.avg_completeness:>5.2f}  "
                f"{s.avg_judge:>6.2f}  "
                f"{s.avg_latency_s:>6.1f}  "
                f"{s.errors:>3}"
            )
        else:
            print(
                f"{s.config_name:<26} "
                f"{s.avg_precision:>10.1%}  "
                f"{s.recall_rate:>8.1%}  "
                f"{s.f1:>6.4f}  "
                f"{s.avg_mrr:>6.4f}  "
                f"{s.avg_latency_s:>6.1f}  "
                f"{s.errors:>3}"
            )

    print("=" * 120)

    # 各指标最优配置
    best_recall = max(summaries, key=lambda s: s.recall_rate)
    best_prec   = max(summaries, key=lambda s: s.avg_precision)
    best_f1     = max(summaries, key=lambda s: s.f1)
    best_mrr    = max(summaries, key=lambda s: s.avg_mrr)
    best_lat    = min(summaries, key=lambda s: s.avg_latency_s)
    print(f"\n✅  Precision@k 最高: {best_prec.config_name}  ({best_prec.avg_precision:.1%})")
    print(f"✅  Recall@k   最高: {best_recall.config_name}  ({best_recall.recall_rate:.1%})")
    print(f"✅  F1         最高: {best_f1.config_name}  ({best_f1.f1:.4f})")
    print(f"✅  MRR        最高: {best_mrr.config_name}  ({best_mrr.avg_mrr:.4f})")
    print(f"✅  响应最快        : {best_lat.config_name}  ({best_lat.avg_latency_s:.1f}s)")
    if has_judge:
        best_judge = max(summaries, key=lambda s: s.avg_judge)
        best_faith = max(summaries, key=lambda s: s.avg_faithfulness)
        best_rel   = max(summaries, key=lambda s: s.avg_relevance)
        best_comp  = max(summaries, key=lambda s: s.avg_completeness)
        print(f"✅  Judge综合最高: {best_judge.config_name}  ({best_judge.avg_judge:.2f}/5)")
        print(f"   ├ 忠实度最高: {best_faith.config_name}  ({best_faith.avg_faithfulness:.2f}/5)")
        print(f"   ├ 相关性最高: {best_rel.config_name}  ({best_rel.avg_relevance:.2f}/5)")
        print(f"   └ 完整性最高: {best_comp.config_name}  ({best_comp.avg_completeness:.2f}/5)")


def print_detail_table(all_results: Dict[str, List[SampleResult]]):
    """打印每条问题的逐 config 命中 + Judge 均分情况"""
    configs = list(all_results.keys())
    first_results = next(iter(all_results.values()))
    questions = [r.question[:22] for r in first_results]
    has_judge = any(r.judge is not None for r in first_results)

    print("\n" + "=" * 105)
    if has_judge:
        print("  逐问题命中详情（✓/✗=召回, 括号内=Judge均分）")
    else:
        print("  逐问题命中详情（✓=召回命中, ✗=未命中）")
    print("=" * 105)

    col_w = 16 if has_judge else 13
    header = f"{'问题':<24} " + "  ".join(f"{c[:col_w]:<{col_w}}" for c in configs)
    print(header)
    print("-" * 105)

    for i, q in enumerate(questions):
        row = f"{q:<24} "
        for cfg in configs:
            r = all_results[cfg][i]
            mark = "✓" if r.recall_hit else "✗"
            if has_judge and r.judge is not None:
                cell = f"{mark}({r.judge.avg:.1f})"
            else:
                cell = mark
            row += f"  {cell:<{col_w}}"
        print(row)
    print("=" * 105)


# =====================================================================
# 主流程
# =====================================================================

CONFIGS_RETRIEVAL = [
    ("C0_Baseline",  retrieve_baseline),
    ("C3_HyDE+MQE",  retrieve_enhanced),
    ("C6_CRAG",      retrieve_crag),       # 增强检索 + Rerank + CRAG 质量分级
]

CONFIGS_FULL = [
    ("C0_Baseline",  retrieve_baseline),
    ("C3_HyDE+MQE",  retrieve_enhanced),
    ("C6_CRAG",      retrieve_crag),       # 用于 full 模式 judge 对比（与 CRAG 完整流程区分）
    # C6_CRAG_Full 单独处理（含完整 CRAG 流程：rewrite + web fallback）
]


def run_tests(args):
    print("=" * 60)
    print("  脐橙 RAG 消融测试")
    print(f"  模式: {args.mode}  |  k={args.k}  |  样本数={len(TEST_DATASET)}")
    print("=" * 60)

    print("\n[初始化] 加载向量知识库...")
    kb = VectorKnowledgeBase()
    memory_store = MemoryStore()
    print("[初始化] 完成\n")

    all_results: Dict[str, List[SampleResult]] = {}
    summaries: List[ConfigSummary] = []

    # ── 仅检索模式：测 4 个配置 ──────────────────────────────
    if args.mode == "retrieval":
        for cfg_name, retrieve_fn in CONFIGS_RETRIEVAL:
            print(f"▶  测试配置: {cfg_name}")
            results = []
            for i, sample in enumerate(TEST_DATASET):
                print(f"   [{i+1}/{len(TEST_DATASET)}] {sample['question'][:30]}...")
                r = evaluate_sample_retrieval_only(kb, sample, retrieve_fn, cfg_name, args.k)
                results.append(r)
                status = "✓" if r.recall_hit else "✗"
                err_info = f"  ERR: {r.error}" if r.error else ""
                print(f"         召回: {status}  MRR: {r.mrr:.2f}  耗时: {r.latency_s}s{err_info}")
            all_results[cfg_name] = results
            summaries.append(summarize(results))
            print()

    # ── 完整模式：测部分配置（+C6 CRAG）────────────────────────
    elif args.mode == "full":
        for cfg_name, retrieve_fn in CONFIGS_FULL:
            print(f"▶  测试配置: {cfg_name}")
            results = []
            for i, sample in enumerate(TEST_DATASET):
                print(f"   [{i+1}/{len(TEST_DATASET)}] {sample['question'][:30]}...")
                r = evaluate_sample_full(kb, memory_store, sample, retrieve_fn, cfg_name, args.k)
                results.append(r)
                status = "✓" if r.recall_hit else "✗"
                ans_status = "✓" if r.answer_hit else "✗"
                judge_info = f"  Judge={r.judge.avg:.1f}" if r.judge else ""
                err_info = f"  ERR: {r.error}" if r.error else ""
                print(f"         召回: {status}  答案: {ans_status}{judge_info}  耗时: {r.latency_s}s{err_info}")
            all_results[cfg_name] = results
            summaries.append(summarize(results))
            print()

        # C6_CRAG_Full: 完整 CRAG 流程（含 rewrite + 质量分级 + 按需联网）
        print("▶  测试配置: C6_CRAG_Full（完整 CRAG 流程）")
        crag_results = []
        for i, sample in enumerate(TEST_DATASET):
            print(f"   [{i+1}/{len(TEST_DATASET)}] {sample['question'][:30]}...")
            r = evaluate_sample_crag(kb, memory_store, sample, args.k, allow_web=False)
            crag_results.append(r)
            status = "✓" if r.recall_hit else "✗"
            ans_status = "✓" if r.answer_hit else "✗"
            judge_info = f"  Judge={r.judge.avg:.1f}" if r.judge else ""
            err_info = f"  ERR: {r.error}" if r.error else ""
            print(f"         召回: {status}  答案: {ans_status}{judge_info}  耗时: {r.latency_s}s{err_info}")
        all_results["C6_CRAG_Full"] = crag_results
        summaries.append(summarize(crag_results))
        print()

    # ── 打印结果 ────────────────────────────────────────────────
    print_results_table(summaries, args.mode)
    print_detail_table(all_results)

    # ── 自动保存结果 ─────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    auto_path = f"rag_result_{args.mode}_{timestamp}.json"
    save_path = args.output if args.output else auto_path

    output_data = {
        "meta": {
            "mode": args.mode,
            "k": args.k,
            "total_samples": len(TEST_DATASET),
            "timestamp": timestamp,
        },
        "summaries": [asdict(s) for s in summaries],
        "details": {
            cfg: [asdict(r) for r in rs]
            for cfg, rs in all_results.items()
        },
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n📄 JSON 结果已保存到: {save_path}")

    # ── Excel 导出 ────────────────────────────────────────────────
    excel_path = save_path.replace(".json", ".xlsx")
    _save_excel(summaries, all_results, args.mode, args.k, timestamp, excel_path)
    print(f"📊 Excel 结果已保存到: {excel_path}")

    return summaries, all_results


# =====================================================================
# Excel 导出
# =====================================================================

def _save_excel(
    summaries: List[ConfigSummary],
    all_results: Dict[str, List[SampleResult]],
    mode: str,
    k: int,
    timestamp: str,
    path: str,
):
    """将汇总指标和逐条结果分别写入两个 Sheet，格式美观。"""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()

    # ── 颜色定义 ──────────────────────────────────────────────────
    HEADER_FILL = PatternFill("solid", start_color="1F4E79")  # 深蓝
    SUBHDR_FILL = PatternFill("solid", start_color="2E75B6")  # 中蓝
    ALT_FILL    = PatternFill("solid", start_color="D6E4F0")  # 浅蓝交替行
    BEST_FILL   = PatternFill("solid", start_color="E2EFDA")  # 浅绿（最优值）
    WHITE_FILL  = PatternFill("solid", start_color="FFFFFF")
    HDR_FONT    = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    BODY_FONT   = Font(name="Arial", size=10)
    BOLD_FONT   = Font(name="Arial", bold=True, size=10)
    CENTER      = Alignment(horizontal="center", vertical="center", wrap_text=True)
    LEFT        = Alignment(horizontal="left",   vertical="center", wrap_text=True)
    thin        = Side(style="thin", color="AAAAAA")
    BORDER      = Border(left=thin, right=thin, top=thin, bottom=thin)

    def _apply_header(cell, text):
        cell.value = text
        cell.font = HDR_FONT
        cell.fill = HEADER_FILL
        cell.alignment = CENTER
        cell.border = BORDER

    def _apply_subheader(cell, text):
        cell.value = text
        cell.font = HDR_FONT
        cell.fill = SUBHDR_FILL
        cell.alignment = CENTER
        cell.border = BORDER

    def _apply_cell(cell, value, bold=False, align=CENTER, fill=WHITE_FILL):
        cell.value = value
        cell.font = BOLD_FONT if bold else BODY_FONT
        cell.alignment = align
        cell.fill = fill
        cell.border = BORDER

    # ======================================================
    # Sheet 1：指标汇总
    # ======================================================
    ws1 = wb.active
    ws1.title = "指标汇总"
    ws1.sheet_view.showGridLines = False
    ws1.row_dimensions[1].height = 20
    ws1.row_dimensions[2].height = 30

    # 标题行
    ws1.merge_cells("A1:O1")
    title_cell = ws1["A1"]
    title_cell.value = f"脐橙 RAG 消融测试  |  模式: {mode}  |  Top-k={k}  |  {timestamp}"
    title_cell.font = Font(name="Arial", bold=True, size=13, color="1F4E79")
    title_cell.alignment = CENTER

    has_judge = mode == "full"

    # 表头
    headers = [
        "配置名称", "样本总数",
        "Precision@k", "Recall@k", "F1",
        "MRR", "答案命中率",
        "忠实度(1-5)", "相关性(1-5)", "完整性(1-5)", "质量得分(1-5)",
        "平均响应时间(s)", "错误数",
    ]
    for col, h in enumerate(headers, start=1):
        _apply_header(ws1.cell(row=2, column=col), h)

    # 数据行
    for row_idx, s in enumerate(summaries, start=3):
        fill = ALT_FILL if row_idx % 2 == 1 else WHITE_FILL
        data = [
            s.config_name,
            s.total,
            f"{s.avg_precision:.1%}",
            f"{s.recall_rate:.1%}",
            f"{s.f1:.4f}",
            f"{s.avg_mrr:.4f}",
            f"{s.answer_accuracy:.1%}" if has_judge else "N/A",
            f"{s.avg_faithfulness:.2f}" if has_judge else "N/A",
            f"{s.avg_relevance:.2f}"    if has_judge else "N/A",
            f"{s.avg_completeness:.2f}" if has_judge else "N/A",
            f"{s.avg_judge:.2f}"        if has_judge else "N/A",
            f"{s.avg_latency_s:.2f}",
            s.errors,
        ]
        for col, val in enumerate(data, start=1):
            _apply_cell(ws1.cell(row=row_idx, column=col), val, fill=fill)

    # 最优值标绿
    metric_col_key = {
        3: "avg_precision",   # Precision
        4: "recall_rate",     # Recall
        5: "f1",              # F1
        6: "avg_mrr",         # MRR
        11: "avg_judge",      # Judge
    }
    last_data_row = 2 + len(summaries)
    for col, attr in metric_col_key.items():
        vals = [getattr(s, attr) for s in summaries]
        if not vals:
            continue
        best_val = max(vals)
        for row_idx, s in enumerate(summaries, start=3):
            if getattr(s, attr) == best_val:
                ws1.cell(row=row_idx, column=col).fill = BEST_FILL
    # 响应时间最小标绿（col 12）
    lat_vals = [s.avg_latency_s for s in summaries]
    best_lat = min(lat_vals)
    for row_idx, s in enumerate(summaries, start=3):
        if s.avg_latency_s == best_lat:
            ws1.cell(row=row_idx, column=12).fill = BEST_FILL

    # 列宽
    col_widths = [22, 8, 12, 12, 10, 10, 12, 13, 13, 13, 14, 16, 8]
    for i, w in enumerate(col_widths, start=1):
        ws1.column_dimensions[get_column_letter(i)].width = w

    # 图例说明
    note_row = last_data_row + 2
    ws1.merge_cells(f"A{note_row}:F{note_row}")
    note_cell = ws1.cell(row=note_row, column=1)
    note_cell.value = "🟢 绿色底色 = 该指标最优值"
    note_cell.font = Font(name="Arial", italic=True, size=9, color="375623")
    note_cell.alignment = LEFT

    # ======================================================
    # Sheet 2：逐条结果明细
    # ======================================================
    ws2 = wb.create_sheet("逐条明细")
    ws2.sheet_view.showGridLines = False

    detail_headers = [
        "配置", "问题", "Precision@k", "Recall命中", "首命中位置", "MRR",
        "答案命中", "忠实度", "相关性", "完整性", "质量均分", "响应时间(s)", "错误信息"
    ]
    for col, h in enumerate(detail_headers, start=1):
        _apply_header(ws2.cell(row=1, column=col), h)

    dr = 2
    for cfg_name, results in all_results.items():
        _apply_subheader(ws2.cell(row=dr, column=1), cfg_name)
        ws2.merge_cells(f"A{dr}:{get_column_letter(len(detail_headers))}{dr}")
        ws2.row_dimensions[dr].height = 18
        dr += 1
        for r in results:
            fill = ALT_FILL if dr % 2 == 0 else WHITE_FILL
            judge_vals = (
                [r.judge.faithfulness, r.judge.relevance, r.judge.completeness, r.judge.avg]
                if r.judge else ["—", "—", "—", "—"]
            )
            row_data = [
                r.config_name,
                r.question,
                f"{r.precision:.1%}",
                "✓" if r.recall_hit else "✗",
                r.hit_index if r.hit_index >= 0 else "未命中",
                f"{r.mrr:.4f}",
                "✓" if r.answer_hit else "✗",
                *[f"{v:.2f}" if isinstance(v, float) else v for v in judge_vals],
                r.latency_s,
                r.error or "",
            ]
            for col, val in enumerate(row_data, start=1):
                _apply_cell(ws2.cell(row=dr, column=col), val, fill=fill,
                            align=LEFT if col == 2 else CENTER)
            dr += 1

    detail_col_widths = [18, 32, 12, 10, 12, 10, 10, 10, 10, 10, 10, 14, 20]
    for i, w in enumerate(detail_col_widths, start=1):
        ws2.column_dimensions[get_column_letter(i)].width = w

    # ======================================================
    # Sheet 3：指标说明
    # ======================================================
    ws3 = wb.create_sheet("指标说明")
    ws3.sheet_view.showGridLines = False
    explanations = [
        ("指标", "说明"),
        ("Precision@k", "返回的 Top-k 文档中相关文档的比例，衡量检索结果的精准程度"),
        ("Recall@k",    "至少有1篇相关文档出现在 Top-k 结果中的样本占比，衡量覆盖能力"),
        ("F1",          "Precision 与 Recall 的调和均值，综合衡量检索质量"),
        ("MRR",         "首个相关文档排名倒数的均值；排名越靠前，MRR 越高"),
        ("答案命中率",  "LLM 生成答案中包含期望关键词的样本比例（full 模式）"),
        ("忠实度",      "LLM-as-Judge：答案是否忠于检索文档，无幻觉（1-5分）"),
        ("相关性",      "LLM-as-Judge：答案是否正面回答问题，不跑题（1-5分）"),
        ("完整性",      "LLM-as-Judge：答案是否覆盖问题所有核心要点（1-5分）"),
        ("质量得分",    "忠实度、相关性、完整性三项均值（1-5分），平均质量得分"),
        ("平均响应时间","检索（+生成）的平均耗时，单位秒"),
    ]
    ws3.column_dimensions["A"].width = 16
    ws3.column_dimensions["B"].width = 60
    for row_idx, (name, desc) in enumerate(explanations, start=1):
        if row_idx == 1:
            _apply_header(ws3.cell(row=1, column=1), name)
            _apply_header(ws3.cell(row=1, column=2), desc)
        else:
            fill = ALT_FILL if row_idx % 2 == 0 else WHITE_FILL
            _apply_cell(ws3.cell(row=row_idx, column=1), name, bold=True, fill=fill)
            _apply_cell(ws3.cell(row=row_idx, column=2), desc, fill=fill, align=LEFT)
        ws3.row_dimensions[row_idx].height = 22

    wb.save(path)


# =====================================================================
# CLI 入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG 增强方法消融测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 快速模式（仅检索，不调用 LLM 生成答案，速度快）
  python test_rag_ablation.py --mode retrieval

  # 完整模式（检索+LLM生成，同时评估答案准确率，较慢）
  python test_rag_ablation.py --mode full

  # 指定 top-k 和输出文件
  python test_rag_ablation.py --mode full --k 5 --output results.json
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "full"],
        default="retrieval",
        help="评估模式：retrieval=仅检索(快)，full=检索+生成(慢)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="每次检索返回的文档数 (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="将结果保存为 JSON 文件的路径（可选）",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
        help="只测前 N 条样本（0 = 全部），用于快速调试",
    )

    args = parser.parse_args()

    if args.subset > 0:
        TEST_DATASET[:] = TEST_DATASET[:args.subset]
        print(f"[调试模式] 只测前 {args.subset} 条样本")

    run_tests(args)