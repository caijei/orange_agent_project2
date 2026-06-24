# rag_service.py
# 脐橙多模态智能体 — 知识库 & 工具服务层
#
# 架构概览：
#   MemoryStore          → 运行时用户记忆（姓名/地点/偏好）
#   VectorKnowledgeBase  → 向量检索 + rerank + 联网搜索 + LLM 调用
#   BaseTool             → 工具基类
#     ├─ MemorySetTool   → 记忆写入
#     ├─ MemoryQueryTool → 记忆读取
#     ├─ KnowledgeQATool → 知识库问答（含改写 + 增强检索 + rerank）
#     ├─ WebSearchTool   → Tavily 联网搜索
#     ├─ ChatTool        → 简单闲聊
#     └─ CommandTool     → 退出/清空命令
#   Router               → 规则优先，规则不命中时走 LLM 路由
#   OrangeAgentService   → 组装所有工具 + Router，对外提供 dispatch()
#   RAGService           → 兼容旧接口的薄封装
#
# 新手阅读路线：
#   1. 先看最下面的 RAGService.ask_question()，这是外部调用入口。
#   2. 再看 OrangeAgentService.dispatch()，它负责“判断该用哪个工具”。
#   3. 然后看 Router，它负责把用户问题分流到不同工具。
#   4. 最后重点看 KnowledgeQATool.run()，这是本地知识库问答的核心流程。
#   5. VectorKnowledgeBase 是底层能力封装，负责真正的向量检索、rerank、LLM、联网搜索。

# json：解析/生成 JSON，主要用于记忆抽取等结构化结果。
import json

# os：读取 .env 中的环境变量，例如模型地址、API Key、向量库路径。
import os

# re：正则表达式，用于判断用户问题、清洗模型输出等。
import re

# TextReRank：DashScope 的重排序模型，用来判断检索结果和问题的相关性。
from dashscope import TextReRank          # Rerank 精排（DashScope gte-rerank）

# TavilyClient：联网搜索客户端，用于天气、价格、政策、新闻等实时信息。
from tavily import TavilyClient
from pathlib import Path
from typing import Dict, List, Optional, Any

# load_dotenv：加载 backend/.env，让 os.getenv(...) 可以读到配置。
from dotenv import load_dotenv

# DashScopeEmbeddings：把文本转成向量，供 Chroma 向量库检索使用。
from langchain_community.embeddings import DashScopeEmbeddings

# Chroma：本地向量数据库，保存知识库切分后的 chunk 向量。
from langchain_community.vectorstores import Chroma

# OpenAI：这里使用 OpenAI 兼容接口，既可以接 DashScope，也可以接 Ollama 兼容服务。
from openai import OpenAI

# 加载同目录下的 .env 文件，读取各项 API Key 和配置
load_dotenv(Path(__file__).with_name(".env"))


# =====================================================
# Store 层：用户记忆仓库
# =====================================================

class MemoryStore:
    """
    运行时内存记忆仓库，存储用户姓名、地点、背景、偏好等信息。
    当前使用 dict 存储，重启后丢失。
    如需持久化，可替换为 json / sqlite / mysql。
    """

    def __init__(self):
        self.data: Dict[str, str] = {}

    def set(self, key: str, value: str):
        """写入一条记忆，value 为空时忽略（不覆盖旧值）"""
        if value:
            self.data[key] = value

    def get(self, key: str, default: str = "") -> str:
        """读取一条记忆，不存在时返回 default"""
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        """判断某条记忆是否存在且非空"""
        return key in self.data and bool(self.data[key])

    def all(self) -> Dict[str, str]:
        """返回全部记忆的副本"""
        return dict(self.data)

    def clear(self):
        """清空所有记忆"""
        self.data.clear()


# =====================================================
# 知识库层：向量检索 + Rerank + LLM + 联网
# =====================================================

class VectorKnowledgeBase:
    """
    封装所有底层能力：
      - Chroma 向量库检索（普通 / 多查询扩展 / HyDE）
      - DashScope gte-rerank 精排
      - OpenAI 兼容接口调用本地 / 云端 LLM
      - Tavily 联网搜索

    你可以把这个类理解成“资料检索工具箱”：
    上层工具只需要问它“帮我找资料/帮我调用模型”，不用关心底层 API 怎么调。
    """

    def __init__(self, collection_name: str = "orange_knowledge"):
        # ── 从环境变量读取配置 ──────────────────────────
        # Chroma 向量库保存位置，通常由 document_processor.py 构建出来。
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

        # DashScope API Key：embedding、rerank、云端模型都可能用到。
        self.api_key           = os.getenv("DASHSCOPE_API_KEY")

        # OpenAI 兼容接口地址。
        # 名字叫 OLLAMA_BASE_URL，但也可以填 DashScope compatible-mode 地址。
        self.ollama_base_url   = os.getenv("OLLAMA_BASE_URL")

        # 实际调用的大语言模型名称，例如 qwen3.6-plus。
        self.ollama_model      = os.getenv("OLLAMA_MODEL")

        # Tavily 联网搜索 API Key，没配置时联网搜索会不可用。
        self.tavily_api_key    = os.getenv("TAVILY_API_KEY")

        # 向量召回数量（建议设大一些，留给 rerank 精排）
        self.search_k      = int(os.getenv("RAG_SEARCH_K", "8"))
        # rerank 后保留送给 LLM 的文档数
        self.rerank_top_k  = int(os.getenv("RAG_RERANK_TOP_K", "4"))
        # rerank 相关性分数阈值，低于此值的文档丢弃
        self.rerank_min_score = float(os.getenv("RERANK_MIN_SCORE", "0.1"))

        # CRAG 检索质量判断阈值
        # best_score >= crag_good_score：GOOD，只用本地知识库
        # best_score <  crag_low_score：LOW，丢弃本地资料，改用联网搜索
        # 其余区间：MEDIUM，本地知识库 + 联网搜索合并
        self.crag_good_score = float(os.getenv("CRAG_GOOD_SCORE", "0.65"))
        self.crag_low_score  = float(os.getenv("CRAG_LOW_SCORE", "0.35"))

        # ── 初始化各客户端 ─────────────────────────────
        # DashScope embedding 模型，用于向量化文本
        self.embeddings = DashScopeEmbeddings(dashscope_api_key=self.api_key)

        # Chroma 向量数据库
        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

        # OpenAI 兼容客户端，指向 DashScope（或 Ollama）
        self.llm_client = OpenAI(
            api_key=self.api_key,
            base_url=self.ollama_base_url,
        )

        # Tavily 联网搜索客户端（未配置 Key 时置 None）
        self.tavily_client = (
            TavilyClient(api_key=self.tavily_api_key)
            if self.tavily_api_key else None
        )

    # ── 基础向量检索 ────────────────────────────────────

    def search_docs(self, query: str, k: Optional[int] = None) -> list:
        """直接用 query 做余弦相似度检索，返回 top-k 文档"""
        top_k = k if k is not None else self.search_k

        # similarity_search 会把 query 转成向量，
        # 再从 Chroma 里找语义最相近的知识片段。
        return self.vectordb.similarity_search(query, k=top_k)

    def _doc_key(self, doc: Any) -> str:
        """优先使用 chunk_id 去重，避免不同检索路径返回同一文本块。"""
        metadata = getattr(doc, "metadata", {}) or {}
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)

        source = metadata.get("source_file", "")
        chunk_index = metadata.get("chunk_index", "")
        if source or chunk_index:
            return f"{source}:{chunk_index}"

        return getattr(doc, "page_content", "")[:120]

    def _weighted_rrf_fuse(self, ranked_doc_lists: List[Dict[str, Any]], limit: int) -> list:
        """
        加权 RRF 融合多个检索结果。
        原问题检索给更高权重，HyDE 和多查询只负责补充召回，减少跑偏文档抢占前排。

        RRF 可以简单理解为“多个排行榜合并”：
        某个文档在多个检索结果里都靠前，它的总分就会更高。
        """
        rank_constant = 60
        scores: Dict[str, float] = {}
        docs_by_key: Dict[str, Any] = {}

        for item in ranked_doc_lists:
            docs = item.get("docs") or []
            weight = float(item.get("weight", 1.0))
            for rank, doc in enumerate(docs, start=1):
                # key 用来给文档去重，避免同一个 chunk 被多种检索方式重复加入。
                key = self._doc_key(doc)
                docs_by_key[key] = doc

                # 排名越靠前，分数越高；weight 用于控制不同检索策略的重要程度。
                scores[key] = scores.get(key, 0.0) + weight / (rank_constant + rank)

        ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
        return [docs_by_key[key] for key in ranked_keys[:limit]]

    # ── 多查询扩展检索 ──────────────────────────────────

    def search_with_multi_query(self, question: str, k: Optional[int] = None) -> list:
        """
        多查询扩展：用 LLM 从 3 个角度改写原问题，分别检索后合并去重。
        命中次数越多的文档排名越靠前（词频投票）。
        目的：提高召回率，避免单一查询漏掉相关文档。
        """
        top_k = k if k is not None else self.search_k

        # 让 LLM 生成 3 个贴近原问题的检索查询，避免泛化到过宽主题。
        prompt = f"""请围绕原问题生成3个用于脐橙知识库检索的改写查询。
要求：
1. 必须保留原问题的核心对象、病害/虫害/管理环节，不要换成更宽泛主题。
2. 可以补充同义词、专业术语或常见表达，但不要添加原问题没有涉及的新问题。
3. 每行只输出一个查询，不要编号，不要解释。

原问题：{question}
改写："""
        try:
            result = self.call_llm(prompt, temperature=0.3)
            expanded = []
            for line in result.strip().split("\n"):
                q = re.sub(r"^\s*[\d一二三四五六七八九十]+[\.、:：\)]\s*", "", line).strip(" -\t")
                if q and q != question and q not in expanded:
                    expanded.append(q)
        except Exception:
            expanded = []

        all_queries = expanded[:3]

        print("\n========== MQE 多查询扩展 ==========")
        print("原问题：", question)
        for i, q in enumerate(all_queries, 1):
            print(f"[扩展查询 {i}] {q}")
        print("===================================\n")

        if not all_queries:
            return []

        ranked_lists = []
        for q in all_queries:
            try:
                ranked_lists.append({
                    "docs": self.vectordb.similarity_search(q, k=top_k),
                    "weight": 1.0,
                })
            except Exception:
                continue

        return self._weighted_rrf_fuse(ranked_lists, limit=top_k * 2)

    # ── HyDE 检索 ───────────────────────────────────────

    def search_with_hyde(self, question: str, k: Optional[int] = None) -> list:
        """
        HyDE（Hypothetical Document Embeddings）：
        先让 LLM 生成一个假设性回答，再用该回答的向量做检索。
        假设答案的语义分布比原问题更贴近知识库文档，检索精度更高。
        """
        top_k = k if k is not None else self.search_k

        prompt = f"""请基于以下问题写一段用于向量检索的脐橙领域假设文档。
要求：
1. 只围绕原问题展开，不要扩展到无关病害、虫害或管理主题。
2. 可以包含症状、原因、发生条件、防治或管理关键词。
3. 不要编造具体农药剂量、年份、地区政策等细节。
4. 输出80到120字左右，只输出正文，不要分析过程。

问题：{question}
假设文档："""
        try:
            hypothetical = self.call_llm(prompt, temperature=0.0)

            print("\n========== HyDE 假设文档检索 ==========")
            print("原问题：", question)
            print("假设文档：")
            print(hypothetical)
            print("======================================\n")

            if not hypothetical:
                return self.search_docs(question, k=top_k)

            return self.vectordb.similarity_search(hypothetical, k=top_k)
        except Exception:
            return self.search_docs(question, k=top_k)

    # ── 增强检索（多查询 + HyDE 融合）─────────────────

    def search_enhanced(self, question: str, k: Optional[int] = None) -> list:
        """
        原问题检索优先，HyDE 和多查询作为补充召回。
        使用加权 RRF 合并候选，避免假设文档或扩展查询跑偏后挤掉原始检索结果。

        这个函数是“召回阶段”的主入口。
        它会同时尝试三种找资料的方法，然后合并结果。
        """
        top_k = k if k is not None else self.search_k

        try:
            # 方法 1：直接用用户问题检索，这是最稳的基础结果。
            original_docs = self.search_docs(question, k=top_k * 2)
        except Exception:
            original_docs = []

        try:
            # 方法 2：HyDE，先生成一段“假设答案”，再用这段假设答案检索。
            hyde_docs = self.search_with_hyde(question, k=top_k)
        except Exception:
            hyde_docs = []

        try:
            # 方法 3：多查询扩展，让模型改写几个相近问法分别检索。
            multi_docs = self.search_with_multi_query(question, k=top_k)
        except Exception:
            multi_docs = []

        merged = self._weighted_rrf_fuse(
            [
                {"docs": original_docs, "weight": 1.0},
                {"docs": multi_docs, "weight": 1.0},
                {"docs": hyde_docs, "weight": 0.8},
            ],
            limit=top_k * 3,
        )

        return merged if merged else self.search_docs(question, k=top_k)

    # ── Rerank 精排 ─────────────────────────────────────

    def rerank_docs_with_scores(self, query: str, docs: list, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        使用 DashScope gte-rerank 对召回文档做精排，并保留 relevance_score。

        CRAG 需要根据 relevance_score 判断本地检索质量，因此这里返回：
        [
            {"doc": Document, "score": 0.82},
            {"doc": Document, "score": 0.61},
        ]

        降级策略：
        - rerank API 调用失败时，保留原始检索顺序，并把 score 置为 None。
        - 后续 CRAG 会把无分数但有文档的情况保守判断为 MEDIUM。
        """
        if not docs:
            return []

        keep = top_k if top_k is not None else self.rerank_top_k

        try:
            # rerank API 只需要文本内容，所以从 Document 中取 page_content。
            passages = [doc.page_content for doc in docs]

            resp = TextReRank.call(
                model="gte-rerank",
                query=query,
                documents=passages,
                top_n=keep,
                return_documents=False,
                api_key=self.api_key,
            )

            if resp.status_code != 200:
                print(f"[Rerank] API 错误: {resp.status_code} {resp.message}")
                return [{"doc": doc, "score": None} for doc in docs[:keep]]

            ranked = resp.output.results

            ranked_items: List[Dict[str, Any]] = []
            for item in ranked:
                score = float(item["relevance_score"])

                # 低于阈值的文档认为不够相关，直接过滤掉。
                if score >= self.rerank_min_score:
                    ranked_items.append({
                        "doc": docs[item["index"]],
                        "score": score,
                    })

            # 兜底：若所有文档都被过滤，至少保留分数最高的一条
            if not ranked_items and ranked:
                best = ranked[0]
                ranked_items = [{
                    "doc": docs[best["index"]],
                    "score": float(best["relevance_score"]),
                }]

            print("\n========== Rerank 结果 ==========")
            for i, item in enumerate(ranked_items, 1):
                doc = item["doc"]
                score = item["score"]
                print(
                    f"  [{i}] score={score:.4f}  "
                    f"{doc.page_content[:60].replace(chr(10), ' ')}..."
                )

            return ranked_items

        except Exception as e:
            print(f"[Rerank] 异常，降级为原序: {e}")
            return [{"doc": doc, "score": None} for doc in docs[:keep]]

    def rerank_docs(self, query: str, docs: list, top_k: Optional[int] = None) -> list:
        """
        兼容旧接口：只返回文档，不返回分数。
        如果其他地方仍调用 rerank_docs，不会受 CRAG 改造影响。
        """
        ranked_items = self.rerank_docs_with_scores(query, docs, top_k=top_k)
        return [item["doc"] for item in ranked_items]

    # ── LLM 调用 ────────────────────────────────────────

    def call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        """
        调用 LLM（通过 OpenAI 兼容接口，指向 DashScope 或 Ollama）。
        返回模型输出的纯文本，失败时抛出异常由调用方处理。
        """
        response = self.llm_client.chat.completions.create(
            model=self.ollama_model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    # ── 联网搜索 ────────────────────────────────────────

    def search_web(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        使用 Tavily 进行联网搜索，适合天气/价格/新闻等实时问题。
        未配置 TAVILY_API_KEY 时直接返回失败，不抛出异常。
        """
        if not self.tavily_client:
            return {
                "success": False,
                "answer": "未配置联网搜索能力。",
                "results": [],
            }

        try:
            result = self.tavily_client.search(
                query=query,
                search_depth="basic",
                topic="general",
                max_results=max_results,
                include_answer="basic",
            )
        except Exception as exc:
            return {
                "success": False,
                "answer": f"联网检索暂时不可用：{exc}",
                "results": [],
            }

        answer = result.get("answer", "") or ""
        items = []
        for item in result.get("results", [])[:max_results]:
            items.append({
                "title":   item.get("title", "未命名结果"),
                "content": (item.get("content") or "").strip(),
                "url":     item.get("url", ""),
            })

        if not answer and not items:
            return {"success": True, "answer": "未检索到有用的联网结果。", "results": []}

        return {"success": True, "answer": answer.strip(), "results": items}


# =====================================================
# Tool 基类
# =====================================================

class BaseTool:
    name = "base_tool"
    description = "基础工具"

    def run(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


# =====================================================
# MemorySetTool：记忆写入
# =====================================================

class MemorySetTool(BaseTool):
    """
    用 LLM 从用户输入中抽取结构化记忆字段，写入 MemoryStore。
    抽取字段：name / location / context / preference_region /
              preference_style / preference_answer_style
    """
    name        = "memory_set"
    description = "使用 LLM 提取用户姓名、地点、背景、偏好，并写入记忆仓库"

    def __init__(self, memory_store: MemoryStore, kb: VectorKnowledgeBase):
        self.memory_store = memory_store
        self.kb = kb

    def _extract_memory_by_llm(self, text: str) -> Dict[str, str]:
        """让 LLM 从原始文本中抽取记忆字段，返回 dict（字段缺失时为空字符串）"""
        prompt = f"""你是一个用户记忆信息抽取器。
        请从用户输入中提取适合长期记住的信息，并严格输出 JSON。

        只允许输出以下 JSON 对象，不要输出解释，不要输出 markdown，不要输出代码块：

        {{
          "name": "",
          "location": "",
          "context": "",
          "preference_region": "",
          "preference_style": "",
          "preference_answer_style": ""
        }}

        抽取规则：
        1. 只提取用户明确说出的信息，不要猜测，不要补充。
        2. 如果某个字段没有被明确提到，必须填空字符串 ""。
        3. location 只写地点本身，例如"赣州"。
        4. context 写用户当前背景或任务，例如"种植脐橙""江西理工大学学生""正在做脐橙知识库""正在写毕业设计"。
        5. preference_region 只有在用户明确表达"以后优先按某地区回答"时才填写，例如"以后优先按赣南回答"。如果只是说自己在某地，不要填写。
        6. preference_style 只有在用户明确表达回答风格要求时才填写，例如"结构化回答""分点回答"。
        7. preference_answer_style 只有在用户明确表达回答表达偏好时才填写，例如"先给结论"。
        8. 不要因为 location 是"赣州"，就自动推断 preference_region 是"赣南/赣州"。

        下面是示例：

        输入：我叫蔡结，在赣州种植脐橙
        输出：
        {{
          "name": "蔡结",
          "location": "赣州",
          "context": "种植脐橙",
          "preference_region": "",
          "preference_style": "",
          "preference_answer_style": ""
        }}

        输入：我是蔡结，江西理工大学学生，在赣州种植脐橙
        输出：
        {{
          "name": "蔡结",
          "location": "赣州",
          "context": "江西理工大学学生；种植脐橙",
          "preference_region": "",
          "preference_style": "",
          "preference_answer_style": ""
        }}

        输入：以后优先按赣南这边的情况回答
        输出：
        {{
          "name": "",
          "location": "",
          "context": "",
          "preference_region": "赣南/赣州",
          "preference_style": "",
          "preference_answer_style": ""
        }}

        输入：以后分点回答，先给结论
        输出：
        {{
          "name": "",
          "location": "",
          "context": "",
          "preference_region": "",
          "preference_style": "结构化回答",
          "preference_answer_style": "先给结论"
        }}

        用户输入：
        {text}

        JSON："""

        try:
            raw = self.kb.call_llm(prompt, temperature=0.0).strip()
        except Exception:
            return {}

        # 去掉 LLM 可能输出的 ```json ... ``` 包裹
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        try:
            data = json.loads(raw)
        except Exception:
            return {}

        if not isinstance(data, dict):
            return {}

        # 只保留允许的字段，并确保值都是字符串
        allowed_keys = {
            "name", "location", "context",
            "preference_region", "preference_style", "preference_answer_style",
        }
        cleaned = {}
        for key in allowed_keys:
            value = data.get(key, "")
            if value is None:
                value = ""
            if not isinstance(value, str):
                value = str(value)
            cleaned[key] = value.strip()

        return cleaned

    def _validate_memory_fields(self, data: Dict[str, str]) -> Dict[str, str]:
        """
        对 LLM 抽取结果做程序级校验，过滤掉明显错误的值。
        例如 name 不应是一段话，location 不应包含逗号。
        """
        validated = dict(data)

        name = validated.get("name", "")
        if name and (len(name) > 20 or "，" in name or "," in name or "。" in name):
            validated["name"] = ""

        location = validated.get("location", "")
        if location and (len(location) > 20 or "，" in location or "," in location or "。" in location):
            validated["location"] = ""

        # context 允许稍长，但超过 50 字截断
        context = validated.get("context", "")
        if context and len(context) > 50:
            validated["context"] = context[:50]

        # preference 字段限长 30 字
        for key in ["preference_region", "preference_style", "preference_answer_style"]:
            value = validated.get(key, "")
            if value and len(value) > 30:
                validated[key] = value[:30]

        return validated

    def run(self, raw_text: str) -> Dict[str, Any]:
        """
        主入口：抽取 → 校验 → 写入记忆仓库 → 组织回复文案
        """
        text = raw_text.strip()

        extracted = self._extract_memory_by_llm(text)
        extracted = self._validate_memory_fields(extracted)

        # 将非空字段逐一写入记忆仓库，并记录被更新的字段名
        updated_fields = []
        for key, value in extracted.items():
            if value:
                self.memory_store.set(key, value)
                updated_fields.append(key)

        # 组织"我记住了"的确认文案
        memory = self.memory_store.all()
        parts = []
        if memory.get("name"):
            parts.append(f"你叫{memory['name']}")
        if memory.get("location"):
            parts.append(f"在{memory['location']}")
        if memory.get("context"):
            parts.append(memory["context"])

        if updated_fields:
            answer = "好的，我记住了。"
            if parts:
                answer += "，".join(parts) + "。"
        else:
            answer = "我没有提取到适合记住的新信息。"

        return {
            "tool_name":        self.name,
            "success":          True,
            "answer":           answer,
            "memory":           memory,
            "extracted_memory": extracted,
            "updated_fields":   updated_fields,
        }


# =====================================================
# MemoryQueryTool：记忆读取
# =====================================================

class MemoryQueryTool(BaseTool):
    """
    从 MemoryStore 读取用户身份、地点、背景信息并组织成自然语言回复。
    支持"我是谁""你叫什么""你在哪"等多种查询意图。
    """
    name        = "memory_query"
    description = "从记忆仓库读取用户身份、地点、背景、偏好信息"

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def run(self, query: str) -> Dict[str, Any]:
        q = query.strip()

        name     = self.memory_store.get("name")
        location = self.memory_store.get("location")
        context  = self.memory_store.get("context")

        # 检测用户想问的具体维度
        ask_identity = ("我是谁" in q) or ("你记得我是谁" in q)
        ask_name     = ("叫什么" in q) or ("名字" in q)
        ask_location = ("在哪" in q) or ("哪里人" in q) or ("在哪里" in q)
        ask_context  = ("做什么" in q) or ("干什么" in q) or ("背景" in q)

        if ask_identity or ask_name or ask_location or ask_context:
            parts = []

            if ask_identity:
                # 综合身份查询：姓名 + 地点 + 背景
                parts.append(f"你是{name}" if name else "我还不知道你是谁")
                if location:
                    parts.append(f"是{location}人")
                if context:
                    parts.append(context)
            else:
                # 分维度查询
                if ask_name:
                    parts.append(f"你叫{name}" if name else "我还不知道你的名字")
                if ask_location:
                    parts.append(f"你在{location}" if location else "我还不知道你在哪里")
                if ask_context:
                    parts.append(f"你目前是{context}" if context else "我还不知道你的背景信息")

            answer = "，".join(parts) + "。"
            return {
                "tool_name": self.name,
                "success":   True,
                "answer":    answer,
                "memory":    self.memory_store.all(),
            }

        # 兜底：展示全部已知记忆
        memory = self.memory_store.all()
        if memory:
            parts = []
            if memory.get("name"):
                parts.append(f"你叫{memory['name']}")
            if memory.get("location"):
                parts.append(f"是{memory['location']}人")
            if memory.get("context"):
                parts.append(memory["context"])
            answer = "我记得：" + "，".join(parts) + "。" if parts else "我记得一些信息。"
        else:
            answer = "我目前还没有记住你的相关信息。"

        return {
            "tool_name": self.name,
            "success":   True,
            "answer":    answer,
            "memory":    memory,
        }


# =====================================================
# KnowledgeQATool：知识库问答（核心工具）
# =====================================================

class KnowledgeQATool(BaseTool):
    """
    知识库问答流程：
      1. 检测问题是否含指代词 → 必要时改写成完整问题
      2. 增强检索（HyDE + 多查询扩展）→ 召回候选文档
      3. Rerank 精排 → 保留最相关文档
      4. 拼接 Prompt → 调用 LLM 生成回答
      5. 保存对话历史，供后续改写使用
    """
    name        = "knowledge_qa"
    description = "执行知识问答：必要时改写问题，检索向量库，并基于文档回答"

    def __init__(
        self,
        kb: VectorKnowledgeBase,
        memory_store: MemoryStore,
        max_history_rounds: int = 3,
    ):
        self.kb                 = kb
        self.memory_store       = memory_store
        self.max_history_rounds = max_history_rounds
        # 对话历史，用于问题改写时的上下文参考
        self.chat_history: List[Dict[str, str]] = []

    # ── 历史管理 ────────────────────────────────────────

    def save_history(self, question: str, answer: str):
        """追加一轮对话，超出窗口时丢弃最旧的"""
        self.chat_history.append({"question": question, "answer": answer})
        max_keep = max(self.max_history_rounds * 2, 6)
        if len(self.chat_history) > max_keep:
            self.chat_history = self.chat_history[-max_keep:]

    def _get_recent_history_text(self) -> str:
        """将最近几轮历史格式化为纯文本，供改写 Prompt 使用"""
        if not self.chat_history:
            return "无历史对话。"
        recent = self.chat_history[-self.max_history_rounds:]
        lines = []
        for turn in recent:
            lines.append(f"用户：{turn.get('question', '').strip()}")
            lines.append(f"助手：{turn.get('answer', '').strip()}")
        return "\n".join(lines)

    # ── 问题改写 ────────────────────────────────────────

    def _should_rewrite(self, query: str) -> bool:
        """
        判断问题是否含有指代词或省略，需要结合历史改写。
        例如："这两种病怎么防治" → 需要改写
             "脐橙什么时候施肥"   → 不需要改写
        """
        refer_words = [
            "这两种", "这两个", "这几个", "这些", "它们", "它",
            "这个", "那个", "上面说的", "前面说的", "刚才说的",
            "前者", "后者", "这种", "那种", "详细介绍一下", "展开讲讲",
            "再详细说说", "继续说", "那呢", "然后呢",
        ]
        return any(w in query for w in refer_words)

    def _rewrite_question(self, question: str) -> str:
        """
        结合历史对话，把含指代的问题改写成完整的独立检索句。
        改写失败时原样返回原问题。
        """
        history_text = self._get_recent_history_text()

        rewrite_prompt = f"""你是一个检索问题改写助手。
请根据历史对话，把用户当前问题改写成一个完整、明确、适合知识库检索的问题。

要求：
1. 如果当前问题中有代词、省略或指代，比如"这两种病""它们""这个方法"等，要结合历史补全。
2. 如果当前问题已经完整，则尽量保持原样。
3. 只输出改写后的问题，不要解释，不要回答。
4. 如果历史无法补全，就原样输出。

【历史对话】
{history_text}

【当前问题】
{question}

【改写后的问题】"""

        try:
            rewritten = self.kb.call_llm(rewrite_prompt, temperature=0.0)
            return rewritten if rewritten else question
        except Exception:
            return question

    # ── Prompt 构建 ─────────────────────────────────────

    def _format_context(self, docs) -> str:
        """把检索到的文档列表格式化为带来源标注的参考资料文本"""
        if not docs:
            return "未检索到相关知识。"

        blocks = []
        for i, doc in enumerate(docs, 1):
            source_file = doc.metadata.get("source_file", "未知来源")
            h1 = doc.metadata.get("h1", "")
            h2 = doc.metadata.get("h2", "")
            h3 = doc.metadata.get("h3", "")

            title_path  = " > ".join([x for x in [h1, h2, h3] if x]).strip()
            source_text = f"{source_file} | {title_path}" if title_path else source_file

            blocks.append(f"[资料{i} | 来源: {source_text}]\n{doc.page_content}")

        return "\n\n".join(blocks)

    def _grade_retrieval(self, ranked_items: List[Dict[str, Any]]) -> tuple[str, float]:
        """
        CRAG 检索质量评估：根据 rerank 的最高相关性分数判断检索质量。

        返回：
        - ("GOOD", best_score)：本地知识库资料可靠，直接使用本地资料
        - ("MEDIUM", best_score)：本地资料有一定相关性，但不够充分，需要联网补充
        - ("LOW", best_score)：本地资料不可靠，丢弃本地资料，改用联网搜索
        """
        if not ranked_items:
            return "LOW", 0.0

        scores = [
            item.get("score")
            for item in ranked_items
            if isinstance(item.get("score"), (int, float))
        ]

        # rerank 失败时可能没有分数；有文档但无分数，保守判断为 MEDIUM
        if not scores:
            return "MEDIUM", 0.0

        best_score = max(scores)

        if best_score >= self.kb.crag_good_score:
            return "GOOD", best_score
        if best_score < self.kb.crag_low_score:
            return "LOW", best_score
        return "MEDIUM", best_score

    def _format_web_context(self, web_result: Dict[str, Any]) -> str:
        """把联网搜索结果格式化为可送入 LLM 的参考资料文本。"""
        if not web_result.get("success"):
            return f"联网搜索失败：{web_result.get('answer', '')}"

        blocks = []

        answer = web_result.get("answer", "")
        if answer:
            blocks.append(f"[联网摘要]\n{answer}")

        for i, item in enumerate(web_result.get("results", []), 1):
            title = item.get("title", "未命名结果")
            content = item.get("content", "")
            url = item.get("url", "")
            blocks.append(
                f"[联网资料{i} | 来源: {title}]\n"
                f"{content}\n"
                f"链接：{url}"
            )

        return "\n\n".join(blocks) if blocks else "未检索到联网资料。"

    def _build_user_memory_text(self) -> str:
        """把用户记忆格式化为 Prompt 中的背景文本"""
        memory = self.memory_store.all()
        if not memory:
            return "无。"

        lines = []
        if memory.get("name"):
            lines.append(f"- 用户姓名：{memory['name']}")
        if memory.get("location"):
            lines.append(f"- 用户所在地：{memory['location']}")
        if memory.get("context"):
            lines.append(f"- 用户背景：{memory['context']}")
        if memory.get("preference_region"):
            lines.append(f"- 地区偏好：{memory['preference_region']}")
        if memory.get("preference_style"):
            lines.append(f"- 回答风格偏好：{memory['preference_style']}")
        if memory.get("preference_answer_style"):
            lines.append(f"- 回答表达偏好：{memory['preference_answer_style']}")

        return "\n".join(lines) if lines else "无。"

    def _build_qa_prompt(self, user_question: str, context: str) -> str:
        """拼装最终送给 LLM 的 QA Prompt"""
        memory_text = self._build_user_memory_text()

        return f"""你是一个脐橙种植知识库问答助手。
请严格根据参考资料回答问题。

要求：
1. 如果资料中没有答案，就明确说"知识库中没有找到相关信息"，不要编造。
2. 如果资料中只有部分答案，就先回答已知部分，并说明资料有限。
3. 回答尽量条理清晰，优先使用分点说明。
4. 用户背景仅用于帮助组织回答重点，不能替代参考资料本身。
5. 不要把用户个人信息当作知识库事实进行扩展推断。

【用户背景】
{memory_text}

【参考资料】
{context}

【问题】
{user_question}

【回答】"""

    # ── 主入口 ──────────────────────────────────────────

    def run(self, query: str, allow_web: bool = True) -> Dict[str, Any]:
        """
        CRAG 版 RAG 问答流程：
          1. 检测指代词，必要时改写问题
          2. 增强检索：HyDE + 多查询
          3. Rerank 精排，并保留 relevance_score
          4. CRAG 质量评估：GOOD / MEDIUM / LOW
          5. 根据质量选择资料来源：
             - GOOD：只用本地知识库
             - LOW：丢弃本地资料，改用联网搜索
             - MEDIUM：本地知识库 + 联网搜索合并
          6. 拼接 Prompt 调用 LLM
          7. 保存历史
        """
        # Step 1：含指代词时改写成完整问题
        # 例子：用户上一轮问“黄龙病怎么防治”，这一轮问“它会传染吗？”
        # “它”指代不清，所以要结合历史改写成“黄龙病会传染吗？”
        rewritten_query = (
            self._rewrite_question(query) if self._should_rewrite(query) else query
        )

        # Step 2：增强检索（HyDE + 多查询），失败时降级普通检索
        # 这一阶段只负责“尽可能多地找出候选资料”，还不判断哪些资料最可靠。
        try:
            candidate_docs = self.kb.search_enhanced(rewritten_query)
        except Exception:
            candidate_docs = self.kb.search_docs(rewritten_query)

        # Step 3：Rerank 精排，并保留分数
        # 向量检索召回的资料可能有噪声；
        # rerank 会重新判断“每个资料片段和问题到底有多相关”。
        ranked_items = self.kb.rerank_docs_with_scores(rewritten_query, candidate_docs)
        docs = [item["doc"] for item in ranked_items]

        # Step 4：CRAG 质量评估
        # CRAG 的核心想法是：先评估本地知识库资料质量，再决定要不要联网补充。
        # GOOD   = 本地资料足够好；
        # MEDIUM = 本地资料有用但可能不够；
        # LOW    = 本地资料质量低，优先考虑联网。
        crag_grade, best_score = self._grade_retrieval(ranked_items)

        print("\n========== CRAG 检索质量评估 ==========")
        print("原问题：", query)
        print("改写后问题：", rewritten_query)
        print("CRAG 等级：", crag_grade)
        print("最高相关性分数：", best_score)

        print("\n========== 本次本地检索命中内容 ==========")
        for i, doc in enumerate(docs, 1):
            print(f"\n--- 命中 chunk {i} ---")
            print(doc.page_content)
            print(doc.metadata)

        # Step 5：根据 CRAG 等级选择上下文来源
        # context 是最终要塞进 Prompt 的“参考资料”。
        # LLM 后面的回答应该基于这些资料，而不是凭空回答。
        local_context = self._format_context(docs) if docs else "未检索到本地知识库资料。"
        web_result = None
        used_web = False

        if crag_grade == "GOOD":
            # 本地资料质量高，只使用本地知识库，避免不必要的联网。
            final_context = f"【本地知识库资料】\n{local_context}"
            source_strategy = "local_only"

        elif crag_grade == "LOW":
            if allow_web:
                # 本地资料质量低，并且允许联网时，直接用联网搜索结果作为主要资料。
                web_result = self.kb.search_web(rewritten_query, max_results=3)
                web_context = self._format_web_context(web_result)
                used_web = True
                final_context = f"【联网搜索资料】\n{web_context}"
                source_strategy = "web_only"
            else:
                # 用户选择“仅本地”时，即使质量低也不能联网。
                final_context = f"【本地知识库资料】\n{local_context}"
                source_strategy = "local_only_web_disabled"

        else:
            # MEDIUM：本地资料有一定价值，但不够充分，联网补充
            if allow_web:
                # 中等质量时，把本地资料和联网资料合在一起给模型。
                web_result = self.kb.search_web(rewritten_query, max_results=3)
                web_context = self._format_web_context(web_result)
                used_web = True
                final_context = (
                    f"【本地知识库资料】\n{local_context}\n\n"
                    f"【联网补充资料】\n{web_context}"
                )
                source_strategy = "local_plus_web"
            else:
                # 如果禁止联网，就只能使用已有的本地资料。
                final_context = f"【本地知识库资料】\n{local_context}"
                source_strategy = "local_only_web_disabled"

        # Step 6：如果完全没有可用资料，直接返回
        # 这里避免在没有任何依据时让模型硬编答案。
        no_local = not docs
        no_web = (
            web_result is None
            or (
                not web_result.get("answer")
                and not web_result.get("results")
            )
        )

        if no_local and (not used_web or no_web):
            if allow_web:
                answer = "知识库中没有找到相关信息，联网搜索也没有获得有效补充。"
            else:
                answer = "知识库中没有找到相关信息。当前为仅本地模式，未进行联网搜索。"

            self.save_history(query, answer)
            return {
                "tool_name": self.name,
                "success": True,
                "answer": answer,
                "rewritten_question": rewritten_query,
                "source_documents": [],
                "crag_grade": crag_grade,
                "retrieval_best_score": best_score,
                "used_web": used_web,
                "source_strategy": source_strategy,
                "web_results": web_result.get("results", []) if web_result else [],
            }

        # Step 7：拼 Prompt → 调用 LLM
        # _build_qa_prompt 会把：
        # - 用户背景记忆
        # - 本地/联网参考资料
        # - 用户问题
        # 拼成一个完整提示词，再交给模型生成最终回答。
        prompt = self._build_qa_prompt(query, final_context)

        try:
            answer = self.kb.call_llm(prompt, temperature=0.1)
            if not answer:
                answer = "知识库中没有找到相关信息。"
        except Exception:
            answer = "本地模型暂时不可用。"

        # Step 8：保存本轮历史
        # 保存最近几轮问答，是为了下一轮遇到“它/这个/上面说的”等指代词时能改写问题。
        self.save_history(query, answer)

        # 返回结构化结果。
        # agent_service.py 通常主要使用 answer；
        # 其他字段可用于调试、评估或生成报告。
        return {
            "tool_name": self.name,
            "success": True,
            "answer": answer,
            "rewritten_question": rewritten_query,
            "source_documents": docs,
            "crag_grade": crag_grade,
            "retrieval_best_score": best_score,
            "used_web": used_web,
            "source_strategy": source_strategy,
            "web_results": web_result.get("results", []) if web_result else [],
        }


# =====================================================
# WebSearchTool：联网搜索
# =====================================================

class WebSearchTool(BaseTool):
    """
    调用 Tavily 联网搜索，适合天气、价格、新闻、政策等实时问题。
    把搜索结果格式化后直接作为回答返回（不再走向量库）。
    """
    name        = "web_search"
    description = "执行联网搜索，适合天气、价格、新闻、政策、市场等实时问题"

    def __init__(self, kb: VectorKnowledgeBase):
        self.kb = kb

    def _format_web_result(self, web_result: Dict[str, Any]) -> str:
        """把 Tavily 返回的结构化结果转为可读文本"""
        if not web_result.get("success"):
            return web_result.get("answer", "联网检索失败。")

        parts = []
        answer = web_result.get("answer", "")
        if answer:
            parts.append(f"联网摘要：{answer}")

        for item in web_result.get("results", []):
            title   = item.get("title", "未命名结果")
            content = item.get("content", "")
            url     = item.get("url", "")
            # 每条结果只展示前 180 字，避免 token 过多
            snippet = content[:180] + ("..." if len(content) > 180 else "")
            parts.append(f"{title}\n{snippet}\n链接：{url}")

        return "\n\n".join(parts) if parts else "未检索到有用的联网结果。"

    def run(self, query: str) -> Dict[str, Any]:
        web_result = self.kb.search_web(query=query, max_results=3)
        answer     = self._format_web_result(web_result)

        return {
            "tool_name":   self.name,
            "success":     web_result.get("success", True),
            "answer":      answer,
            "web_answer":  web_result.get("answer", ""),
            "web_results": web_result.get("results", []),
        }


# =====================================================
# ChatTool：简单闲聊
# =====================================================

class ChatTool(BaseTool):
    """
    处理打招呼、感谢、再见等固定闲聊，直接查 dict 返回，不走 LLM。
    """
    name        = "chat"
    description = "处理简单闲聊"

    def run(self, query: str) -> Dict[str, Any]:
        q = query.strip()
        chat_map = {
            "你好":     "你好，我可以帮你查询脐橙种植知识、病虫害防治、水肥管理等内容。",
            "您好":     "您好，我可以帮你查询脐橙种植知识、病虫害防治、水肥管理等内容。",
            "谢谢":     "不客气。",
            "多谢":     "不客气。",
            "再见":     "再见。",
            "拜拜":     "拜拜。",
            "你是谁":   "我是一个基于脐橙知识库的问答助手。",
            "你能做什么": "我可以根据你的脐橙知识库回答品种、栽培管理、施肥、灌溉、病虫害等问题。",
        }
        return {
            "tool_name": self.name,
            "success":   True,
            "answer":    chat_map.get(q, "你好。"),
        }


# =====================================================
# CommandTool：退出 / 清空命令
# =====================================================

class CommandTool(BaseTool):
    """
    处理系统命令：退出（exit/quit/q）、清空记忆、清空对话历史。
    """
    name        = "command"
    description = "处理退出、清空等命令"

    def __init__(self, memory_store: MemoryStore, knowledge_qa_tool: KnowledgeQATool):
        self.memory_store       = memory_store
        self.knowledge_qa_tool  = knowledge_qa_tool

    def run(self, command: str) -> Dict[str, Any]:
        cmd = command.strip().lower()

        if cmd in {"q", "quit", "exit"}:
            return {"tool_name": self.name, "success": True, "action": "exit", "answer": "exit"}

        if cmd in {"clear_memory", "reset_memory"}:
            self.memory_store.clear()
            return {"tool_name": self.name, "success": True, "action": "clear_memory", "answer": "已清空记忆。"}

        if cmd in {"clear_history", "reset_history"}:
            self.knowledge_qa_tool.chat_history.clear()
            return {"tool_name": self.name, "success": True, "action": "clear_history", "answer": "已清空对话历史。"}

        return {"tool_name": self.name, "success": False, "action": "unknown", "answer": f"未知命令：{command}"}


# =====================================================
# Router：意图路由
# =====================================================

class Router:
    """
    两级路由策略，优先级：规则匹配 > LLM 判断

    规则匹配速度快、无 LLM 消耗，适合高频固定模式。
    LLM 判断覆盖规则未命中的长尾情况，准确率更高。

    简单说：
    用户输入进来后，先判断它是“问知识库”、还是“问天气/新闻”、还是“闲聊/记忆/命令”。
    判断结果就是一个工具名，例如 knowledge_qa 或 web_search。
    """

    VALID_TOOLS = {"memory_set", "memory_query", "knowledge_qa", "web_search", "chat", "command"}

    def __init__(self, kb: VectorKnowledgeBase):
        self.kb = kb

    def _route_by_rules(self, text: str) -> Optional[str]:
        """
        基于关键词/正则的规则路由，命中时直接返回工具名，否则返回 None。
        按优先级从上到下匹配：command > memory_query > memory_set > web_search > chat
        """
        q = text.strip()
        q_lower = q.lower()

        # 退出/清空命令
        if q_lower in {"q", "quit", "exit", "clear_memory", "reset_memory", "clear_history", "reset_history"}:
            return "command"

        # 查询记忆（用户问"你记得我是谁"等）
        memory_query_patterns = [
            "我是谁", "你记得我是谁", "你还记得我是谁",
            "你记得我叫什么", "你还记得我叫什么",
            "你记得我的名字", "你还记得我的名字",
            "你记得我在哪", "你还记得我在哪",
            "你记得我做什么", "你还记得我做什么",
            "你记得我在哪里", "你还记得我在哪里",
        ]
        if any(p in q for p in memory_query_patterns):
            return "memory_query"

        # 写入记忆（用户告知个人信息）
        memory_set_patterns = [
            "记住", "请记住", "你要记得", "记一下", "帮我记住",
            "我叫", "我的名字是", "名字是", "我在", "来自",
            "以后回答", "以后请按", "你以后", "从现在开始", "今后",
        ]
        if any(p in q for p in memory_set_patterns):
            return "memory_set"
        if re.search(r"我是[\u4e00-\u9fa5A-Za-z]{2,20}人", q):
            return "memory_set"

        # 联网搜索（实时信息关键词）
        web_search_patterns = [
            "最新", "最近", "今天", "当前", "实时",
            "天气", "气温", "温度", "下雨", "降雨",
            "价格", "行情", "新闻", "政策", "市场",
            "本周", "本月", "今年", "近期", "现在",
        ]
        if any(p in q for p in web_search_patterns):
            return "web_search"

        # 简单闲聊（精确匹配，避免误伤）
        chat_patterns = {"你好", "您好", "谢谢", "多谢", "再见", "拜拜", "你是谁", "你能做什么"}
        if q in chat_patterns:
            return "chat"

        return None  # 规则未命中，交给 LLM 路由

    def _normalize_tool_name(self, text: str) -> str:
        """
        对 LLM 输出的工具名做容错处理：
        - 完全匹配 → 直接返回
        - 包含合法工具名 → 提取
        - 都不满足 → 兜底 knowledge_qa
        """
        cleaned = text.strip().lower()

        if cleaned in self.VALID_TOOLS:
            return cleaned

        for tool_name in self.VALID_TOOLS:
            if tool_name in cleaned:
                return tool_name

        return "knowledge_qa"

    def _route_by_llm(self, text: str) -> str:
        """
        让 LLM 根据用户输入判断应调用哪个工具。
        Prompt 包含大量 few-shot 示例，确保输出格式稳定。
        """
        prompt = f"""你是一个工具路由器。
    你的任务是：根据用户输入，判断应该调用哪个工具。

    你只能从以下工具中选择一个：
    - memory_set: 用户在告诉你个人信息、背景、偏好，希望你记住
    - memory_query: 用户在询问你是否记得他的名字、地点、背景等
    - knowledge_qa: 用户在询问脐橙知识、病虫害、水肥管理、栽培等专业问题
    - chat: 用户在闲聊、打招呼、感谢
    - command: 用户在发出退出、清空等命令
    - web_search: 用户在询问天气、价格、新闻、政策、市场等需要最新实时信息的问题

    分类原则：
    1. 用户在"提供信息给你记住"，归类为 memory_set
    2. 用户在"询问你记不记得他的信息"，归类为 memory_query
    3. 用户在"问天气、价格、新闻、政策、市场等实时信息"，归类为 web_search
    4. 用户在"问脐橙专业知识"，归类为 knowledge_qa
    5. 用户在"寒暄、感谢"，归类为 chat
    6. 用户在"退出、清空"，归类为 command
    7. 只输出一个工具名，不要解释，不要输出其他内容

    下面是一些示例：

    输入：脐橙什么时候施肥
    输出：knowledge_qa

    输入：脐橙黄龙病怎么防治
    输出：knowledge_qa

    输入：可以详细介绍这两种病吗
    输出：knowledge_qa

    输入：我叫蔡结
    输出：memory_set

    输入：我是赣州人，在赣州种植脐橙
    输出：memory_set

    输入：以后优先按赣南这边的情况回答
    输出：memory_set

    输入：你还记得我叫什么
    输出：memory_query

    输入：我是谁
    输出：memory_query

    输入：你记得我在哪吗
    输出：memory_query

    输入：帮我回忆一下我说过什么
    输出：memory_query

    输入：你好
    输出：chat

    输入：谢谢
    输出：chat

    输入：exit
    输出：command

    输入：clear_memory
    输出：command

    输入：脐橙会得什么病
    输出：knowledge_qa

    输入：木虱什么时候打药
    输出：knowledge_qa

    输入：我现在在做脐橙知识库整理，请记住
    输出：memory_set

    输入：你还记得我是在做什么项目吗
    输出：memory_query

    输入：今天赣州天气怎么样
    输出：web_search

    输入：最近脐橙市场价格如何
    输出：web_search

    输入：今年脐橙相关政策有什么变化
    输出：web_search

    现在开始分类。

    输入：{text}
    输出："""

        try:
            result = self.kb.call_llm(prompt, temperature=0.0)
            return self._normalize_tool_name(result)
        except Exception:
            return "knowledge_qa"   # 调用失败时兜底知识库问答

    def route(self, text: str) -> str:
        """对外统一路由入口：规则优先，规则不命中再走 LLM"""
        # 先用关键词规则判断，命中就直接返回，速度最快。
        rule_result = self._route_by_rules(text)
        if rule_result:
            return rule_result

        # 规则判断不了的复杂表达，再交给 LLM 判断应该调用哪个工具。
        return self._route_by_llm(text)


# =====================================================
# OrangeAgentService：应用服务层
# =====================================================

class OrangeAgentService:
    """
    最终应用层，组装所有组件并提供 dispatch() 接口：
      Router → 选择工具 → Tool.run() → 返回结果

    这个类是本文件内部真正的“总调度器”。
    RAGService.ask_question() 最终会调用它的 dispatch()。
    """

    def __init__(self, collection_name: str = "orange_knowledge"):
        # 共享组件
        # memory_store 保存运行期用户记忆；kb 保存知识库和模型调用能力。
        self.memory_store = MemoryStore()
        self.kb           = VectorKnowledgeBase(collection_name=collection_name)

        # 各工具实例
        # 每个工具都只负责一种任务，dispatch() 会根据 Router 的结果选择其中一个。
        self.memory_set_tool   = MemorySetTool(self.memory_store, self.kb)
        self.memory_query_tool = MemoryQueryTool(self.memory_store)
        self.knowledge_qa_tool = KnowledgeQATool(self.kb, self.memory_store)
        self.web_search_tool   = WebSearchTool(self.kb)
        self.chat_tool         = ChatTool()
        self.command_tool      = CommandTool(self.memory_store, self.knowledge_qa_tool)

        # 路由器
        self.router = Router(self.kb)

        # 工具注册表
        self.tools: Dict[str, BaseTool] = {
            "memory_set":   self.memory_set_tool,
            "memory_query": self.memory_query_tool,
            "knowledge_qa": self.knowledge_qa_tool,
            "web_search":   self.web_search_tool,
            "chat":         self.chat_tool,
            "command":      self.command_tool,
        }

    def dispatch(self, user_input: str, allow_web: bool = True) -> Dict[str, Any]:
        """
        主调度方法：
          1. Router 判断工具名
          2. 按工具名调用对应 Tool.run()
          3. 在结果中附加 routed_tool 字段

        allow_web：
          - True：允许 CRAG 在 LOW / MEDIUM 时联网补充
          - False：仅本地知识库，不允许联网搜索
        """
        text = user_input.strip()

        # 第一步：判断这句话应该交给哪个工具处理。
        tool_name = self.router.route(text)

        # 第二步：从工具注册表里取出对应工具实例。
        tool = self.tools[tool_name]

        if tool_name == "memory_set":
            # 用户告诉系统“我叫...”“以后按赣南情况回答...”时走这里。
            result = tool.run(raw_text=text)

        elif tool_name == "memory_query":
            # 用户问“你记得我是谁吗？”时走这里。
            result = tool.run(query=text)

        elif tool_name == "knowledge_qa":
            # 大多数脐橙种植、病虫害、水肥管理问题会走这里。
            result = tool.run(query=text, allow_web=allow_web)

        elif tool_name == "web_search":
            if allow_web:
                # 天气、价格、新闻、政策等实时信息走联网搜索。
                result = tool.run(query=text)
            else:
                # local 模式下禁止联网，即使 Router 判断它适合联网，也要拦住。
                result = {
                    "tool_name": "web_search",
                    "success": False,
                    "answer": "当前为仅本地知识库模式，已禁止联网搜索。",
                }

        elif tool_name == "chat":
            # 简单寒暄，不走大模型，直接返回固定回复。
            result = tool.run(query=text)

        elif tool_name == "command":
            # exit、clear_memory、clear_history 等命令。
            result = tool.run(command=text)

        else:
            result = {
                "tool_name": "unknown",
                "success": False,
                "answer": "没有找到合适的工具。",
            }

        # 附加路由结果，方便上层调试：这次到底用了哪个工具。
        result["routed_tool"] = tool_name
        return result


# =====================================================
# RAGService：兼容旧接口
# =====================================================

class RAGService:
    """
    对外保留统一入口，方便沿用原来的调用方式。
    内部已升级为 Router + Tool 调度架构。

    其他文件一般只需要创建 RAGService，然后调用 ask_question()。
    例如 agent_service.py 里的 search_knowledge_base 工具会调用这里。
    """

    def __init__(self, collection_name: str = "orange_knowledge"):
        # 内部真正干活的是 OrangeAgentService。
        self.agent = OrangeAgentService(collection_name=collection_name)

    def ask_question(self, query: str, allow_web: bool = True) -> Dict[str, Any]:
        # 对外暴露的最简单入口：
        # 输入用户问题，返回包含 answer、routed_tool、source_documents 等字段的结果。
        return self.agent.dispatch(query, allow_web=allow_web)


# =====================================================
# 命令行运行入口（本地调试用）
# =====================================================

if __name__ == "__main__":
    service = RAGService()
    print("Orange Agent 已启动，输入 q 退出。")

    while True:
        question = input("\n请输入问题: ").strip()
        if not question:
            continue

        result = service.ask_question(question)

        # 收到 exit 命令时退出循环
        if result.get("routed_tool") == "command" and result.get("answer") == "exit":
            break

        print("\n路由工具：")
        print(result.get("routed_tool"))

        if result.get("rewritten_question"):
            print("\n检索问题：")
            print(result["rewritten_question"])

        print("\n回答：")
        print(result["answer"])
