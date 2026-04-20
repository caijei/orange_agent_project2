import json
import os
import re

from tavily import TavilyClient
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))


# =====================================
# Store 层
# =====================================

class MemoryStore:
    """
    简单的内存记忆仓库。
    后续你可以替换成 json / sqlite / mysql。
    """

    def __init__(self):
        self.data: Dict[str, str] = {}

    def set(self, key: str, value: str):
        if value:
            self.data[key] = value

    def get(self, key: str, default: str = "") -> str:
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.data and bool(self.data[key])

    def all(self) -> Dict[str, str]:
        return dict(self.data)

    def clear(self):
        self.data.clear()


class VectorKnowledgeBase:
    """
        知识库能力：
        - 向量检索
        - 调用本地模型
        - 联网检索
        """

    def __init__(self, collection_name: str = "orange_knowledge"):
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        self.ollama_model = os.getenv("OLLAMA_MODEL")
        self.search_k = int(os.getenv("RAG_SEARCH_K", "4"))
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

        self.embeddings = DashScopeEmbeddings(dashscope_api_key=self.api_key)

        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

        self.llm_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.ollama_base_url,
        )

        self.tavily_client = TavilyClient(api_key=self.tavily_api_key) if self.tavily_api_key else None

    def search_docs(self, query: str, k: Optional[int] = None):
        top_k = k if k is not None else self.search_k
        return self.vectordb.similarity_search(query, k=top_k)

    def search_with_multi_query(self, question: str, k: Optional[int] = None) -> list:
        """多查询扩展：生成3个不同角度的问题分别检索，合并去重后按命中频率排序"""
        top_k = k if k is not None else self.search_k
        prompt = f"""请从3个不同角度改写以下问题，用于农业知识库检索。
每行输出一个改写问题，只输出问题本身，不要编号和解释。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。

原问题：{question}
改写："""
        try:
            result = self.call_llm(prompt, temperature=0.3)
            expanded = [q.strip() for q in result.strip().split("\n") if q.strip()]
        except Exception:
            expanded = []

        all_queries = [question] + expanded[:3]
        print(all_queries)
        # 统计每个文档被命中的次数，命中越多排越前
        hit_count: Dict[str, int] = {}
        doc_map: Dict[str, Any] = {}
        for q in all_queries:
            try:
                for doc in self.vectordb.similarity_search(q, k=top_k):
                    key = doc.page_content[:80]
                    hit_count[key] = hit_count.get(key, 0) + 1
                    doc_map[key] = doc
            except Exception:
                continue

        sorted_keys = sorted(hit_count.keys(), key=lambda x: -hit_count[x])
        return [doc_map[k] for k in sorted_keys][: top_k * 2]

    def search_with_hyde(self, question: str, k: Optional[int] = None) -> list:
        """HyDE：先生成假设答案，用答案向量检索，语义比问题更贴近文档"""
        top_k = k if k is not None else self.search_k
        prompt = f"""请用2-3句话写一个关于以下问题的专业假设性答案，用于辅助农业知识检索。
内容要包含相关专业术语，不需要完全准确。用于向量检索的查询文档（不要分析过程）

问题：{question}
假设答案："""
        try:
            hypothetical = self.call_llm(prompt, temperature=0.1)
            if not hypothetical:
                return self.search_docs(question, k=top_k)
            return self.vectordb.similarity_search(hypothetical, k=top_k)
        except Exception:
            return self.search_docs(question, k=top_k)

    def search_enhanced(self, question: str, k: Optional[int] = None) -> list:
        """多查询 + HyDE 结合：并行检索后合并，HyDE结果优先"""
        top_k = k if k is not None else self.search_k
        try:
            hyde_docs = self.search_with_hyde(question, k=top_k)
        except Exception:
            hyde_docs = []
        #print(hyde_docs)
        try:
            multi_docs = self.search_with_multi_query(question, k=top_k)
        except Exception:
            multi_docs = []
        #print(multi_docs)
        seen: set = set()
        merged = []
        for doc in hyde_docs + multi_docs:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        return merged[: top_k * 2] if merged else self.search_docs(question, k=top_k)

    def call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        response = self.llm_client.chat.completions.create(
            model=self.ollama_model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    def search_web(self, query: str, max_results: int = 3) -> Dict[str, Any]:
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
                "title": item.get("title", "未命名结果"),
                "content": (item.get("content") or "").strip(),
                "url": item.get("url", ""),
            })

        if not answer and not items:
            return {
                "success": True,
                "answer": "未检索到有用的联网结果。",
                "results": [],
            }

        return {
            "success": True,
            "answer": answer.strip(),
            "results": items,
        }


# =====================================
# Tool 基类
# =====================================

class BaseTool:
    name = "base_tool"
    description = "基础工具"

    def run(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


# =====================================
# memory_set_tool
# =====================================

class MemorySetTool(BaseTool):
    name = "memory_set"
    description = "使用 LLM 提取用户姓名、地点、背景、偏好，并写入记忆仓库"

    def __init__(self, memory_store: MemoryStore, kb: VectorKnowledgeBase):
        self.memory_store = memory_store
        self.kb = kb

    def _extract_memory_by_llm(self, text: str) -> Dict[str, str]:
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
        3. location 只写地点本身，例如“赣州”。
        4. context 写用户当前背景或任务，例如“种植脐橙”“江西理工大学学生”“正在做脐橙知识库”“正在写毕业设计”。
        5. preference_region 只有在用户明确表达“以后优先按某地区回答”时才填写，例如“以后优先按赣南回答”。如果只是说自己在某地，不要填写。
        6. preference_style 只有在用户明确表达回答风格要求时才填写，例如“结构化回答”“分点回答”。
        7. preference_answer_style 只有在用户明确表达回答表达偏好时才填写，例如“先给结论”。
        8. 不要因为 location 是“赣州”，就自动推断 preference_region 是“赣南/赣州”。

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
        # print(raw)
        # 去掉可能的 ```json ... ``` 包裹
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        try:
            data = json.loads(raw)
        except Exception:
            return {}

        if not isinstance(data, dict):
            return {}

        allowed_keys = {
            "name",
            "location",
            "context",
            "preference_region",
            "preference_style",
            "preference_answer_style",
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
        对 LLM 抽取结果做最基本的程序校验，避免明显错误写入。
        """
        validated = dict(data)

        # name 简单约束：不能太长，不能明显是句子
        name = validated.get("name", "")
        if name:
            if len(name) > 20 or "，" in name or "," in name or "。" in name:
                validated["name"] = ""

        # location 简单约束
        location = validated.get("location", "")
        if location:
            if len(location) > 20 or "，" in location or "," in location or "。" in location:
                validated["location"] = ""

        # context 可稍长，但不要过长
        context = validated.get("context", "")
        if context and len(context) > 50:
            validated["context"] = context[:50]

        # preference 字段限制长度
        for key in ["preference_region", "preference_style", "preference_answer_style"]:
            value = validated.get(key, "")
            if value and len(value) > 30:
                validated[key] = value[:30]

        return validated

    def run(self, raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()

        extracted = self._extract_memory_by_llm(text)
        extracted = self._validate_memory_fields(extracted)

        updated_fields = []

        for key, value in extracted.items():
            if value:
                self.memory_store.set(key, value)
                updated_fields.append(key)

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
            "tool_name": self.name,
            "success": True,
            "answer": answer,
            "memory": memory,
            "extracted_memory": extracted,
            "updated_fields": updated_fields,
        }


# =====================================
# memory_query_tool
# =====================================

class MemoryQueryTool(BaseTool):
    name = "memory_query"
    description = "从记忆仓库读取用户身份、地点、背景、偏好信息"

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def run(self, query: str) -> Dict[str, Any]:
        q = query.strip()

        name = self.memory_store.get("name")
        location = self.memory_store.get("location")
        context = self.memory_store.get("context")

        ask_identity = ("我是谁" in q) or ("你记得我是谁" in q)
        ask_name = ("叫什么" in q) or ("名字" in q)
        ask_location = ("在哪" in q) or ("哪里人" in q) or ("在哪里" in q)
        ask_context = ("做什么" in q) or ("干什么" in q) or ("背景" in q)

        # 如果是复合查询，优先组合回答
        if ask_identity or ask_name or ask_location or ask_context:
            parts = []

            if ask_identity:
                if name:
                    parts.append(f"你是{name}")
                else:
                    parts.append("我还不知道你是谁")

                if location:
                    parts.append(f"是{location}人")
                if context:
                    parts.append(context)

            else:
                if ask_name:
                    parts.append(f"你叫{name}" if name else "我还不知道你的名字")
                if ask_location:
                    parts.append(f"你在{location}" if location else "我还不知道你在哪里")
                if ask_context:
                    parts.append(f"你目前是{context}" if context else "我还不知道你的背景信息")

            answer = "，".join(parts) + "。"
            return {
                "tool_name": self.name,
                "success": True,
                "answer": answer,
                "memory": self.memory_store.all(),
            }

        # 默认兜底
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
            "success": True,
            "answer": answer,
            "memory": memory,
        }


# =====================================
# knowledge_qa_tool
# =====================================

class KnowledgeQATool(BaseTool):
    name = "knowledge_qa"
    description = "执行知识问答：必要时改写问题，检索向量库，并基于文档回答"

    def __init__(self, kb: VectorKnowledgeBase, memory_store: MemoryStore, max_history_rounds: int = 3):
        self.kb = kb
        self.memory_store = memory_store
        self.max_history_rounds = max_history_rounds
        self.chat_history: List[Dict[str, str]] = []

    def save_history(self, question: str, answer: str):
        self.chat_history.append({
            "question": question,
            "answer": answer,
        })
        max_keep = max(self.max_history_rounds * 2, 6)
        if len(self.chat_history) > max_keep:
            self.chat_history = self.chat_history[-max_keep:]

    def _get_recent_history_text(self) -> str:
        if not self.chat_history:
            return "无历史对话。"

        recent_history = self.chat_history[-self.max_history_rounds:]
        lines = []
        for turn in recent_history:
            lines.append(f"用户：{turn.get('question', '').strip()}")
            lines.append(f"助手：{turn.get('answer', '').strip()}")
        return "\n".join(lines)

    def _should_rewrite(self, query: str) -> bool:
        refer_words = [
            "这两种", "这两个", "这几个", "这些", "它们", "它",
            "这个", "那个", "上面说的", "前面说的", "刚才说的",
            "前者", "后者", "这种", "那种", "详细介绍一下", "展开讲讲",
            "再详细说说", "继续说", "那呢", "然后呢"
        ]
        return any(w in query for w in refer_words)

    def _rewrite_question(self, question: str) -> str:
        history_text = self._get_recent_history_text()

        rewrite_prompt = f"""你是一个检索问题改写助手。
请根据历史对话，把用户当前问题改写成一个完整、明确、适合知识库检索的问题。

要求：
1. 如果当前问题中有代词、省略或指代，比如“这两种病”“它们”“这个方法”等，要结合历史补全。
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

    def _format_context(self, docs) -> str:
        if not docs:
            return "未检索到相关知识。"

        blocks = []
        for i, doc in enumerate(docs, 1):
            source_file = doc.metadata.get("source_file", "未知来源")
            h1 = doc.metadata.get("h1", "")
            h2 = doc.metadata.get("h2", "")
            h3 = doc.metadata.get("h3", "")

            title_path = " > ".join([x for x in [h1, h2, h3] if x]).strip()
            source_text = source_file
            if title_path:
                source_text = f"{source_file} | {title_path}"

            blocks.append(f"[资料{i} | 来源: {source_text}]\n{doc.page_content}")

        return "\n\n".join(blocks)

    def _build_user_memory_text(self) -> str:
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
        memory_text = self._build_user_memory_text()

        return f"""你是一个脐橙种植知识库问答助手。
请严格根据参考资料回答问题。

要求：
1. 如果资料中没有答案，就明确说“知识库中没有找到相关信息”，不要编造。
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

    def run(self, query: str) -> Dict[str, Any]:
        rewritten_query = self._rewrite_question(query) if self._should_rewrite(query) else query

        # ✅ 使用增强检索（多查询扩展 + HyDE），fallback 到普通检索
        try:
            docs = self.kb.search_enhanced(rewritten_query)
        except Exception:
            docs = self.kb.search_docs(rewritten_query)

        if not docs:
            answer = "知识库中没有找到相关信息。"
            self.save_history(query, answer)
            return {
                "tool_name": self.name,
                "success": True,
                "answer": answer,
                "rewritten_question": rewritten_query,
                "source_documents": [],
            }

        context = self._format_context(docs)
        prompt = self._build_qa_prompt(query, context)

        try:
            answer = self.kb.call_llm(prompt, temperature=0.1)
            if not answer:
                answer = "知识库中没有找到相关信息。"
        except Exception:
            answer = "本地模型暂时不可用。"

        self.save_history(query, answer)

        return {
            "tool_name": self.name,
            "success": True,
            "answer": answer,
            "rewritten_question": rewritten_query,
            "source_documents": docs,
        }


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "执行联网搜索，适合天气、价格、新闻、政策、市场等实时问题"

    def __init__(self, kb: VectorKnowledgeBase):
        self.kb = kb

    def _format_web_result(self, web_result: Dict[str, Any]) -> str:
        if not web_result.get("success"):
            return web_result.get("answer", "联网检索失败。")

        parts = []
        answer = web_result.get("answer", "")
        if answer:
            parts.append(f"联网摘要：{answer}")

        for item in web_result.get("results", []):
            title = item.get("title", "未命名结果")
            content = item.get("content", "")
            url = item.get("url", "")
            snippet = content[:180] + ("..." if len(content) > 180 else "")
            parts.append(f"{title}\n{snippet}\n链接：{url}")

        return "\n\n".join(parts) if parts else "未检索到有用的联网结果。"

    def run(self, query: str) -> Dict[str, Any]:
        web_result = self.kb.search_web(query=query, max_results=3)
        answer = self._format_web_result(web_result)

        return {
            "tool_name": self.name,
            "success": web_result.get("success", True),
            "answer": answer,
            "web_answer": web_result.get("answer", ""),
            "web_results": web_result.get("results", []),
        }


# =====================================
# chat_tool
# =====================================

class ChatTool(BaseTool):
    name = "chat"
    description = "处理简单闲聊"

    def run(self, query: str) -> Dict[str, Any]:
        q = query.strip()
        chat_map = {
            "你好": "你好，我可以帮你查询脐橙种植知识、病虫害防治、水肥管理等内容。",
            "您好": "您好，我可以帮你查询脐橙种植知识、病虫害防治、水肥管理等内容。",
            "谢谢": "不客气。",
            "多谢": "不客气。",
            "再见": "再见。",
            "拜拜": "拜拜。",
            "你是谁": "我是一个基于脐橙知识库的问答助手。",
            "你能做什么": "我可以根据你的脐橙知识库回答品种、栽培管理、施肥、灌溉、病虫害等问题。",
        }

        return {
            "tool_name": self.name,
            "success": True,
            "answer": chat_map.get(q, "你好。"),
        }


# =====================================
# command_tool
# =====================================

class CommandTool(BaseTool):
    name = "command"
    description = "处理退出、清空等命令"

    def __init__(self, memory_store: MemoryStore, knowledge_qa_tool: KnowledgeQATool):
        self.memory_store = memory_store
        self.knowledge_qa_tool = knowledge_qa_tool

    def run(self, command: str) -> Dict[str, Any]:
        cmd = command.strip().lower()

        if cmd in {"q", "quit", "exit"}:
            return {
                "tool_name": self.name,
                "success": True,
                "action": "exit",
                "answer": "exit",
            }

        if cmd in {"clear_memory", "reset_memory"}:
            self.memory_store.clear()
            return {
                "tool_name": self.name,
                "success": True,
                "action": "clear_memory",
                "answer": "已清空记忆。",
            }

        if cmd in {"clear_history", "reset_history"}:
            self.knowledge_qa_tool.chat_history.clear()
            return {
                "tool_name": self.name,
                "success": True,
                "action": "clear_history",
                "answer": "已清空对话历史。",
            }

        return {
            "tool_name": self.name,
            "success": False,
            "action": "unknown",
            "answer": f"未知命令：{command}",
        }


# =====================================
# Router 层
# =====================================

class Router:
    """
    路由策略：
    1. 先规则判断
    2. 规则不确定时，再走 LLM Router
    3. 最终输出合法工具名
    """

    VALID_TOOLS = {"memory_set", "memory_query", "knowledge_qa", "web_search", "chat", "command"}

    def __init__(self, kb: VectorKnowledgeBase):
        self.kb = kb

    def _route_by_rules(self, text: str) -> Optional[str]:
        q = text.strip()
        q_lower = q.lower()

        if q_lower in {"q", "quit", "exit", "clear_memory", "reset_memory", "clear_history", "reset_history"}:
            return "command"

        memory_query_patterns = [
            "我是谁",
            "你记得我是谁",
            "你还记得我是谁",
            "你记得我叫什么",
            "你还记得我叫什么",
            "你记得我的名字",
            "你还记得我的名字",
            "你记得我在哪",
            "你还记得我在哪",
            "你记得我做什么",
            "你还记得我做什么",
            "你记得我在哪里",
            "你还记得我在哪里",
        ]
        if any(p in q for p in memory_query_patterns):
            return "memory_query"

        memory_set_patterns = [
            "记住", "请记住", "你要记得", "记一下", "帮我记住",
            "我叫", "我的名字是", "名字是", "我在", "来自",
            "以后回答", "以后请按", "你以后", "从现在开始", "今后",
        ]
        if any(p in q for p in memory_set_patterns):
            return "memory_set"

        if re.search(r"我是[\u4e00-\u9fa5A-Za-z]{2,20}人", q):
            return "memory_set"

        web_search_patterns = [
            "最新", "最近", "今天", "当前", "实时",
            "天气", "气温", "温度", "下雨", "降雨",
            "价格", "行情", "新闻", "政策", "市场",
            "本周", "本月", "今年", "近期", "现在",
        ]
        if any(p in q for p in web_search_patterns):
            return "web_search"

        chat_patterns = {
            "你好", "您好", "谢谢", "多谢", "再见", "拜拜",
            "你是谁", "你能做什么"
        }
        if q in chat_patterns:
            return "chat"

        return None

    def _normalize_tool_name(self, text: str) -> str:
        cleaned = text.strip().lower()

        if cleaned in self.VALID_TOOLS:
            return cleaned

        for tool_name in self.VALID_TOOLS:
            if tool_name in cleaned:
                return tool_name

        return "knowledge_qa"

    def _route_by_llm(self, text: str) -> str:
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
    1. 用户在“提供信息给你记住”，归类为 memory_set
    2. 用户在“询问你记不记得他的信息”，归类为 memory_query
    3. 用户在“问天气、价格、新闻、政策、市场等实时信息”，归类为 web_search
    4. 用户在“问脐橙专业知识”，归类为 knowledge_qa
    5. 用户在“寒暄、感谢”，归类为 chat
    6. 用户在“退出、清空”，归类为 command
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
            return "knowledge_qa"

    def route(self, text: str) -> str:
        rule_result = self._route_by_rules(text)
        if rule_result:
            return rule_result

        return self._route_by_llm(text)


# =====================================
# 应用服务：Router + Tools
# =====================================

class OrangeAgentService:
    """
    最终应用层：
    - Router 决定调用哪个 tool
    - Tool 执行具体任务
    """

    def __init__(self, collection_name: str = "orange_knowledge"):
        self.memory_store = MemoryStore()
        self.kb = VectorKnowledgeBase(collection_name=collection_name)

        self.memory_set_tool = MemorySetTool(self.memory_store, self.kb)
        self.memory_query_tool = MemoryQueryTool(self.memory_store)
        self.knowledge_qa_tool = KnowledgeQATool(self.kb, self.memory_store)
        self.web_search_tool = WebSearchTool(self.kb)
        self.chat_tool = ChatTool()
        self.command_tool = CommandTool(self.memory_store, self.knowledge_qa_tool)

        self.router = Router(self.kb)

        self.tools: Dict[str, BaseTool] = {
            "memory_set": self.memory_set_tool,
            "memory_query": self.memory_query_tool,
            "knowledge_qa": self.knowledge_qa_tool,
            "web_search": self.web_search_tool,
            "chat": self.chat_tool,
            "command": self.command_tool,
        }

    def dispatch(self, user_input: str) -> Dict[str, Any]:
        text = user_input.strip()
        tool_name = self.router.route(text)
        tool = self.tools[tool_name]

        if tool_name == "memory_set":
            result = tool.run(raw_text=text)
        elif tool_name == "memory_query":
            result = tool.run(query=text)
        elif tool_name == "knowledge_qa":
            result = tool.run(query=text)
        elif tool_name == "web_search":
            result = tool.run(query=text)
        elif tool_name == "chat":
            result = tool.run(query=text)
        elif tool_name == "command":
            result = tool.run(command=text)
        else:
            result = {
                "tool_name": "unknown",
                "success": False,
                "answer": "没有找到合适的工具。",
            }

        result["routed_tool"] = tool_name
        return result


# =====================================
# 兼容旧接口：类似原来的 ask_question
# =====================================

class RAGService:
    """
    对外保留一个统一入口，方便你继续沿用原来的调用方式。
    现在内部已经不是单一 RAG，而是 Router + Tool 调度。
    """

    def __init__(self, collection_name: str = "orange_knowledge"):
        self.agent = OrangeAgentService(collection_name=collection_name)

    def ask_question(self, query: str) -> Dict[str, Any]:
        return self.agent.dispatch(query)


# =====================================
# 运行入口
# =====================================

if __name__ == "__main__":
    service = RAGService()
    print("Orange Agent 已启动，输入 q 退出。")

    while True:
        question = input("\n请输入问题: ").strip()
        if not question:
            continue

        result = service.ask_question(question)

        if result.get("routed_tool") == "command" and result.get("answer") == "exit":
            break

        print("\n路由工具：")
        print(result.get("routed_tool"))

        if result.get("rewritten_question"):
            print("\n检索问题：")
            print(result["rewritten_question"])

        print("\n回答：")
        print(result["answer"])