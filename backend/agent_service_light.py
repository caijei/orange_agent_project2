import asyncio
import os
import re
from pathlib import Path
from typing import List

from dashscope import MultiModalConversation
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

from rag_service import RAGService

load_dotenv(Path(__file__).with_name(".env"))


class OrangeAgent:
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

        self.rag_service = RAGService()
        self.llm_client = OpenAI(api_key="ollama", base_url=self.ollama_base_url)
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key) if self.tavily_api_key else None

    # =========================
    # 文本 / 图片辅助函数
    # =========================

    def _extract_image_paths(self, user_message: str) -> List[str]:
        quoted_paths = re.findall(
            r"'([^']+\.(?:jpg|jpeg|png|webp))'",
            user_message,
            flags=re.IGNORECASE,
        )
        unique_paths = []
        seen = set()
        for path in quoted_paths:
            normalized = path.strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_paths.append(normalized)
        return unique_paths

    def _strip_system_image_hint(self, user_message: str) -> str:
        cleaned = re.sub(
            r"[。.]?\s*请综合分析这\s*\d+\s*张图片[^:：]*[:：]\s*(?:'[^']+'\s*,?\s*)+[。.]?",
            "",
            user_message,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    def _needs_web_search(self, query: str) -> bool:
        web_keywords = [
            "最新",
            "最近",
            "今天",
            "当前",
            "实时",
            "天气",
            "气温",
            "下雨",
            "降雨",
            "温度",
            "价格",
            "行情",
            "新闻",
            "政策",
            "市场",
            "本周",
            "本月",
            "今年",
            "近期",
            "现在",
        ]
        return any(keyword in query for keyword in web_keywords)

    def _is_orange_domain_query(self, query: str) -> bool:
        domain_keywords = [
            "脐橙",
            "柑橘",
            "果园",
            "园地",
            "苗木",
            "修剪",
            "施肥",
            "灌溉",
            "病害",
            "虫害",
            "黄龙病",
            "溃疡病",
            "木虱",
            "红蜘蛛",
            "采后",
            "贮藏",
            "品种",
            "果实",
        ]
        return any(keyword in query for keyword in domain_keywords)

    def _should_use_web(self, query: str, search_mode: str = "auto") -> bool:
        mode = (search_mode or "auto").lower()

        if mode == "web":
            return True
        if mode == "local":
            return False

        return self._needs_web_search(query)
    # =========================
    # 联网检索
    # =========================

    def _search_web(self, query: str) -> str:
        if not self.tavily_client:
            return "未配置联网搜索能力。"

        try:
            result = self.tavily_client.search(
                query=query,
                search_depth="basic",
                topic="general",
                max_results=3,
                include_answer="basic",
            )
        except Exception as exc:
            return f"联网检索暂时不可用：{exc}"

        parts = []
        answer = result.get("answer")
        if answer:
            parts.append(f"联网摘要：{answer}")

        for item in result.get("results", [])[:3]:
            title = item.get("title", "未命名结果")
            content = (item.get("content") or "").strip()
            url = item.get("url", "")
            snippet = content[:180] + ("..." if len(content) > 180 else "")
            parts.append(f"{title}\n{snippet}\n链接：{url}")

        return "\n\n".join(parts) if parts else "未检索到有用的联网结果。"

    # =========================
    # 图片诊断
    # =========================

    def image_diagnose(self, image_path: str) -> str:
        clean_path = image_path.strip(' "\'')
        if not os.path.exists(clean_path):
            return f"找不到图片文件：{clean_path}，请检查路径是否正确。"

        abs_path = os.path.abspath(clean_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{abs_path}"},
                    {
                        "text": (
                            "你是专业的脐橙病虫害诊断专家。请识别图片中的脐橙叶片、果实或树体是否存在病害、虫害或生理异常。"
                            "请按“现象判断、可能原因、处理建议、是否需要线下复核”四部分给出结论。"
                        )
                    },
                ],
            }
        ]

        try:
            response = MultiModalConversation.call(
                model="qwen-vl-plus",
                messages=messages,
                api_key=self.api_key,
            )
            if response.status_code != 200:
                return f"图片诊断失败：错误码 {response.code}，原因：{response.message}"

            content = response.output.choices[0].message.content
            if isinstance(content, list):
                return "".join(item.get("text", "") for item in content)
            return str(content)
        except Exception as exc:
            return f"调用图片诊断模型时发生错误：{exc}"

    # =========================
    # RAG 新架构访问辅助
    # =========================

    def _get_qa_tool(self):
        return self.rag_service.agent.knowledge_qa_tool

    def _update_rag_history(self, question: str, answer: str):
        qa_tool = self._get_qa_tool()
        qa_tool.save_history(question, answer)

    def _stream_llm_answer(self, question: str, source_docs):
        """
        基于 knowledge_qa_tool 的 prompt 构造逻辑，做流式回答。
        """
        qa_tool = self._get_qa_tool()
        context = qa_tool._format_context(source_docs)
        prompt = qa_tool._build_qa_prompt(question, context)

        response = self.llm_client.chat.completions.create(
            model=self.ollama_model,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        accumulated = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                accumulated += delta
                yield delta

        if not accumulated.strip():
            accumulated = "抱歉，当前本地问答模型暂时没有返回内容，请稍后再试。"
            yield accumulated

        self._update_rag_history(question, accumulated)

    # =========================
    # 回答整合
    # =========================

    def _synthesize_answer(self, question: str, rag_answer: str, web_result: str, image_results: List[str]) -> str:
        sections = []

        if rag_answer:
            sections.append(f"【知识库结果】\n{rag_answer}")

        if web_result:
            sections.append(f"【联网补充】\n{web_result}")

        if image_results:
            for index, result in enumerate(image_results, 1):
                sections.append(f"【图片分析 {index}】\n{result}")

        if not sections:
            return "抱歉，我暂时没有检索到可用信息。"

        if rag_answer and not web_result and not image_results:
            return rag_answer

        prompt = (
            "你是专业的脐橙种植助手。请根据给定材料整合出一份清晰、可执行的中文答复。\n"
            "要求：\n"
            "1. 优先保留知识库中的稳定结论。\n"
            "2. 联网结果仅作补充，若与知识库不一致，要明确提示。\n"
            "3. 如果包含图片分析，请单独说明它来自图像判断，建议用户结合实地症状复核。\n"
            "4. 不要编造材料中没有的信息。\n\n"
            f"【用户问题】\n{question}\n\n"
            f"{chr(10).join(sections)}\n\n"
            "【最终答复】"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.ollama_model,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content or ""
            return answer.strip() or "\n\n".join(sections)
        except Exception:
            return "\n\n".join(sections)

    # =========================
    # 同步问答
    # =========================

    def chat(self, user_message: str, search_mode: str = "auto") -> str:
        image_paths = self._extract_image_paths(user_message)
        clean_query = self._strip_system_image_hint(user_message)

        if not clean_query and image_paths:
            clean_query = "请分析这些脐橙图片中的病虫害或异常症状。"

        use_web = bool(clean_query) and self._should_use_web(clean_query, search_mode=search_mode)

        rag_answer = ""
        if clean_query and self._is_orange_domain_query(clean_query):
            rag_result = self.rag_service.ask_question(clean_query)
            rag_answer = rag_result.get("answer", "")

        # local 模式下不联网
        web_result = ""
        if clean_query and use_web:
            web_result = self._search_web(clean_query)

        image_results = []
        for path in image_paths:
            image_results.append(self.image_diagnose(path))

        # 如果 local 模式且前面没命中脐橙领域问题，做一次兜底知识库查询
        if not rag_answer and not web_result and not image_results:
            rag_result = self.rag_service.ask_question(clean_query or user_message)
            rag_answer = rag_result.get("answer", "")

        return self._synthesize_answer(
            clean_query or user_message,
            rag_answer,
            web_result,
            image_results,
        )

    # =========================
    # 异步流式问答
    # =========================

    async def chat_stream(self, user_message: str, search_mode: str = "auto"):
        image_paths = self._extract_image_paths(user_message)
        clean_query = self._strip_system_image_hint(user_message)

        if not clean_query and image_paths:
            clean_query = "请分析这些脐橙图片中的病虫害或异常症状。"

        mode = (search_mode or "auto").lower()
        if mode not in {"auto", "web", "local"}:
            mode = "auto"

        # 1. 图片场景：统一走 chat() 整合
        if image_paths:
            answer = self.chat(user_message, search_mode=mode)
            chunk_size = 20
            for index in range(0, len(answer), chunk_size):
                yield answer[index:index + chunk_size]
                await asyncio.sleep(0.04)
            return

        # 2. 纯文本场景，先判断模式
        if clean_query:
            use_web = self._should_use_web(clean_query, search_mode=mode)

            # web 模式 或 auto 下命中联网条件：直接走联网整合
            if use_web:
                answer = self.chat(user_message, search_mode=mode)
                chunk_size = 20
                for index in range(0, len(answer), chunk_size):
                    yield answer[index:index + chunk_size]
                    await asyncio.sleep(0.04)
                return

            # local 模式 / auto 但未命中联网：走知识库流式
            rag_result = self.rag_service.ask_question(clean_query)

            routed_tool = rag_result.get("routed_tool")
            source_docs = rag_result.get("source_documents", [])
            rag_answer = rag_result.get("answer", "")

            if routed_tool == "knowledge_qa" and source_docs:
                try:
                    for chunk in self._stream_llm_answer(clean_query, source_docs):
                        yield chunk
                        await asyncio.sleep(0)
                    return
                except Exception:
                    pass

            final_text = rag_answer or "知识库中没有找到相关信息。"
            self._update_rag_history(clean_query, final_text)
            yield final_text
            return

        # 3. 兜底
        answer = self.chat(user_message, search_mode=mode)
        chunk_size = 20
        for index in range(0, len(answer), chunk_size):
            yield answer[index:index + chunk_size]
            await asyncio.sleep(0.04)