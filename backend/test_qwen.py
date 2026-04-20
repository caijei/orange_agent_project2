class RAGTool(Tool):
    """RAG工具

    提供完整的 RAG 能力：
    - 添加多格式文档（PDF、Office、图片、音频等）
    - 智能检索与召回
    - LLM 增强问答
    - 知识库管理
    """

    def __init__(
            self,
            knowledge_base_path: str = "./knowledge_base",
            qdrant_url: str = None,
            qdrant_api_key: str = None,
            collection_name: str = "rag_knowledge_base",
            rag_namespace: str = "default"
    ):
        # 初始化RAG管道
        self._pipelines: Dict[str, Dict[str, Any]] = {}
        self.llm = HelloAgentsLLM()

        # 创建默认管道
        default_pipeline = create_rag_pipeline(
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
            rag_namespace=self.rag_namespace
        )
        self._pipelines[self.rag_namespace] = default_pipeline


    def _convert_to_markdown(path: str) -> str:
    """
    Universal document reader using MarkItDown with enhanced PDF processing.
    核心功能：将任意格式文档转换为Markdown文本

    支持格式：
    - 文档：PDF、Word、Excel、PowerPoint
    - 图像：JPG、PNG、GIF（通过OCR）
    - 音频：MP3、WAV、M4A（通过转录）
    - 文本：TXT、CSV、JSON、XML、HTML
    - 代码：Python、JavaScript、Java等
    """
    if not os.path.exists(path):
        return ""

    # 对PDF文件使用增强处理
    ext = (os.path.splitext(path)[1] or '').lower()
    if ext == '.pdf':
        return _enhanced_pdf_processing(path)

    # 其他格式使用MarkItDown统一转换
    md_instance = _get_markitdown_instance()
    if md_instance is None:
        return _fallback_text_reader(path)

    try:
        result = md_instance.convert(path)
        markdown_text = getattr(result, "text_content", None)
        if isinstance(markdown_text, str) and markdown_text.strip():
            print(f"[RAG] MarkItDown转换成功: {path} -> {len(markdown_text)} chars Markdown")
            return markdown_text
        return ""
    except Exception as e:
        print(f"[WARNING] MarkItDown转换失败 {path}: {e}")
        return _fallback_text_reader(path)


    def _split_paragraphs_with_headings(text: str) -> List[Dict]:
    """根据标题层次分割段落，保持语义完整性"""
    lines = text.splitlines()
    heading_stack: List[str] = []
    paragraphs: List[Dict] = []
    buf: List[str] = []
    char_pos = 0

    def flush_buf(end_pos: int):
        if not buf:
            return
        content = "\n".join(buf).strip()
        if not content:
            return
        paragraphs.append({
            "content": content,
            "heading_path": " > ".join(heading_stack) if heading_stack else None,
            "start": max(0, end_pos - len(content)),
            "end": end_pos,
        })

    for ln in lines:
        raw = ln
        if raw.strip().startswith("#"):
            # 处理标题行
            flush_buf(char_pos)
            level = len(raw) - len(raw.lstrip('#'))
            title = raw.lstrip('#').strip()

            if level <= 0:
                level = 1
            if level <= len(heading_stack):
                heading_stack = heading_stack[:level - 1]
            heading_stack.append(title)

            char_pos += len(raw) + 1
            continue

        # 段落内容累积
        if raw.strip() == "":
            flush_buf(char_pos)
            buf = []
        else:
            buf.append(raw)
        char_pos += len(raw) + 1

    flush_buf(char_pos)

    if not paragraphs:
        paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]

    return paragraphs


    def _chunk_paragraphs(paragraphs: List[Dict], chunk_tokens: int, overlap_tokens: int) -> List[Dict]:
    """基于Token数量的智能分块"""
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0

    while i < len(paragraphs):
        p = paragraphs[i]
        p_tokens = _approx_token_len(p["content"]) or 1

        if cur_tokens + p_tokens <= chunk_tokens or not cur:
            cur.append(p)
            cur_tokens += p_tokens
            i += 1
        else:
            # 生成当前分块
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end = cur[-1]["end"]
            heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)

            chunks.append({
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
            })

            # 构建重叠部分
            if overlap_tokens > 0 and cur:
                kept: List[Dict] = []
                kept_tokens = 0
                for x in reversed(cur):
                    t = _approx_token_len(x["content"]) or 1
                    if kept_tokens + t > overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept))
                cur_tokens = kept_tokens
            else:
                cur = []
                cur_tokens = 0

    # 处理最后一个分块
    if cur:
        content = "\n\n".join(x["content"] for x in cur)
        start = cur[0]["start"]
        end = cur[-1]["end"]
        heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)

        chunks.append({
            "content": content,
            "start": start,
            "end": end,
            "heading_path": heading_path,
        })

    return chunks
    def _approx_token_len(text: str) -> int:
    """近似估计Token长度，支持中英文混合"""
    # CJK字符按1 token计算
    cjk = sum(1 for ch in text if _is_cjk(ch))
    # 其他字符按空白分词计算
    non_cjk_tokens = len([t for t in text.split() if t])
    return cjk + non_cjk_tokens

    def _is_cjk(ch: str) -> bool:
    """判断是否为CJK字符"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or  # CJK统一汉字
        0x3400 <= code <= 0x4DBF or  # CJK扩展A
        0x20000 <= code <= 0x2A6DF or # CJK扩展B
        0x2A700 <= code <= 0x2B73F or # CJK扩展C
        0x2B740 <= code <= 0x2B81F or # CJK扩展D
        0x2B820 <= code <= 0x2CEAF or # CJK扩展E
        0xF900 <= code <= 0xFAFF      # CJK兼容汉字
    )


    def index_chunks(
            store=None,
            chunks: List[Dict] = None,
            cache_db: Optional[str] = None,
            batch_size: int = 64,
            rag_namespace: str = "default"
    ) -> None:
        """
        Index markdown chunks with unified embedding and Qdrant storage.
        Uses百炼 API with fallback to sentence-transformers.
        """
        if not chunks:
            print("[RAG] No chunks to index")
            return

        # 使用统一嵌入模型
        embedder = get_text_embedder()
        dimension = get_dimension(384)

        # 创建默认Qdrant存储
        if store is None:
            store = _create_default_vector_store(dimension)
            print(f"[RAG] Created default Qdrant store with dimension {dimension}")

        # 预处理Markdown文本以获得更好的嵌入质量
        processed_texts = []
        for c in chunks:
            raw_content = c["content"]
            processed_content = _preprocess_markdown_for_embedding(raw_content)
            processed_texts.append(processed_content)

        print(f"[RAG] Embedding start: total_texts={len(processed_texts)} batch_size={batch_size}")

        # 批量编码
        vecs: List[List[float]] = []
        for i in range(0, len(processed_texts), batch_size):
            part = processed_texts[i:i + batch_size]
            try:
                # 使用统一嵌入器（内部处理缓存）
                part_vecs = embedder.encode(part)

                # 标准化为List[List[float]]格式
                if not isinstance(part_vecs, list):
                    if hasattr(part_vecs, "tolist"):
                        part_vecs = [part_vecs.tolist()]
                    else:
                        part_vecs = [list(part_vecs)]

                # 处理向量格式和维度
                for v in part_vecs:
                    try:
                        if hasattr(v, "tolist"):
                            v = v.tolist()
                        v_norm = [float(x) for x in v]

                        # 维度检查和调整
                        if len(v_norm) != dimension:
                            print(f"[WARNING] 向量维度异常: 期望{dimension}, 实际{len(v_norm)}")
                            if len(v_norm) < dimension:
                                v_norm.extend([0.0] * (dimension - len(v_norm)))
                            else:
                                v_norm = v_norm[:dimension]

                        vecs.append(v_norm)
                    except Exception as e:
                        print(f"[WARNING] 向量转换失败: {e}, 使用零向量")
                        vecs.append([0.0] * dimension)

            except Exception as e:
                print(f"[WARNING] Batch {i} encoding failed: {e}")
                # 实现重试机制
                # ... 重试逻辑 ...

            print(f"[RAG] Embedding progress: {min(i + batch_size, len(processed_texts))}/{len(processed_texts)}")

    def _prompt_mqe(query: str, n: int) -> List[str]:
        """使用LLM生成多样化的查询扩展"""
    try:
        from ...core.llm import HelloAgentsLLM
        llm = HelloAgentsLLM()
        prompt = [
            {"role": "system",
             "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"},
            {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"}
        ]
        text = llm.invoke(prompt)
        lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
        outs = [ln for ln in lines if ln]
        return outs[:n] or [query]
    except Exception:
        return [query]

    def _prompt_hyde(query: str) -> Optional[str]:
        """生成假设性文档用于改善检索"""

    try:
        from ...core.llm import HelloAgentsLLM
        llm = HelloAgentsLLM()
        prompt = [
            {"role": "system",
             "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
            {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"}
        ]
        return llm.invoke(prompt)
    except Exception:
        return None

    def search_vectors_expanded(
            store=None,
            query: str = "",
            top_k: int = 8,
            rag_namespace: Optional[str] = None,
            only_rag_data: bool = True,
            score_threshold: Optional[float] = None,
            enable_mqe: bool = False,
            mqe_expansions: int = 2,
            enable_hyde: bool = False,
            candidate_pool_multiplier: int = 4,
    ) -> List[Dict]:
        """
        Search with query expansion using unified embedding and Qdrant.
        """
        if not query:
            return []

        # 创建默认存储
        if store is None:
            store = _create_default_vector_store()

        # 查询扩展
        expansions: List[str] = [query]

        if enable_mqe and mqe_expansions > 0:
            expansions.extend(_prompt_mqe(query, mqe_expansions))
        if enable_hyde:
            hyde_text = _prompt_hyde(query)
            if hyde_text:
                expansions.append(hyde_text)

        # 去重和修剪
        uniq: List[str] = []
        for e in expansions:
            if e and e not in uniq:
                uniq.append(e)
        expansions = uniq[: max(1, len(uniq))]

        # 分配候选池
        pool = max(top_k * candidate_pool_multiplier, 20)
        per = max(1, pool // max(1, len(expansions)))

        # 构建RAG数据过滤器
        where = {"memory_type": "rag_chunk"}
        if only_rag_data:
            where["is_rag_data"] = True
            where["data_source"] = "rag_pipeline"
        if rag_namespace:
            where["rag_namespace"] = rag_namespace

        # 收集所有扩展查询的结果
        agg: Dict[str, Dict] = {}
        for q in expansions:
            qv = embed_query(q)
            hits = store.search_similar(
                query_vector=qv,
                limit=per,
                score_threshold=score_threshold,
                where=where
            )
            for h in hits:
                mid = h.get("metadata", {}).get("memory_id", h.get("id"))
                s = float(h.get("score", 0.0))
                if mid not in agg or s > float(agg[mid].get("score", 0.0)):
                    agg[mid] = h

        # 按分数排序返回
        merged = list(agg.values())
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return merged[:top_k]

