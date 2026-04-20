import json
import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LangchainDocument

load_dotenv(Path(__file__).with_name('.env'))


class DocumentProcessor:
    def __init__(self, collection_name: str = 'orange_knowledge'):
        self.persist_directory = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        self.collection_name = collection_name

        self.chunk_output_dir = os.getenv('CHUNK_OUTPUT_DIR', './chunks')
        self.chunk_size = int(os.getenv('RAG_CHUNK_SIZE', '600'))
        self.chunk_overlap = int(os.getenv('RAG_CHUNK_OVERLAP', '100'))

        self.embeddings = DashScopeEmbeddings(dashscope_api_key=self.api_key)

    def _ensure_api_key(self):
        if not self.api_key:
            raise ValueError('缺少 DASHSCOPE_API_KEY，无法创建 embedding 并写入向量库。')

    def _reset_vectorstore(self):
        persist_path = Path(self.persist_directory)
        if persist_path.exists():
            shutil.rmtree(persist_path)
        persist_path.mkdir(parents=True, exist_ok=True)

    def _persist_documents(self, documents: list[LangchainDocument]):
        if not documents:
            raise ValueError('没有可写入向量库的文档。')

        self._ensure_api_key()
        self._reset_vectorstore()

        Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )

        print(f'已写入 {len(documents)} 个文本块到向量库: {self.persist_directory}')

    def _clean_markdown(self, text: str) -> str:
        """
        基础清洗：
        1. 去掉首尾空白
        2. 删除纯分隔线
        3. 压缩连续空行为单个空行
        """
        cleaned_lines = []
        prev_blank = False

        for raw_line in text.splitlines():
            line = raw_line.rstrip()

            stripped = line.strip()

            if stripped in {'---', '***', '___'}:
                continue

            if not stripped:
                if not prev_blank:
                    cleaned_lines.append('')
                prev_blank = True
                continue

            cleaned_lines.append(line)
            prev_blank = False

        return '\n'.join(cleaned_lines).strip()

    def _parse_markdown_sections(self, text: str) -> list[dict]:
        """
        按 Markdown 标题解析 section
        支持:
        # 一级标题
        ## 二级标题
        ### 三级标题
        """
        cleaned = self._clean_markdown(text)
        if not cleaned:
            return []

        lines = cleaned.split('\n')

        sections = []
        current_h1 = ''
        current_h2 = ''
        current_h3 = ''
        buffer = []

        def flush_section():
            nonlocal buffer, sections, current_h1, current_h2, current_h3
            content = '\n'.join(buffer).strip()
            if content:
                sections.append({
                    'h1': current_h1,
                    'h2': current_h2,
                    'h3': current_h3,
                    'content': content
                })
            buffer = []

        h1_pattern = re.compile(r'^#\s+(.*)$')
        h2_pattern = re.compile(r'^##\s+(.*)$')
        h3_pattern = re.compile(r'^###\s+(.*)$')

        for line in lines:
            stripped = line.strip()

            m1 = h1_pattern.match(stripped)
            if m1:
                flush_section()
                current_h1 = m1.group(1).strip()
                current_h2 = ''
                current_h3 = ''
                continue

            m2 = h2_pattern.match(stripped)
            if m2:
                flush_section()
                current_h2 = m2.group(1).strip()
                current_h3 = ''
                continue

            m3 = h3_pattern.match(stripped)
            if m3:
                flush_section()
                current_h3 = m3.group(1).strip()
                continue

            buffer.append(line)

        flush_section()

        return sections

    def _split_long_text(self, text: str) -> list[str]:
        """
        对单个 section 的正文做二次切块：
        1. 优先按空行分段
        2. 再按长度拼接
        3. 如果单段过长，再硬切并保留 overlap
        """
        text = text.strip()
        if not text:
            return []

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not paragraphs:
            return []

        chunks = []
        buffer = ''

        for para in paragraphs:
            candidate = f'{buffer}\n\n{para}'.strip() if buffer else para

            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer.strip())

            if len(para) <= self.chunk_size:
                buffer = para
                continue

            long_para_chunks = self._force_split_text(para)
            chunks.extend(long_para_chunks)
            buffer = ''

        if buffer.strip():
            chunks.append(buffer.strip())

        return chunks

    def _force_split_text(self, text: str) -> list[str]:
        """
        对超长段落强制切分，优先按句子聚合，实在不行再按字符硬切
        """
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        sentence_parts = re.split(r'(?<=[。！？；.!?;])', text)
        sentence_parts = [s.strip() for s in sentence_parts if s.strip()]

        if len(sentence_parts) <= 1:
            return self._hard_split_with_overlap(text)

        chunks = []
        buffer = ''

        for sentence in sentence_parts:
            candidate = f'{buffer}{sentence}' if buffer else sentence
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer.strip())

            if len(sentence) <= self.chunk_size:
                buffer = sentence
            else:
                chunks.extend(self._hard_split_with_overlap(sentence))
                buffer = ''

        if buffer.strip():
            chunks.append(buffer.strip())

        return chunks

    def _hard_split_with_overlap(self, text: str) -> list[str]:
        """
        最终兜底：按字符硬切，并保留 overlap
        """
        text = text.strip()
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break

            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks

    def generate_chunks_from_markdown_directory(self, markdown_dir: str):
        """
        第一步：
        读取 markdown 文件
        -> 按标题切 section
        -> section 过长再切块
        -> 导出为 json，供手动修改
        """
        directory = Path(markdown_dir)
        if not directory.exists():
            raise FileNotFoundError(f'找不到 markdown 目录: {directory}')

        markdown_files = sorted(directory.glob('*.md'))
        if not markdown_files:
            raise ValueError(f'{directory} 中没有可用的 Markdown 文件。')

        output_dir = Path(self.chunk_output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_sections = 0
        total_chunks = 0

        for markdown_file in markdown_files:
            text = markdown_file.read_text(encoding='utf-8')
            sections = self._parse_markdown_sections(text)
            total_sections += len(sections)

            chunk_records = []
            chunk_counter = 1

            for section_index, section in enumerate(sections, start=1):
                section_chunks = self._split_long_text(section['content'])

                for inner_index, chunk_text in enumerate(section_chunks, start=1):
                    chunk_records.append({
                        'chunk_id': f'{markdown_file.stem}_{chunk_counter}',
                        'source_file': markdown_file.name,
                        'section_index': section_index,
                        'chunk_index': chunk_counter,
                        'section_chunk_index': inner_index,
                        'h1': section['h1'],
                        'h2': section['h2'],
                        'h3': section['h3'],
                        'content': chunk_text
                    })
                    chunk_counter += 1

            output_file = output_dir / f'{markdown_file.stem}.json'
            output_file.write_text(
                json.dumps(chunk_records, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )

            print(
                f'{markdown_file.name} -> '
                f'{len(sections)} 个 section，'
                f'{len(chunk_records)} 个 chunk，'
                f'已保存到 {output_file}'
            )

            total_chunks += len(chunk_records)

        print('\n处理完成')
        print(f'共处理 Markdown 文件数: {len(markdown_files)}')
        print(f'共解析 section 数: {total_sections}')
        print(f'共生成 chunk 数: {total_chunks}')
        print(f'请手动修改目录中的 json 文件: {output_dir}')

    def build_vectorstore_from_chunk_directory(self, chunk_dir: str):
        """
        第二步：
        读取人工修改后的 chunk json
        -> 转成 LangChain Document
        -> 写入 Chroma
        """
        directory = Path(chunk_dir)
        if not directory.exists():
            raise FileNotFoundError(f'找不到 chunk 目录: {directory}')

        json_files = sorted(directory.glob('*.json'))
        if not json_files:
            raise ValueError(f'{directory} 中没有可用的 JSON chunk 文件。')

        documents: list[LangchainDocument] = []

        for json_file in json_files:
            try:
                records = json.loads(json_file.read_text(encoding='utf-8'))
            except json.JSONDecodeError as e:
                print(f'跳过 JSON 格式错误文件: {json_file}，原因: {e}')
                continue

            if not isinstance(records, list):
                print(f'跳过格式异常文件: {json_file}')
                continue

            for record in records:
                content = str(record.get('content', '')).strip()
                if not content:
                    continue

                metadata = {
                    'chunk_id': record.get('chunk_id', ''),
                    'source_file': record.get('source_file', json_file.name),
                    'section_index': record.get('section_index', 0),
                    'chunk_index': record.get('chunk_index', 0),
                    'section_chunk_index': record.get('section_chunk_index', 0),
                    'h1': record.get('h1', ''),
                    'h2': record.get('h2', ''),
                    'h3': record.get('h3', '')
                }

                documents.append(
                    LangchainDocument(
                        page_content=content,
                        metadata=metadata
                    )
                )

        print(f'从 {directory} 读取到 {len(json_files)} 个 chunk 文件，共 {len(documents)} 个文本块')
        self._persist_documents(documents)


if __name__ == '__main__':
    processor = DocumentProcessor()

    base_dir = os.path.dirname(__file__)

    markdown_dir = os.path.abspath(
        os.path.join(base_dir, '..', 'docs_clean', 'markdown')
    )

    chunk_dir = os.path.abspath(
        os.path.join(base_dir, '.', 'chunks')
    )

    # 第一步：生成 chunk 文件，手动修改
    processor.generate_chunks_from_markdown_directory(markdown_dir)

    # 第二步：手动修改 chunks 目录中的 json 后，再取消下面这行注释执行入库
    #processor.build_vectorstore_from_chunk_directory(chunk_dir)