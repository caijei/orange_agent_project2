"""Knowledge base loader for navel orange Q&A system."""

import os
from pathlib import Path
from typing import List

from langchain.schema import Document


def load_knowledge_base(data_dir: str) -> List[Document]:
    """Load all text files from the knowledge base directory."""
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {data_dir}")

    topic_map = {
        "varieties.txt": "品种",
        "cultivation.txt": "栽培技术",
        "diseases.txt": "病虫害防治",
        "nutrition.txt": "营养价值",
        "market.txt": "市场与产业",
        "climate.txt": "生长环境与气候",
    }

    for filename, topic in topic_map.items():
        file_path = data_path / filename
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            chunks = _split_by_section(content, filename, topic)
            documents.extend(chunks)

    return documents


def _split_by_section(content: str, source: str, topic: str) -> List[Document]:
    """Split document content into sections by headings."""
    lines = content.split("\n")
    sections: List[Document] = []
    current_section: List[str] = []
    current_heading = ""

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            if current_section:
                text = "\n".join(current_section).strip()
                if text:
                    sections.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": source,
                                "topic": topic,
                                "heading": current_heading,
                            },
                        )
                    )
            current_heading = stripped.lstrip("#").strip()
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        text = "\n".join(current_section).strip()
        if text:
            sections.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "topic": topic,
                        "heading": current_heading,
                    },
                )
            )

    # Fallback: if no sections found, use the whole file
    if not sections and content.strip():
        sections.append(
            Document(
                page_content=content,
                metadata={"source": source, "topic": topic, "heading": ""},
            )
        )

    return sections
