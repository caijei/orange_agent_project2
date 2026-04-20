"""Vector store management for the navel orange knowledge base."""

import os
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import config


def build_vector_store(documents: List[Document]) -> FAISS:
    """Build a FAISS vector store from the given documents."""
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def save_vector_store(vector_store: FAISS, path: str) -> None:
    """Save the vector store to disk."""
    Path(path).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(path)


def load_vector_store(path: str) -> Optional[FAISS]:
    """Load the vector store from disk. Returns None if not found."""
    index_path = Path(path) / "index.faiss"
    if not index_path.exists():
        return None
    embeddings = _get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def get_or_build_vector_store(
    documents: List[Document], store_path: str
) -> FAISS:
    """Load existing vector store or build a new one from documents."""
    vector_store = load_vector_store(store_path)
    if vector_store is None:
        vector_store = build_vector_store(documents)
        save_vector_store(vector_store, store_path)
    return vector_store


def _get_embeddings() -> OpenAIEmbeddings:
    """Create an OpenAI embeddings instance."""
    return OpenAIEmbeddings(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        model=config.EMBEDDING_MODEL,
    )
