"""Tests for the navel orange knowledge Q&A system."""

import pytest
from unittest.mock import MagicMock

from knowledge.loader import load_knowledge_base, _split_by_section
from agents.tools import calculate_yield_estimate, set_vector_store


# ===== Knowledge loader tests =====

def test_load_knowledge_base_returns_documents():
    docs = load_knowledge_base("./knowledge/data")
    assert len(docs) > 0


def test_load_knowledge_base_has_metadata():
    docs = load_knowledge_base("./knowledge/data")
    for doc in docs:
        assert "source" in doc.metadata
        assert "topic" in doc.metadata
        assert doc.page_content  # non-empty


def test_load_knowledge_base_covers_all_topics():
    docs = load_knowledge_base("./knowledge/data")
    topics = {doc.metadata["topic"] for doc in docs}
    expected = {"品种", "栽培技术", "病虫害防治", "营养价值", "市场与产业", "生长环境与气候"}
    assert expected.issubset(topics)


def test_split_by_section_basic():
    content = "# Title\n## Section 1\nContent 1\n## Section 2\nContent 2\n"
    docs = _split_by_section(content, "test.txt", "测试")
    assert len(docs) >= 2
    headings = [d.metadata["heading"] for d in docs]
    assert "Section 1" in headings
    assert "Section 2" in headings


def test_split_by_section_empty():
    docs = _split_by_section("", "empty.txt", "空")
    assert docs == []


def test_load_knowledge_base_missing_dir():
    with pytest.raises(FileNotFoundError):
        load_knowledge_base("./nonexistent_dir")


# ===== Tool tests =====

def test_calculate_yield_estimate_basic():
    result = calculate_yield_estimate.invoke(
        {"area_mu": 10.0, "yield_per_mu": 4000.0, "price_per_kg": 4.0}
    )
    assert "40000" in result  # total yield
    assert "160,000" in result  # gross revenue
    assert "110,000" in result  # net profit


def test_calculate_yield_estimate_zero():
    result = calculate_yield_estimate.invoke(
        {"area_mu": 0.0, "yield_per_mu": 0.0, "price_per_kg": 0.0}
    )
    assert "0" in result


def test_search_knowledge_base_no_store():
    """Search tool returns graceful message when vector store is not loaded."""
    from agents.tools import search_knowledge_base
    set_vector_store(None)
    result = search_knowledge_base.invoke({"query": "纽荷尔"})
    assert "知识库" in result


# ===== API endpoint tests =====

@pytest.fixture
def client():
    """Create a TestClient with a mock vector store, bypassing the OpenAI call."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [
        MagicMock(
            page_content="纽荷尔脐橙是优良品种。",
            metadata={"topic": "品种", "heading": "纽荷尔脐橙"},
        )
    ]

    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from app import app

    set_vector_store(mock_vs)
    # Patch in the app module's namespace so lifespan doesn't call OpenAI
    with patch("app.get_or_build_vector_store", return_value=mock_vs):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_topics_endpoint(client):
    resp = client.get("/api/topics")
    assert resp.status_code == 200
    data = resp.json()
    assert "topics" in data
    assert len(data["topics"]) == 6


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "脐橙" in resp.text


def test_clear_session_endpoint(client):
    resp = client.delete("/api/session/test_session_123")
    assert resp.status_code == 200
