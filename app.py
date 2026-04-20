"""Main FastAPI application for the navel orange knowledge Q&A system."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from agents.orange_agent import create_orange_agent
from agents.tools import set_vector_store
from config import config
from knowledge.loader import load_knowledge_base
from knowledge.vector_store import get_or_build_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global agent instance (one per application, holds conversation memory)
_agent_executor = None
# Per-session agents keyed by session_id
_session_agents: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise knowledge base and vector store on startup."""
    logger.info("Loading knowledge base...")
    try:
        documents = load_knowledge_base(config.KNOWLEDGE_BASE_PATH)
        logger.info(f"Loaded {len(documents)} document chunks.")

        logger.info("Building / loading vector store...")
        vector_store = get_or_build_vector_store(documents, config.VECTOR_STORE_PATH)
        set_vector_store(vector_store)
        logger.info("Vector store ready.")
    except Exception as exc:
        logger.error(f"Failed to initialise knowledge base: {exc}")
        raise

    yield

    _session_agents.clear()
    logger.info("Application shutdown complete.")


app = FastAPI(
    title="脐橙知识问答系统",
    description="基于Agent的脐橙知识问答系统 —— 本科毕业设计",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------- Request / Response models ----------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户提问")
    session_id: str = Field(default="default", description="会话ID，用于保持对话记忆")


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
    message: str


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main Q&A web interface."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", message="脐橙知识问答系统运行正常")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Handle a Q&A request and return the agent's answer."""
    session_id = req.session_id or "default"

    if session_id not in _session_agents:
        _session_agents[session_id] = create_orange_agent()

    agent_executor = _session_agents[session_id]

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent_executor.invoke({"input": req.question}),
        )
        answer = result.get("output", "抱歉，无法生成回答，请稍后重试。")
    except Exception as exc:
        logger.error(f"Agent error for session {session_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"问答服务异常：{str(exc)}")

    return ChatResponse(answer=answer, session_id=session_id)


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    if session_id in _session_agents:
        del _session_agents[session_id]
    return {"message": f"会话 {session_id} 已清除"}


@app.get("/api/topics")
async def list_topics():
    """Return the list of knowledge topics available."""
    topics = [
        {"id": "varieties", "name": "品种介绍", "icon": "🍊"},
        {"id": "cultivation", "name": "栽培技术", "icon": "🌱"},
        {"id": "diseases", "name": "病虫害防治", "icon": "🐛"},
        {"id": "nutrition", "name": "营养价值", "icon": "💊"},
        {"id": "market", "name": "市场与产业", "icon": "📈"},
        {"id": "climate", "name": "生长环境", "icon": "🌤️"},
    ]
    return {"topics": topics}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level="info",
    )
