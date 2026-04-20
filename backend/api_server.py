"""
api_server v2 — 配合 agent_service_v2 使用
主要变化：
  - session_id 传入 agent，实现多轮对话记忆
  - 新增 /api/clear_memory 接口（清除某 session 的历史）
  - 全局只创建一个 OrangeAgent 实例（内部按 session 隔离记忆）
"""

import base64
import json
import os
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal, Optional

from agent_service import OrangeAgent  # ← 改成 v2

app = FastAPI(title="脐橙多模态智能体 API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_IMAGE_DIR = "temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# 全局单例 Agent（内部按 session_id 管理记忆）
agent = OrangeAgent()


class ChatRequest(BaseModel):
    query: str
    session_id: str
    images_base64: Optional[list[str]] = None
    search_mode: Literal["auto", "web", "local"] = "auto"


class ClearMemoryRequest(BaseModel):
    session_id: str


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_input = request.query
    session_id = request.session_id
    images_list = request.images_base64 or []
    search_mode = request.search_mode

    if not user_input.strip() and not images_list:
        raise HTTPException(status_code=400, detail="内容不能为空")

    # 保存图片
    saved_paths = []
    for i, b64_str in enumerate(images_list):
        header, encoded = b64_str.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        path = f"{TEMP_IMAGE_DIR}/up_{session_id}_{i}_{uuid.uuid4().hex[:4]}.jpg"
        with open(path, "wb") as f:
            f.write(img_bytes)
        saved_paths.append(path)

    final_query = user_input
    if saved_paths:
        paths_str = ", ".join([f"'{p}'" for p in saved_paths])
        final_query += f"。请综合分析这 {len(saved_paths)} 张图片，路径：{paths_str}。"

    async def event_generator():
        try:
            async for chunk in agent.chat_stream(
                final_query,
                session_id=session_id,
                search_mode=search_mode,
            ):
                payload = json.dumps({"text": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/clear_memory")
async def clear_memory(request: ClearMemoryRequest):
    """清除某个 session 的对话历史，让用户开启新话题"""
    agent.clear_memory(request.session_id)
    return {"status": "ok", "message": f"已清除 session {request.session_id} 的对话记忆"}


@app.get("/api/health")
async def health():
    return {"status": "running"}


if __name__ == "__main__":
    print("FastAPI v2 后端已启动：http://localhost:8888")
    uvicorn.run(app, host="127.0.0.1", port=8888)