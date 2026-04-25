"""
api_server v4 — 保留用户名密码，但不做密码哈希，不做 JWT
说明：
  POST /api/register       注册（用户名 + 密码，密码明文存库）
  POST /api/login          登录（用户名 + 密码，明文比对）
  GET  /api/sessions       获取当前用户所有会话
  POST /api/sessions       新建会话
  PUT  /api/sessions/{id}  重命名会话
  DELETE /api/sessions/{id} 删除会话
  GET  /api/sessions/{id}/messages  获取会话历史消息

身份识别方式：
  后续请求统一带请求头：X-Username: 用户名
  不再使用 JWT / Authorization: Bearer
"""

import base64
import json
import os
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal, Optional

from agent_service import OrangeAgent
from database import (
    init_db,
    create_user,
    get_user_by_username,
    create_session,
    get_sessions_by_user,
    update_session_title,
    delete_session,
    session_belongs_to_user,
    save_message,
    get_messages_by_session,
)

# ── 初始化 ─────────────────────────────────────────────
init_db()

app = FastAPI(title="脐橙多模态智能体 API v4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_IMAGE_DIR = "temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

agent = OrangeAgent()


# ── 请求/响应模型 ───────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    query: str
    session_id: str
    images_base64: Optional[list[str]] = None
    search_mode: Literal["auto", "web", "local"] = "auto"


class CreateSessionRequest(BaseModel):
    title: str = "新对话"


class RenameSessionRequest(BaseModel):
    title: str


class ClearMemoryRequest(BaseModel):
    session_id: str


# ── 当前用户：从请求头 X-Username 取 ────────────────────

def get_current_user_by_name(
    x_username: Optional[str] = Header(default=None, alias="X-Username")
) -> dict:
    if not x_username or not x_username.strip():
        raise HTTPException(status_code=401, detail="缺少用户名，请先登录")

    username = x_username.strip()
    user = get_user_by_username(username)
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在，请先注册")

    return user


# ── 注册 / 登录 ────────────────────────────────────────

@app.post("/api/register")
async def register(req: RegisterRequest):
    username = req.username.strip()
    password = req.password

    if len(username) < 2:
        raise HTTPException(status_code=400, detail="用户名至少2个字符")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少6位")

    # 这里不做哈希，直接明文存储
    user_id = create_user(username, password)
    if user_id is None:
        raise HTTPException(status_code=409, detail="用户名已存在")

    return {"username": username}


@app.post("/api/login")
async def login(req: LoginRequest):
    username = req.username.strip()
    password = req.password

    if len(username) < 2:
        raise HTTPException(status_code=400, detail="用户名至少2个字符")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少6位")

    user = get_user_by_username(username)
    if user is None or user["password_hash"] != password:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    return {"username": user["username"]}


# ── 会话管理 ───────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions(current_user: dict = Depends(get_current_user_by_name)):
    sessions = get_sessions_by_user(current_user["id"])
    return {"sessions": sessions}


@app.post("/api/sessions")
async def new_session(
    req: CreateSessionRequest,
    current_user: dict = Depends(get_current_user_by_name),
):
    session_id = f"{current_user['id']}_{uuid.uuid4().hex[:12]}"
    title = req.title.strip() if req.title and req.title.strip() else "新对话"
    session = create_session(session_id, current_user["id"], title)
    return session


@app.put("/api/sessions/{session_id}")
async def rename_session(
    session_id: str,
    req: RenameSessionRequest,
    current_user: dict = Depends(get_current_user_by_name),
):
    if not session_belongs_to_user(session_id, current_user["id"]):
        raise HTTPException(status_code=403, detail="无权操作此会话")

    update_session_title(session_id, req.title.strip() or "新对话")
    return {"status": "ok"}


@app.delete("/api/sessions/{session_id}")
async def remove_session(
    session_id: str,
    current_user: dict = Depends(get_current_user_by_name),
):
    if not session_belongs_to_user(session_id, current_user["id"]):
        raise HTTPException(status_code=403, detail="无权操作此会话")

    delete_session(session_id)
    agent.clear_memory(session_id)
    return {"status": "ok"}


@app.get("/api/sessions/{session_id}/messages")
async def get_history(
    session_id: str,
    current_user: dict = Depends(get_current_user_by_name),
):
    if not session_belongs_to_user(session_id, current_user["id"]):
        raise HTTPException(status_code=403, detail="无权访问此会话")

    messages = get_messages_by_session(session_id)
    return {"messages": messages}


# ── 聊天（无 JWT，按用户名识别用户）─────────────────────

@app.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user_by_name),
):
    user_input = request.query or ""
    session_id = request.session_id
    images_list = request.images_base64 or []
    search_mode = request.search_mode

    if not user_input.strip() and not images_list:
        raise HTTPException(status_code=400, detail="内容不能为空")

    if not session_belongs_to_user(session_id, current_user["id"]):
        raise HTTPException(status_code=403, detail="无权访问此会话")

    saved_paths = []
    for i, b64_str in enumerate(images_list):
        try:
            _, encoded = b64_str.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            path = f"{TEMP_IMAGE_DIR}/up_{session_id}_{i}_{uuid.uuid4().hex[:4]}.jpg"
            with open(path, "wb") as f:
                f.write(img_bytes)
            saved_paths.append(path)
        except Exception:
            continue

    final_query = user_input
    if saved_paths:
        paths_str = ", ".join([f"'{p}'" for p in saved_paths])
        if final_query.strip():
            final_query += f"。请综合分析这 {len(saved_paths)} 张图片，路径：{paths_str}。"
        else:
            final_query = f"请综合分析这 {len(saved_paths)} 张图片，路径：{paths_str}。"

    save_message(
        session_id=session_id,
        role="user",
        content=user_input,
        images=images_list if images_list else None,
    )

    async def event_generator():
        accumulated = ""
        ask_user_content = ""  # 单独收集追问内容，用于写库
        try:
            async for chunk in agent.chat_stream(
                final_query,
                session_id=session_id,
                search_mode=search_mode,
            ):
                if chunk.startswith("__ASK_USER__:"):
                    # 提取追问文本，等流结束后写库
                    ask_user_content = chunk.replace("__ASK_USER__:", "").strip()
                elif not chunk.startswith("__STATUS__:"):
                    accumulated += chunk

                payload = json.dumps({"text": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

            yield "data: [DONE]\n\n"

            # 正常回答写库
            if accumulated.strip():
                save_message(
                    session_id=session_id,
                    role="assistant",
                    content=accumulated,
                )

            # 追问写库，刷新后才能从历史恢复
            if ask_user_content:
                save_message(
                    session_id=session_id,
                    role="assistant",
                    content=ask_user_content,
                )

        except Exception as e:
            error_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── 其他接口 ───────────────────────────────────────────

@app.post("/api/clear_memory")
async def clear_memory(
    request: ClearMemoryRequest,
    current_user: dict = Depends(get_current_user_by_name),
):
    if not session_belongs_to_user(request.session_id, current_user["id"]):
        raise HTTPException(status_code=403, detail="无权操作此会话")

    agent.clear_memory(request.session_id)
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    return {"status": "running"}


@app.get("/api/me")
async def me(current_user: dict = Depends(get_current_user_by_name)):
    return {"id": current_user["id"], "username": current_user["username"]}


if __name__ == "__main__":
    print("FastAPI v4 后端已启动：http://localhost:8888")
    uvicorn.run(app, host="127.0.0.1", port=8888)