"""
api_server — 保留用户名密码，但不做密码哈希，不做 JWT
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

# 新手阅读提示：
# 这个文件是整个后端的“入口”和“接口层”。
# 前端页面发来的请求会先到这里，例如注册、登录、创建会话、发送聊天消息。
# 这里不直接负责知识库检索和大模型推理，而是把聊天任务交给 OrangeAgent。
# 可以把分工理解为：
# - api_server.py：接收请求、检查身份、返回结果
# - database.py：保存用户、会话、聊天记录
# - agent_service.py：组织智能体、调用工具、流式生成回答
# - rag_service.py：负责知识库检索、重排、联网补充等 RAG 能力

# base64 用于把前端传来的图片字符串还原成图片文件。
# json 用于把流式返回的数据包装成 JSON。
# os 用于创建临时图片目录。
# uuid 用于生成不容易重复的会话 ID 和图片文件名。
import base64
import json
import os
import uuid

# uvicorn 用于启动 FastAPI 服务。
# FastAPI 是后端框架；HTTPException 用来返回错误状态码。
# Depends 用于接口依赖，例如进入接口前先检查当前用户。
# Header 用于读取请求头里的 X-Username。
# StreamingResponse 用于 SSE 流式返回 AI 回复。
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal, Optional

# OrangeAgent 是真正负责“思考和调用工具”的智能体。
from agent_service import OrangeAgent

# 这些函数来自 database.py，负责操作 SQLite 数据库。
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

# 后端启动时先初始化数据库。
# 如果 users、sessions、messages 这些表不存在，init_db() 会自动创建。
init_db()

# 创建 FastAPI 应用对象。
# 后面所有 @app.get、@app.post、@app.put、@app.delete 都是在注册接口。
app = FastAPI(title="脐橙多模态智能体")

# CORS 跨域配置：
# 前端开发服务器通常是 http://localhost:5173，
# 后端服务通常是 http://localhost:8888。
# 浏览器会把不同端口视为不同来源，所以这里明确允许前端访问后端。
# 允许 localhost:5173 这个前端来访问我。

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 用户上传图片后，后端会先把图片保存到 temp_images 目录，
# 然后把图片路径交给智能体的图片诊断工具使用。
TEMP_IMAGE_DIR = "temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# 全局创建一个 OrangeAgent 实例。
# 所有聊天请求共用它；它内部会按 session_id 区分不同会话的短期记忆。
agent = OrangeAgent()


# ── 请求/响应模型 ───────────────────────────────────────

# 下面这些类继承 BaseModel，用来规定“前端请求体应该长什么样”。
# FastAPI 会自动根据这些模型解析和校验 JSON。

class RegisterRequest(BaseModel):
    # 注册需要用户名和密码。
    username: str
    password: str


class LoginRequest(BaseModel):
    # 登录同样需要用户名和密码。
    username: str
    password: str


class ChatRequest(BaseModel):
    # 用户输入的问题文本。
    query: str

    # 当前消息属于哪个会话。
    session_id: str

    # 图片列表。前端会把图片转成 base64 字符串传过来；没有图片时为空。
    images_base64: Optional[list[str]] = None

    # 检索模式：
    # auto  = 智能判断是否联网；
    # web   = 使用联网搜索；
    # local = 只使用本地知识库。
    search_mode: Literal["auto", "web", "local"] = "auto"


class CreateSessionRequest(BaseModel):
    # 创建会话时的标题，不传参数就叫“新对话”。
    title: str = "新对话"


class RenameSessionRequest(BaseModel):
    # 重命名会话时的新标题。
    title: str


class ClearMemoryRequest(BaseModel):
    # 要清空智能体记忆的会话 ID。
    session_id: str


# ── 当前用户：从请求头 X-Username 取 ────────────────────

def get_current_user_by_name(
    x_username: Optional[str] = Header(default=None, alias="X-Username")
) -> dict:
    """
    从请求头 X-Username 中识别当前用户。

    本项目没有使用 JWT/token，而是让前端每次请求都带用户名。
    这种方式适合本地演示和毕设展示，但真实上线时应换成更安全的登录方案。
    """
    # 请求头里没有用户名，说明前端没有提供登录身份。
    if not x_username or not x_username.strip():
        raise HTTPException(status_code=401, detail="缺少用户名，请先登录")

    # 去掉用户名首尾空格，然后去数据库查询用户。
    username = x_username.strip()
    user = get_user_by_username(username)

    # 用户名不存在时，拒绝访问需要登录的接口。
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在，请先注册")

    return user


# ── 注册 / 登录 ────────────────────────────────────────

@app.post("/api/register")
async def register(req: RegisterRequest):
    # 取出前端传来的用户名和密码。
    username = req.username.strip()
    password = req.password

    # 做基础校验，避免空用户名、过短用户名或过短密码。
    if len(username) < 2:
        raise HTTPException(status_code=400, detail="用户名至少2个字符")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少6位")

    # 这里不做哈希，直接明文存储。
    # 注意：这只适合演示项目；真实项目应该保存密码哈希，而不是保存原始密码。
    user_id = create_user(username, password)

    # create_user 返回 None，表示用户名已经存在。
    if user_id is None:
        raise HTTPException(status_code=409, detail="用户名已存在")

    # 注册成功后把用户名返回给前端。
    return {"username": username}


@app.post("/api/login")
async def login(req: LoginRequest):
    # 取出登录表单中的用户名和密码。
    username = req.username.strip()
    password = req.password

    # 与注册保持一致的基础校验。
    if len(username) < 2:
        raise HTTPException(status_code=400, detail="用户名至少2个字符")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少6位")

    # 根据用户名查用户，并直接比对明文密码。
    # 字段名叫 password_hash，但当前实际存的是明文密码。
    user = get_user_by_username(username)
    if user is None or user["password_hash"] != password:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 登录成功后，前端会保存 username，并在后续请求头里带上 X-Username。
    return {"username": user["username"]}


# ── 会话管理 ───────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions(current_user: dict = Depends(get_current_user_by_name)):
    # 获取当前登录用户的所有会话。
    # current_user 来自 Depends(get_current_user_by_name)，不是前端直接传进来的。
    sessions = get_sessions_by_user(current_user["id"])
    return {"sessions": sessions}


@app.post("/api/sessions")
async def new_session(
    req: CreateSessionRequest,
    current_user: dict = Depends(get_current_user_by_name),
):
    # 生成会话 ID。
    # 格式类似：用户ID_随机字符串，比如 3_a1b2c3d4e5f6。
    session_id = f"{current_user['id']}_{uuid.uuid4().hex[:12]}"

    # 如果前端传了标题，就使用前端标题；否则使用默认标题。
    title = req.title.strip() if req.title and req.title.strip() else "新对话"

    # 把新会话写入数据库。
    session = create_session(session_id, current_user["id"], title)
    return session


@app.put("/api/sessions/{session_id}")
async def rename_session(
    session_id: str,
    req: RenameSessionRequest,
    current_user: dict = Depends(get_current_user_by_name),
):
    # 修改前先检查这个会话是否属于当前用户。
    # 这样可以防止用户伪造别人的 session_id 去修改标题。
    if not session_belongs_to_user(session_id, current_user["id"]):
        raise HTTPException(status_code=403, detail="无权操作此会话")

    # 标题为空时退回默认值“新对话”。
    update_session_title(session_id, req.title.strip() or "新对话")
    return {"status": "ok"}


@app.delete("/api/sessions/{session_id}")
async def remove_session(
    session_id: str,
    current_user: dict = Depends(get_current_user_by_name),
):
    # 删除前同样检查会话归属。
    if not session_belongs_to_user(session_id, current_user["id"]):
        raise HTTPException(status_code=403, detail="无权操作此会话")

    # 删除数据库里的会话和对应消息。
    delete_session(session_id)

    # 清空智能体内存中这个会话的短期记忆。
    agent.clear_memory(session_id)
    return {"status": "ok"}


@app.get("/api/sessions/{session_id}/messages")
async def get_history(
    session_id: str,
    current_user: dict = Depends(get_current_user_by_name),
):
    # 读取历史消息前，也必须确认会话属于当前用户。
    if not session_belongs_to_user(session_id, current_user["id"]):
        raise HTTPException(status_code=403, detail="无权访问此会话")

    # 从数据库读取该会话下的所有消息，返回给前端恢复聊天界面。
    messages = get_messages_by_session(session_id)
    return {"messages": messages}


# ── 聊天（无 JWT，按用户名识别用户）─────────────────────

@app.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user_by_name),
):
    # 从请求体中取出聊天所需的数据。
    user_input = request.query or ""
    session_id = request.session_id
    images_list = request.images_base64 or []
    search_mode = request.search_mode

    # 用户必须至少输入文字或上传图片，否则没有可处理的内容。
    if not user_input.strip() and not images_list:
        raise HTTPException(status_code=400, detail="内容不能为空")

    # 聊天前检查会话归属，防止用户访问或写入别人的会话。
    if not session_belongs_to_user(session_id, current_user["id"]):
        raise HTTPException(status_code=403, detail="无权访问此会话")

    # 如果前端上传了图片，它们会以 base64 字符串形式传到后端。
    # 这里把图片逐张解码并保存为本地 jpg 文件。
    saved_paths = []
    for i, b64_str in enumerate(images_list):
        try:
            # base64 图片通常形如：
            # data:image/jpeg;base64,/9j/4AAQ...
            # 逗号前是格式说明，逗号后才是真正的图片内容。
            _, encoded = b64_str.split(",", 1)
            img_bytes = base64.b64decode(encoded)

            # 文件名里放入 session_id、图片序号和随机后缀，减少重名概率。
            path = f"{TEMP_IMAGE_DIR}/up_{session_id}_{i}_{uuid.uuid4().hex[:4]}.jpg"
            with open(path, "wb") as f:
                f.write(img_bytes)
            saved_paths.append(path)
        except Exception:
            # 单张图片处理失败时跳过，不让整个聊天请求失败。
            continue

    # final_query 是真正交给智能体的内容。
    # 如果有图片，需要把图片路径补充进问题里，智能体后续才能调用图片诊断工具。
    final_query = user_input
    if saved_paths:
        paths_str = ", ".join([f"'{p}'" for p in saved_paths])
        if final_query.strip():
            final_query += f"。请综合分析这 {len(saved_paths)} 张图片，路径：{paths_str}。"
        else:
            final_query = f"请综合分析这 {len(saved_paths)} 张图片，路径：{paths_str}。"

    # 先保存用户原始消息。
    # 这里保存 user_input，而不是 final_query，是为了让历史记录显示用户真正输入的内容。
    save_message(
        session_id=session_id,
        role="user",
        content=user_input,
        images=images_list if images_list else None,
    )

    async def event_generator():
        """
        SSE 流式响应生成器。

        普通接口是等结果全部生成完再一次性返回；
        这里是一边从智能体拿回答片段，一边推给前端，所以页面能显示“正在输出”的效果。
        """
        accumulated = ""
        ask_user_content = ""  # 单独收集追问内容，用于写库
        try:
            # 调用智能体的流式聊天方法。
            # chunk 可能是普通回答，也可能是特殊状态标记。
            async for chunk in agent.chat_stream(
                final_query,
                session_id=session_id,
                search_mode=search_mode,
            ):
                if chunk.startswith("__ASK_USER__:"):
                    # 提取追问文本，等流结束后写库
                    ask_user_content = chunk.replace("__ASK_USER__:", "").strip()
                elif not chunk.startswith("__STATUS__:"):
                    # __STATUS__ 是过程提示，例如“正在查询知识库”，不应该保存进正式回答。
                    accumulated += chunk

                # 按 SSE 格式返回给前端。
                # 格式必须是 data: xxx\n\n，前端才能按事件流读取。
                payload = json.dumps({"text": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

            # 告诉前端本次回答已经结束。
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
            # 出错时也用 SSE 返回错误信息，前端可以把错误显示在聊天气泡里。
            error_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_payload}\n\n"

    # text/event-stream 表示这是 SSE 流式响应。
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── 其他接口 ───────────────────────────────────────────

@app.post("/api/clear_memory")
async def clear_memory(
    request: ClearMemoryRequest,
    current_user: dict = Depends(get_current_user_by_name),
):
    # 清空记忆前先确认当前用户拥有这个会话。
    if not session_belongs_to_user(request.session_id, current_user["id"]):
        raise HTTPException(status_code=403, detail="无权操作此会话")

    # 这里只清空智能体内存中的短期记忆，不删除数据库里的聊天记录。
    agent.clear_memory(request.session_id)
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    # 健康检查接口：用于快速确认后端服务是否正常启动。
    return {"status": "running"}


@app.get("/api/me")
async def me(current_user: dict = Depends(get_current_user_by_name)):
    # 返回当前登录用户信息。
    return {"id": current_user["id"], "username": current_user["username"]}


if __name__ == "__main__":
    # 直接运行 python api_server.py 时，从这里启动后端服务。
    print("FastAPI v4 后端已启动：http://localhost:8888")
    uvicorn.run(app, host="127.0.0.1", port=8888)
