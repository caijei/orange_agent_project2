"""
database.py — SQLite 数据库层
表结构：
  users    : 用户账号
  sessions : 对话会话
  messages : 聊天消息
"""

# 新手阅读提示：
# 这个文件只负责“存数据”和“取数据”，不负责接口、不负责 AI 回答。
# api_server.py 会调用这里的函数来完成注册、登录、会话管理和聊天记录保存。
# 数据关系可以理解为：
# 一个 user 可以有多个 session；
# 一个 session 可以有多条 message。

# sqlite3 是 Python 自带的轻量数据库工具，不需要单独安装数据库服务器。
import sqlite3

# os 用来读取环境变量，例如 DB_PATH。
import os

# datetime 用来记录用户、会话、消息的创建时间。
from datetime import datetime

# 类型提示：让函数参数和返回值更容易看懂。
from typing import List, Optional, Dict, Any
from pathlib import Path

# 数据库文件路径。
# 如果环境变量里设置了 DB_PATH，就使用环境变量；
# 否则默认使用当前运行目录下的 orange_agent.db。
DB_PATH = os.getenv("DB_PATH", "./orange_agent.db")


def get_conn() -> sqlite3.Connection:
    """
    创建并返回一个 SQLite 数据库连接。

    每次执行数据库操作时，都会先调用这个函数获取连接。
    操作完成后，调用方需要 conn.close() 关闭连接。
    """
    # 连接数据库文件；如果文件不存在，SQLite 会自动创建。
    conn = sqlite3.connect(DB_PATH)

    # 让查询结果可以通过字段名访问。
    # 例如 row["username"]，而不是只能用 row[1]。
    conn.row_factory = sqlite3.Row  # 让查询结果可以用列名访问
    return conn


def init_db():
    """
    初始化数据库，创建所有表（若不存在）。

    api_server.py 启动时会先调用 init_db()。
    因为使用 CREATE TABLE IF NOT EXISTS，所以重复调用不会重复建表。
    """
    conn = get_conn()
    cursor = conn.cursor()

    # 用户表
    # users 用来保存账号信息。
    # id 是自增主键；username 唯一；password_hash 当前实际保存的是明文密码。
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT    NOT NULL
        )
    """)

    # 会话表
    # sessions 用来保存每个用户的聊天窗口。
    # user_id 指向 users.id，表示这个会话属于哪个用户。
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id         TEXT    PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            title      TEXT    NOT NULL DEFAULT '新对话',
            created_at TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # 消息表
    # messages 用来保存每个会话中的聊天记录。
    # role 通常是 user 或 assistant；
    # images 保存的是 JSON 字符串，不是图片文件本身。
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT    NOT NULL,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            images     TEXT,           -- JSON 字符串，存图片 base64 列表
            created_at TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    # 对数据库做了创建/修改操作后，需要 commit 才会真正保存。
    conn.commit()

    # 用完连接要关闭，避免资源占用。
    conn.close()


# ─── 用户相关 ────────────────────────────────────────

def create_user(username: str, password_hash: str) -> Optional[int]:
    """
    创建用户，返回新用户 id；用户名重复返回 None。

    对应 api_server.py 里的 /api/register 注册接口。
    """
    try:
        conn = get_conn()
        cursor = conn.cursor()

        # 使用 ? 占位符传参，而不是手动拼接 SQL 字符串。
        # 这样可以减少 SQL 注入风险。
        cursor.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.now().isoformat()),
        )
        conn.commit()

        # lastrowid 是刚刚插入的用户自增 id。
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        # username 有 UNIQUE 约束，重复注册会触发 IntegrityError。
        return None


def get_user_by_username(username: str) -> Optional[Dict]:
    """
    根据用户名查询用户。

    用途：
    1. 登录时查用户并比对密码；
    2. 根据请求头 X-Username 识别当前用户。
    """
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    # 查到就转成普通 dict；没查到就返回 None。
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """
    根据用户 id 查询用户。

    当前项目中不一定频繁使用，但保留这个函数方便以后扩展。
    """
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ─── 会话相关 ────────────────────────────────────────

def create_session(session_id: str, user_id: int, title: str = "新对话") -> Dict:
    """
    创建一个新的聊天会话。

    对应 api_server.py 里的 POST /api/sessions。
    """
    conn = get_conn()
    conn.execute(
        "INSERT INTO sessions (id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
        (session_id, user_id, title, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

    # 返回给前端，前端会用这些信息更新左侧会话列表。
    return {"id": session_id, "user_id": user_id, "title": title}


def get_sessions_by_user(user_id: int) -> List[Dict]:
    """
    获取某个用户的所有会话，按创建时间倒序。

    对应 api_server.py 里的 GET /api/sessions。
    """
    conn = get_conn()
    cursor = conn.cursor()

    # 只取当前用户自己的会话，避免把其他用户的会话也返回给前端。
    cursor.execute(
        "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_session_title(session_id: str, title: str):
    """
    修改会话标题。

    注意：这个函数只负责修改数据库，不负责权限校验。
    会话是否属于当前用户，由 api_server.py 先调用 session_belongs_to_user 判断。
    """
    conn = get_conn()
    conn.execute(
        "UPDATE sessions SET title = ? WHERE id = ?",
        (title, session_id),
    )
    conn.commit()
    conn.close()


def delete_session(session_id: str):
    """
    删除会话及其所有消息。

    对应 api_server.py 里的 DELETE /api/sessions/{session_id}。
    """
    conn = get_conn()

    # 先删除消息，再删除会话，避免留下没有归属的消息记录。
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


def session_belongs_to_user(session_id: str, user_id: int) -> bool:
    """
    校验会话归属权。

    这是一个非常重要的安全检查：
    用户只能访问、修改、删除自己的会话，不能操作别人的会话。
    """
    conn = get_conn()
    cursor = conn.cursor()

    # 同时匹配 session_id 和 user_id。
    # 查得到，说明这个会话确实属于当前用户。
    cursor.execute(
        "SELECT id FROM sessions WHERE id = ? AND user_id = ?",
        (session_id, user_id),
    )
    row = cursor.fetchone()
    conn.close()
    return row is not None


# ─── 消息相关 ────────────────────────────────────────

def save_message(session_id: str, role: str, content: str, images: Optional[List[str]] = None):
    """
    保存一条聊天消息。

    role 通常是：
    - user：用户发的消息；
    - assistant：AI 回复的消息。

    对应 api_server.py 里的 /api/chat：
    用户消息会先保存一次，AI 完成回复后再保存一次。
    """
    import json as _json
    conn = get_conn()

    # images 是 Python 列表，SQLite 不能直接存列表。
    # 所以这里先把图片列表转换成 JSON 字符串再存入 images 字段。
    conn.execute(
        "INSERT INTO messages (session_id, role, content, images, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            session_id,
            role,
            content,
            _json.dumps(images, ensure_ascii=False) if images else None,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_messages_by_session(session_id: str) -> List[Dict]:
    """
    获取某个会话下的所有聊天消息。

    对应 api_server.py 里的 GET /api/sessions/{session_id}/messages。
    前端刷新页面或切换历史会话时，会调用这个接口恢复聊天记录。
    """
    import json as _json
    conn = get_conn()
    cursor = conn.cursor()

    # 按时间升序，保证消息显示顺序是从第一条到最后一条。
    cursor.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)

        # 数据库里的 images 是 JSON 字符串；
        # 返回给前端前，转换回列表。没有图片时返回空列表，方便前端处理。
        d["images"] = _json.loads(d["images"]) if d["images"] else []
        result.append(d)
    return result
