"""
database.py — SQLite 数据库层
表结构：
  users    : 用户账号
  sessions : 对话会话
  messages : 聊天消息
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "./orange_agent.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 让查询结果可以用列名访问
    return conn


def init_db():
    """初始化数据库，创建所有表（若不存在）"""
    conn = get_conn()
    cursor = conn.cursor()

    # 用户表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT    NOT NULL
        )
    """)

    # 会话表
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

    conn.commit()
    conn.close()


# ─── 用户相关 ────────────────────────────────────────

def create_user(username: str, password_hash: str) -> Optional[int]:
    """创建用户，返回新用户 id；用户名重复返回 None"""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.now().isoformat()),
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None


def get_user_by_username(username: str) -> Optional[Dict]:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ─── 会话相关 ────────────────────────────────────────

def create_session(session_id: str, user_id: int, title: str = "新对话") -> Dict:
    conn = get_conn()
    conn.execute(
        "INSERT INTO sessions (id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
        (session_id, user_id, title, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return {"id": session_id, "user_id": user_id, "title": title}


def get_sessions_by_user(user_id: int) -> List[Dict]:
    """获取用户所有会话，按创建时间倒序"""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_session_title(session_id: str, title: str):
    conn = get_conn()
    conn.execute(
        "UPDATE sessions SET title = ? WHERE id = ?",
        (title, session_id),
    )
    conn.commit()
    conn.close()


def delete_session(session_id: str):
    """删除会话及其所有消息"""
    conn = get_conn()
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


def session_belongs_to_user(session_id: str, user_id: int) -> bool:
    """校验会话归属权"""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM sessions WHERE id = ? AND user_id = ?",
        (session_id, user_id),
    )
    row = cursor.fetchone()
    conn.close()
    return row is not None


# ─── 消息相关 ────────────────────────────────────────

def save_message(session_id: str, role: str, content: str, images: Optional[List[str]] = None):
    import json as _json
    conn = get_conn()
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
    import json as _json
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["images"] = _json.loads(d["images"]) if d["images"] else []
        result.append(d)
    return result