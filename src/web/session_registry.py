"""Phase 2：会话生命周期管理 — 元数据、Engine 实例池、磁盘持久化。

SessionRegistry 管理 SessionStore（元数据）和 Engine（按需创建）。
REST 端点只操作元数据，Engine 在 WebSocket 连接时延迟创建。
"""
from __future__ import annotations

import asyncio
import json
import os

from core.engine import Engine
from core.permissions import PermissionChecker
from core.session import SessionStore, _sanitize_cwd, _SESSIONS_ROOT, _now_iso


class SessionRegistry:
    """管理 Web UI 所有会话的元数据和 Engine 实例。"""

    def __init__(self, engine_factory, app_config, cwd: str, sandbox=None):
        self._factory = engine_factory
        self._app_config = app_config
        self._cwd = cwd
        self._sandbox = sandbox
        self._sessions: dict[str, SessionStore] = {}        # session_id → SessionStore
        self._engines: dict[str, Engine] = {}                # session_id → Engine
        self._checkers: dict[str, PermissionChecker] = {}   # session_id → PermissionChecker
        self._active_queues: dict[str, asyncio.Queue] = {}  # session_id → send_queue（同会话互斥）

    # ------------------------------------------------------------------
    # 会话 CRUD（REST 端点用）
    # ------------------------------------------------------------------

    def create_session(self) -> dict:
        """创建新会话（仅元数据，Engine 延迟到 WebSocket 连接时创建）。"""
        store = SessionStore(cwd=self._cwd, model=self._app_config.model)
        self._sessions[store.session_id] = store
        return {"session_id": store.session_id, "cwd": self._cwd}

    def list_sessions(self) -> list[dict]:
        """返回当前工作目录下的所有会话，按更新时间倒序。"""
        metas = SessionStore.list_sessions(self._cwd)
        return [
            {
                "session_id": m.session_id,
                "title": m.title,
                "cwd": m.cwd,
                "model": m.model,
                "created_at": m.created_at,
                "updated_at": m.updated_at,
                "message_count": m.message_count,
            }
            for m in metas
        ]

    def rename_session(self, session_id: str, title: str) -> bool:
        """重命名会话标题。直接更新 meta.json 文件。"""
        d = _SESSIONS_ROOT / _sanitize_cwd(self._cwd)
        meta_path = d / f"{session_id}.meta.json"
        if not meta_path.exists():
            return False
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["title"] = title
        meta["updated_at"] = _now_iso()
        # 原子写入：写临时文件 → os.replace
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False)
        os.replace(tmp, meta_path)
        # 同步内存中的 SessionStore（如果已加载）
        store = self._sessions.get(session_id)
        if store:
            store._title = title
        return True

    def remove_session(self, session_id: str) -> bool:
        """删除会话：清理 Engine + 删除 JSONL 和 meta 文件。"""
        self._engines.pop(session_id, None)
        self._checkers.pop(session_id, None)
        self._sessions.pop(session_id, None)

        d = _SESSIONS_ROOT / _sanitize_cwd(self._cwd)
        deleted = False
        for ext in (".jsonl", ".meta.json"):
            f = d / f"{session_id}{ext}"
            if f.exists():
                f.unlink()
                deleted = True
        return deleted

    # ------------------------------------------------------------------
    # Engine 生命周期（WebSocket 端点用）
    # ------------------------------------------------------------------

    def get_or_create_engine(
        self, session_id: str, permission_checker: PermissionChecker, system_prompt: str
    ) -> Engine:
        """获取已有 Engine 或创建新实例（WebSocket 连接时调用）。

        重连场景：Engine 已存在则直接返回，PermissionChecker 的 prompt_handler
        由 P3 的 server.py 通过 get_checker() + set_prompt_handler() 替换。
        """
        if session_id not in self._engines:
            store = self._sessions.get(session_id)
            if store is None:
                # 恢复历史会话：从磁盘加载元数据
                meta, _ = SessionStore.load_session(session_id, self._cwd)
                if meta is None:
                    raise ValueError(f"Session {session_id} not found")
                store = SessionStore(
                    cwd=self._cwd, model=meta.model, session_id=session_id,
                )
                self._sessions[session_id] = store
            engine = self._factory(
                app_config=self._app_config,
                cwd=self._cwd,
                sandbox=self._sandbox,
                permission_checker=permission_checker,
                system_prompt=system_prompt,
                session_store=store,
            )
            self._engines[session_id] = engine
            self._checkers[session_id] = permission_checker
        return self._engines[session_id]

    def get_checker(self, session_id: str) -> PermissionChecker | None:
        """返回会话的 PermissionChecker（P3 重连时替换 prompt_handler 用）。"""
        return self._checkers.get(session_id)

    def get_messages(self, session_id: str) -> list[dict]:
        """加载会话历史消息（WebSocket 连接时发给前端恢复对话）。"""
        return SessionStore.load_messages(session_id, self._cwd)

    # ------------------------------------------------------------------
    # 连接追踪 & 优雅关闭（Phase 5）
    # ------------------------------------------------------------------

    def register_connection(self, session_id: str, send_queue: asyncio.Queue):
        """注册 WebSocket 连接。若同会话已有旧连接，踢旧接新。"""
        old = self._active_queues.get(session_id)
        if old is not None:
            try:
                old.put_nowait(("_kicked",))
            except asyncio.QueueFull:
                pass
        self._active_queues[session_id] = send_queue

    def unregister_connection(self, session_id: str, send_queue: asyncio.Queue):
        """移除 WebSocket 连接追踪（仅当队列匹配，防止误删新连接）。"""
        current = self._active_queues.get(session_id)
        if current is send_queue:
            self._active_queues.pop(session_id, None)

    def shutdown(self):
        """通知所有活跃客户端关闭，清理引擎资源。"""
        for q in list(self._active_queues.values()):
            try:
                q.put_nowait(("_shutdown",))
            except asyncio.QueueFull:
                pass
        self._active_queues.clear()
        self._engines.clear()
        self._checkers.clear()
