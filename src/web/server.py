"""Phase 2：FastAPI + WebSocket 服务，集成会话管理和 Engine 桥接。

启动方式：python -m src.web.server [--port 8123]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from web.permission import WebPermissionHandler

# 确保 src/ 在 import 路径中（以模块方式运行时）
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.config import load_app_config
from core.context import build_system_prompt
from core.permissions import PermissionChecker
from features.memory import get_memory_dir, ensure_memory_dir
from features.skills import discover_skills, build_skills_prompt_section
from web.session_registry import SessionRegistry

# ============================================================
# 辅助函数
# ============================================================

_executor = ThreadPoolExecutor(max_workers=10)

_WEB_STATIC = _THIS_FILE.parent / "static"


def _iterate(gen, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    """线程工作函数：迭代同步生成器，将事件推入异步队列。"""
    try:
        for event in gen:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)
    except Exception as exc:
        asyncio.run_coroutine_threadsafe(
            queue.put(("error", f"Engine 执行异常：{exc}")), loop,
        )
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(("_done",)), loop)


def _serialize_event(event: tuple) -> dict:
    """将 Engine 事件元组转换为可 JSON 序列化的字典。"""
    etype = event[0]
    result: dict[str, Any] = {"type": etype}

    if etype == "text":
        result["text"] = event[1]
    elif etype in ("tool_call", "tool_executing", "tool_result"):
        result["tool_name"] = event[1]
        result["tool_input"] = event[2]
        if etype == "tool_result":
            tr = event[3]
            result["tool_content"] = tr.content if hasattr(tr, "content") else str(tr)
            result["is_error"] = getattr(tr, "is_error", False)
        result["tool_use_id"] = event[4]
    elif etype == "error":
        result["message"] = event[1]
    elif etype == "permission_request":
        result.update(event[1])

    return result


# ============================================================
# 应用初始化
# ============================================================

_cwd = str(Path.cwd())
print(f"[web] 工作目录: {_cwd}")

# 加载配置
_app_config = load_app_config(argparse.Namespace(
    config=None, provider=None, model=None, max_tokens=None,
    api_key=None, base_url=None,
))
print(f"[web] 模型: {_app_config.model}")

# 记忆 & skill 初始化（同 CLI）
_memory_dir = get_memory_dir(Path(_cwd))
ensure_memory_dir(_memory_dir)
discover_skills(_cwd)
_skills_section = build_skills_prompt_section()
_system_prompt = build_system_prompt(cwd=_cwd, model=_app_config.model, memory_dir=_memory_dir)
if _skills_section:
    _system_prompt = _system_prompt + "\n\n" + _skills_section

# 复用 CLI 的 _build_engine 工厂
from tui.app import _build_engine

_registry = SessionRegistry(
    engine_factory=_build_engine,
    app_config=_app_config,
    cwd=_cwd,
    sandbox=None,
)

# FastAPI 应用
app = FastAPI(title="super-code Web UI", docs_url=None, redoc_url=None)


# ============================================================
# 静态文件
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = _WEB_STATIC / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>super-code Web UI</h1><p>Static files not found.</p>")


# ============================================================
# REST：会话管理
# ============================================================

@app.post("/api/sessions")
async def create_session():
    """创建新会话，返回 session_id 和 cwd。"""
    result = _registry.create_session()
    return JSONResponse(result)


@app.get("/api/sessions")
async def list_sessions():
    """列出当前工作目录下的所有历史会话。"""
    return JSONResponse(_registry.list_sessions())


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, req: Request):
    body = await req.json()
    title = body.get("title", "")
    if not title:
        return JSONResponse({"error": "title is required"}, status_code=400)
    ok = _registry.rename_session(session_id, title)
    if not ok:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return JSONResponse({"ok": True})


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话及其所有磁盘文件。"""
    ok = _registry.remove_session(session_id)
    if not ok:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return JSONResponse({"ok": True})


# ============================================================
# WebSocket：聊天端点（按 session_id 隔离）
# ============================================================

@app.websocket("/chat/{session_id}")
async def ws_chat(ws: WebSocket, session_id: str):
    await ws.accept()
    loop = asyncio.get_running_loop()
    send_queue: asyncio.Queue = asyncio.Queue()

    # 后台发送任务：持续消费 send_queue，推送到 WebSocket
    async def _sender():
        while True:
            event = await send_queue.get()
            try:
                await ws.send_json(_serialize_event(event))
            except Exception:
                break

    sender_task = asyncio.create_task(_sender())

    try:
        # 检查是否重连（已有 checker 表示 Engine 存活）
        existing_checker = _registry.get_checker(session_id)
        if existing_checker is not None:
            handler = WebPermissionHandler(send_queue, loop)
            existing_checker.set_prompt_handler(handler)
            engine = _registry.get_or_create_engine(
                session_id, existing_checker, _system_prompt,
            )
        else:
            handler = WebPermissionHandler(send_queue, loop)
            checker = PermissionChecker(prompt_handler=handler)
            engine = _registry.get_or_create_engine(session_id, checker, _system_prompt)

        # 发送历史消息，恢复对话上下文
        history = _registry.get_messages(session_id)
        if history:
            await ws.send_json({"type": "history", "messages": history})

        # 消息循环：接收文本 / 权限响应 / 中止
        while True:
            raw = await ws.receive_text()

            # 尝试解析 JSON（权限响应/中止命令）
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = None

            if data and data.get("type") == "permission_response":
                handler.resolve(data["request_id"], data["choice"])
                continue

            if data and data.get("type") == "abort":
                engine.abort()
                continue

            if data and data.get("type") == "stop":
                engine.abort()
                continue

            # 普通文本消息
            if not raw.strip():
                continue
            _executor.submit(_iterate, engine.submit(raw), send_queue, loop)

    except WebSocketDisconnect:
        pass
    except ValueError as e:
        await ws.send_json({"type": "error", "message": str(e)})
    finally:
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass


# ============================================================
# 入口
# ============================================================

def main():
    import uvicorn

    parser = argparse.ArgumentParser(prog="super-code-web", description="super-code Web UI")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    print(f"[web] 启动服务: http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
