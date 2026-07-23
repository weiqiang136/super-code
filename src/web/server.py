"""Phase 4：FastAPI + WebSocket 服务，含 ChatServer 封装。

启动方式：python -m src.web.server [--port 8123] 或 super-code --web
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

# 确保 src/ 在 import 路径中（以模块方式运行时）
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.config import load_app_config, AppConfig
from core.context import build_system_prompt
from core.permissions import PermissionChecker
from features.memory import get_memory_dir, ensure_memory_dir
from features.skills import discover_skills, build_skills_prompt_section
from web.permission import WebPermissionHandler
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
        # AbortedError 是用户主动中止（点停止/Ctrl+C），不是异常，静默结束
        exc_name = type(exc).__name__
        if exc_name == "AbortedError":
            return
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
# ChatServer 封装
# ============================================================

class ChatServer:
    """Web UI 服务端：管理 Engine 池、REST 路由、WebSocket 端点。"""

    def __init__(self, app_config: AppConfig, cwd: str, sandbox=None):
        self._app_config = app_config
        self._cwd = cwd
        self._sandbox = sandbox

        # 记忆 & skill 初始化（同 CLI）
        memory_dir = get_memory_dir(Path(cwd))
        ensure_memory_dir(memory_dir)
        discover_skills(cwd)
        skills_section = build_skills_prompt_section()
        system_prompt = build_system_prompt(
            cwd=cwd, model=app_config.model, memory_dir=memory_dir,
        )
        if skills_section:
            system_prompt = system_prompt + "\n\n" + skills_section
        self._system_prompt = system_prompt

        # 复用 CLI 的 _build_engine 工厂
        from tui.app import _build_engine
        self._registry = SessionRegistry(
            engine_factory=_build_engine,
            app_config=app_config,
            cwd=cwd,
            sandbox=sandbox,
        )

        # FastAPI 应用
        self._app = FastAPI(title="super-code Web UI", docs_url=None, redoc_url=None)
        self._register_routes()
        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # 路由注册
    # ------------------------------------------------------------------

    def _register_routes(self):
        app = self._app
        registry = self._registry
        system_prompt = self._system_prompt

        # ----- 静态文件 -----
        @app.get("/", response_class=HTMLResponse)
        async def index():
            index_path = _WEB_STATIC / "index.html"
            if index_path.exists():
                return index_path.read_text(encoding="utf-8")
            return HTMLResponse("<h1>super-code Web UI</h1><p>Static files not found.</p>")

        # ----- 服务信息 -----
        @app.get("/api/info")
        async def server_info():
            return JSONResponse({
                "cwd": self._cwd,
                "model": self._app_config.model,
            })

        # ----- 会话 CRUD -----
        @app.post("/api/sessions")
        async def create_session():
            result = registry.create_session()
            return JSONResponse(result)

        @app.get("/api/sessions")
        async def list_sessions():
            return JSONResponse(registry.list_sessions())

        @app.patch("/api/sessions/{session_id}")
        async def rename_session(session_id: str, req: Request):
            body = await req.json()
            title = body.get("title", "")
            if not title:
                return JSONResponse({"error": "title is required"}, status_code=400)
            ok = registry.rename_session(session_id, title)
            if not ok:
                return JSONResponse({"error": "session not found"}, status_code=404)
            return JSONResponse({"ok": True})

        @app.delete("/api/sessions/{session_id}")
        async def delete_session(session_id: str):
            ok = registry.remove_session(session_id)
            if not ok:
                return JSONResponse({"error": "session not found"}, status_code=404)
            return JSONResponse({"ok": True})

        # ----- WebSocket 聊天 -----
        @app.websocket("/chat/{session_id}")
        async def ws_chat(ws: WebSocket, session_id: str):
            await ws.accept()
            loop = asyncio.get_running_loop()
            send_queue: asyncio.Queue = asyncio.Queue()

            async def _sender():
                while True:
                    event = await send_queue.get()
                    if event[0] == "_kicked":
                        try:
                            await ws.send_json({"type": "kicked", "message": "会话已在其他标签页打开"})
                        except Exception:
                            pass
                        break
                    if event[0] == "_shutdown":
                        try:
                            await ws.send_json({"type": "shutdown", "message": "服务端关闭"})
                        except Exception:
                            pass
                        break
                    try:
                        await ws.send_json(_serialize_event(event))
                    except Exception:
                        break

            sender_task = asyncio.create_task(_sender())

            # 注册连接（踢旧连接）
            registry.register_connection(session_id, send_queue)

            try:
                # 重连检查
                existing_checker = registry.get_checker(session_id)
                if existing_checker is not None:
                    handler = WebPermissionHandler(send_queue, loop)
                    existing_checker.set_prompt_handler(handler)
                    engine = registry.get_or_create_engine(
                        session_id, existing_checker, system_prompt,
                    )
                else:
                    handler = WebPermissionHandler(send_queue, loop)
                    checker = PermissionChecker(prompt_handler=handler)
                    engine = registry.get_or_create_engine(
                        session_id, checker, system_prompt,
                    )

                # 发送历史消息
                history = registry.get_messages(session_id)
                if history:
                    await ws.send_json({"type": "history", "messages": history})

                # 消息循环
                while True:
                    raw = await ws.receive_text()
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = None

                    if data and data.get("type") == "permission_response":
                        handler.resolve(data["request_id"], data["choice"])
                        continue
                    if data and data.get("type") in ("abort", "stop"):
                        engine.abort()
                        continue
                    if not raw.strip():
                        continue
                    _executor.submit(_iterate, engine.submit(raw), send_queue, loop)

            except WebSocketDisconnect:
                pass
            except ValueError as e:
                await ws.send_json({"type": "error", "message": str(e)})
            finally:
                registry.unregister_connection(session_id, send_queue)
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # 启动 & 关闭
    # ------------------------------------------------------------------

    def shutdown(self):
        """通知所有客户端，清理 Engine 和 MCP 资源。"""
        self._registry.shutdown()
        try:
            from mcp.loader import shutdown_mcp
            shutdown_mcp()
        except Exception:
            pass

    def run(self, port: int = 8123, open_browser: bool = True):
        """启动 uvicorn 服务器。"""
        if open_browser:

            def _open_delayed():
                time.sleep(0.5)
                webbrowser.open(f"http://localhost:{port}")

            threading.Thread(target=_open_delayed, daemon=True).start()

        import uvicorn
        uvicorn.run(self._app, host="127.0.0.1", port=port, log_level="info")


# ============================================================
# 模块直接启动入口（python -m src.web.server）
# ============================================================

def main():
    parser = argparse.ArgumentParser(prog="super-code-web", description="super-code Web UI")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    cwd = str(Path.cwd())
    print(f"[web] 工作目录: {cwd}")

    app_config = load_app_config(argparse.Namespace(
        config=None, provider=None, model=None, max_tokens=None,
        api_key=None, base_url=None,
    ))
    print(f"[web] 模型: {app_config.model}")

    server = ChatServer(app_config=app_config, cwd=cwd)
    print(f"[web] 启动服务: http://localhost:{args.port}")
    server.run(port=args.port)


if __name__ == "__main__":
    main()
