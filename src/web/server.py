"""Phase 1B：Engine→WebSocket 核心管道。

最小 FastAPI 应用，通过 WebSocket 将 Engine.submit() 事件流桥接到浏览器。
启动方式：python -m src.web.server
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# 确保 src/ 在 import 路径中（以模块方式运行时）
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.config import load_app_config
from core.context import build_system_prompt
from core.engine import Engine
from core.permissions import PermissionChecker
from core.session import SessionStore
from features.cost_tracker import CostTracker
from features.memory import get_memory_dir, ensure_memory_dir
from features.skills import discover_skills, build_skills_prompt_section

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

    return result


# ============================================================
# 应用初始化
# ============================================================

app = FastAPI(title="super-code", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
async def index():
    return (_WEB_STATIC / "index.html").read_text(encoding="utf-8")


# ============================================================
# WebSocket 端点
# ============================================================

@app.websocket("/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    send_queue: asyncio.Queue = asyncio.Queue()

    # 发送协程：从队列取出事件，推送到 WebSocket
    async def _sender():
        while True:
            event = await send_queue.get()
            if event[0] == "_done":
                return
            try:
                await ws.send_json(_serialize_event(event))
            except Exception:
                return

    engine: Engine | None = None

    try:
        while True:
            raw = await ws.receive_text()
            if not raw.strip():
                continue
            print(f"[DEBUG] 收到消息: {raw[:80]}...")

            # P1：单实例 Engine，首条消息时创建
            if engine is None:
                cwd = str(Path.cwd())
                app_config = load_app_config(argparse.Namespace(
                    config=None, provider=None, model=None, max_tokens=None,
                    api_key=None, base_url=None,
                ))
                memory_dir = get_memory_dir(Path(cwd))
                ensure_memory_dir(memory_dir)
                discover_skills(cwd)
                skills_section = build_skills_prompt_section()
                system_prompt = build_system_prompt(cwd=cwd, model=app_config.model,
                                                    memory_dir=memory_dir)
                if skills_section:
                    system_prompt = system_prompt + "\n\n" + skills_section

                from tui.app import _build_engine
                engine = _build_engine(
                    app_config=app_config,
                    cwd=cwd,
                    sandbox=None,
                    permission_checker=PermissionChecker(auto_approve=True),
                    system_prompt=system_prompt,
                    session_store=SessionStore(cwd=cwd, model=app_config.model),
                    cost_tracker=CostTracker(),
                )

            # 提交用户消息 → 在线程池中迭代事件流
            _executor.submit(_iterate, engine.submit(raw), send_queue, loop)
            await _sender()

    except WebSocketDisconnect:
        pass


# ============================================================
# 入口
# ============================================================

def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8123, log_level="info")


if __name__ == "__main__":
    main()
