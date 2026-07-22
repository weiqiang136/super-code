"""Phase 3：Web 端权限确认适配器。

WebPermissionHandler 将权限确认请求通过 asyncio.Queue 桥接到浏览器，
在线程池中的 Engine 工作线程里阻塞等待浏览器返回结果。
"""
from __future__ import annotations

import asyncio
import threading
import uuid


class WebPermissionHandler:
    """通过 WebSocket 发送权限确认请求，阻塞等待浏览器返回结果。

    设计要点：
    - __call__ 在 Engine 工作线程（线程池）中调用，阻塞等待
    - resolve 在事件循环线程中调用，解除阻塞
    - 120 秒超时：浏览器断连后 Engine 线程不会永久挂死
    """

    def __init__(self, ws_send_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self._ws_queue = ws_send_queue
        self._loop = loop
        self._pending: dict[str, dict] = {}  # request_id → {event, result}
        self._timeout = 120  # 浏览器断连超时（秒）

    def __call__(self, tool, inputs) -> str:
        """Engine 工作线程内调用。阻塞等待浏览器返回 allow/deny/always。

        返回值语义与 PermissionChecker._prompt_user 一致：
        "allow" / "deny" / "always"
        """
        request_id = uuid.uuid4().hex[:8]
        event = threading.Event()
        entry = {"event": event, "result": None}
        self._pending[request_id] = entry

        # 截断参数值，避免超大内容撑爆 WebSocket 消息
        safe_inputs = {}
        for k, v in inputs.items():
            s = str(v)
            safe_inputs[k] = s[:500] + ("..." if len(s) > 500 else "")

        asyncio.run_coroutine_threadsafe(
            self._ws_queue.put(("permission_request", {
                "request_id": request_id,
                "tool_name": tool.name,
                "inputs": safe_inputs,
            })),
            self._loop,
        )

        if not event.wait(timeout=self._timeout):
            self._pending.pop(request_id, None)
            return "deny"

        result = entry.get("result", "deny")
        self._pending.pop(request_id, None)
        return result

    def resolve(self, request_id: str, choice: str) -> None:
        """浏览器返回结果时由 WebSocket handler（事件循环线程）调用。"""
        entry = self._pending.get(request_id)
        if entry:
            entry["result"] = choice
            entry["event"].set()
