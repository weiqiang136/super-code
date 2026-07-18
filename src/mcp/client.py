"""MCP Client — 通过 stdio 与 MCP server 子进程通信。

MCP 协议简介：
  - 传输层：子进程的 stdin/stdout，每条消息是一行 JSON（JSON-RPC 2.0）
  - 握手流程：client 发 initialize → server 回 result → client 发 initialized 通知
  - 获取工具：发 tools/list → server 返回工具列表（name, description, inputSchema）
  - 调用工具：发 tools/call → server 返回 content 列表
"""
from __future__ import annotations

import json
import subprocess
import threading
from typing import Any


class MCPError(Exception):
    pass


class MCPClient:
    """管理单个 MCP server 子进程的生命周期和 JSON-RPC 通信。"""

    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None):
        self.name = name
        self._proc = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,      # 向子进程发送数据
            stdout=subprocess.PIPE,     # 从子进程接收数据
            stderr=subprocess.DEVNULL,  # 忽略 server 的调试输出
            env=env,                    # 子进程的环境变量，默认继承父进程环境
            text=True,                  # 用文本模式，而不是bytes
            encoding="utf-8",           # 
        )
        self._lock = threading.Lock()  # 保证多线程下请求串行，避免消息交错
        self._next_id = 1
        self._handshake()

    # ------------------------------------------------------------------ #
    # 公开接口
    # ------------------------------------------------------------------ #

    def list_tools(self) -> list[dict[str, Any]]:
        """返回 server 暴露的工具列表，每项含 name / description / inputSchema。"""
        result = self._call("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用指定工具，返回纯文本结果。"""
        result = self._call("tools/call", {"name": tool_name, "arguments": arguments})
        # MCP 返回 content 列表，每项有 type 和 text
        parts = [
            item.get("text", "")
            for item in result.get("content", [])
            if item.get("type") == "text"
        ]
        return "\n".join(parts) or "(no output)"

    def close(self):
        try:
            self._proc.terminate()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #

    def _handshake(self):
        """MCP 握手：initialize → initialized。必须在首次 tools/list 之前完成。"""
        self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "super-code", "version": "1.0"},
        })
        # initialized 是通知（notification），没有 id，不等待响应
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _call(self, method: str, params: dict) -> dict[str, Any]:
        """发送 JSON-RPC 请求并等待对应响应。"""
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
            self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
            return self._recv(req_id)

    def _send(self, obj: dict):
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

    def _recv(self, expected_id: int) -> dict[str, Any]:
        """
            读取行 直到收到匹配 id 的响应（跳过 server 主动推送的通知）。
            发出去的id，要和响应的id对起来
        """
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise MCPError(f"MCP server '{self.name}' closed unexpectedly")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # 忽略非 JSON 行（server 可能输出日志）
            # 通知没有 id，跳过
            if "id" not in msg:
                continue
            if msg["id"] != expected_id:    # 发出去的id，要和响应的id对起来
                continue  # 不属于本次请求，继续等
            if "error" in msg:
                raise MCPError(f"MCP error: {msg['error']}")
            return msg.get("result", {})
