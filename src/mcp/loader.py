"""MCP Loader — 读取 MCP 配置，启动所有 MCP server 并返回工具列表。

配置文件格式：
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {}          // 可选，追加到当前环境变量
    }
  }
}

查找顺序：当前工作目录（.mcp.json）→ exe 同级目录（mcp.json）→ 全局配置目录（~/.config/super-code/mcp.json）
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from core.tool import Tool
from mcp.client import MCPClient
from mcp.tool_proxy import build_mcp_tools

# 全局持有所有已启动的 client，程序退出时统一关闭
_active_clients: list[MCPClient] = []


def load_mcp_tools(cwd: str | Path = ".") -> list[Tool]:
    """读取配置、启动 server、返回所有 MCP 工具代理。启动失败的 server 跳过并打印警告。"""
    config = _find_config(Path(cwd))
    if config is None:
        return []

    servers = config.get("mcpServers", {})
    tools: list[Tool] = []

    for name, spec in servers.items():  # 这里的servers是个字典，所以上述注释是没问题的
        command = spec.get("command", "")
        # Windows 下 Popen 不会自动补 .cmd/.bat 后缀，需用 shutil.which 解析为绝对路径
        resolved = shutil.which(command)
        if resolved:
            command = resolved
        args = spec.get("args", [])
        # 合并当前环境变量 + server 自定义 env（server 可能需要 API key 等）
        env = {**os.environ, **spec.get("env", {})} if spec.get("env") else None

        try:
            client = MCPClient(name=name, command=command, args=args, env=env)
            _active_clients.append(client)
            server_tools = build_mcp_tools(client)
            tools.extend(server_tools)
            print(f"[MCP] {name}: {len(server_tools)} tool(s) loaded")
        except Exception as e:
            # 单个 server 启动失败不影响整体，打印警告继续
            print(f"[MCP] Warning: failed to start server '{name}': {e}")

    return tools


def shutdown_mcp():
    """关闭所有 MCP server 子进程，在程序退出时调用。"""
    for client in _active_clients:
        client.close()
    _active_clients.clear()


def _find_config(cwd: Path) -> dict | None:
    """按优先级查找 .mcp.json：项目目录 → 便携目录(exe同级) → 全局配置目录。"""
    from core.config import get_portable_dir
    portable_mcp = get_portable_dir() / "mcp.json"
    for candidate in [cwd / ".mcp.json", portable_mcp, Path.home() / ".config" / "super-code" / "mcp.json"]:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None
