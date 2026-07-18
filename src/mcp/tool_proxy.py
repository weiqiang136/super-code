"""MCP Tool Proxy — 将 MCP server 的每个工具包装成项目通用的 Tool 实例。

工具命名规则：mcp__{serverName}__{toolName}
  - 双下划线分隔，避免与本地工具名冲突
  - LLM 看到的就是这个名字，调用时也用这个名字
"""
from __future__ import annotations

from core.tool import Tool, ToolResult
from mcp.client import MCPClient


class MCPToolProxy(Tool):
    """代理单个 MCP 工具，把调用转发给对应的 MCPClient。"""

    def __init__(self, client: MCPClient, tool_def: dict):
        self._client = client
        # 原始工具名（server 内部使用）
        self._tool_name = tool_def["name"]
        # 对外暴露的名字加上 server 前缀，避免冲突
        self._name = f"mcp__{client.name}__{self._tool_name}"
        self._description = tool_def.get("description", "")
        # inputSchema 直接透传 MCP server 返回的 JSON Schema
        self._input_schema = tool_def.get("inputSchema", {"type": "object", "properties": {}})

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> dict:
        return self._input_schema

    def is_read_only(self) -> bool:
        # MCP 工具默认需要权限确认，保守处理
        return False

    def get_activity_description(self, **kwargs) -> str | None:
        return f"MCP {self._client.name}: {self._tool_name}"

    def execute(self, **kwargs) -> ToolResult:
        try:
            result = self._client.call_tool(self._tool_name, kwargs)
            return ToolResult(content=result)
        except Exception as e:
            return ToolResult(content=f"MCP error: {e}", is_error=True)


def build_mcp_tools(client: MCPClient) -> list[MCPToolProxy]:
    """从一个 MCP server 获取所有工具并返回代理列表。"""
    try:
        tool_defs = client.list_tools()
    except Exception as e:
        raise RuntimeError(f"Failed to list tools from MCP server '{client.name}': {e}") from e
    return [MCPToolProxy(client, td) for td in tool_defs]
