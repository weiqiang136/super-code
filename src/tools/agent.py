from __future__ import annotations

import json

from core.tool import Tool, ToolResult
from features.worker_manager import WorkerManager


class AgentTool(Tool):
    name = "Agent"
    description = (
        "Spawn a background worker for research, implementation, or "
        "verification. Returns immediately with a task_id. Final results "
        "arrive later as a <task-notification> user message."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short label for the worker task"},
            "prompt": {"type": "string", "description": "Self-contained instructions for the worker"},
            "subagent_type": {
                "type": "string",
                "enum": ["worker"],
                "default": "worker",
                "description": "Only 'worker' is currently supported",
            },
        },
        "required": ["description", "prompt"],
    }

    def get_activity_description(self, **kwargs) -> str | None:
        desc = kwargs.get("description", "")
        return f"Running agent: {desc}" if desc else "Running agent…"

    def __init__(self, manager: WorkerManager):
        self._manager = manager

    def execute(self, description: str = "", prompt: str = "",
                subagent_type: str = "worker", **_) -> ToolResult:
        # 长 prompt（多 worker 并行、单 prompt 数百字）走 OpenAI 兼容流式协议时，
        # tool_call.arguments 的 JSON 字符串可能在传输中被截断，llm.py 解析失败
        # 静默退化为 {} → 这里收到空 description/prompt。直接调用会让 Python 抛
        # "missing required positional arguments"，TUI 显示为 Tool error，
        # 用户和模型都不知道是 JSON 截断导致的。改为返回明确 is_error tool_result，
        # 模型读到提示后会重试一次（多数情况第二次能成）。
        if not description or not prompt:
            return ToolResult(
                content=(
                    "Error: Agent tool requires both 'description' (non-empty) and "
                    "'prompt' (non-empty). The previous call had empty/missing input "
                    "— this usually means the tool_call arguments JSON was truncated "
                    "in transit. Retry with a complete JSON input. If the prompt is "
                    "very long, consider splitting the task into smaller workers."
                ),
                is_error=True,
            )
        try:
            payload = self._manager.spawn(
                description=description, prompt=prompt, subagent_type=subagent_type)
        except ValueError as exc:
            return ToolResult(content=f"Error: {exc}", is_error=True)
        return ToolResult(content=json.dumps(payload, ensure_ascii=False))


class SendMessageTool(Tool):
    name = "SendMessage"
    description = (
        "Continue an existing idle worker by task_id. Use this after a worker "
        "has already reported back and you want it to take another step."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Worker task id to continue"},
            "message": {"type": "string", "description": "Next self-contained instruction"},
        },
        "required": ["to", "message"],
    }

    def __init__(self, manager: WorkerManager):
        self._manager = manager

    def execute(self, to: str = "", message: str = "", **_) -> ToolResult:
        # 同 AgentTool.execute：JSON 截断时 to/message 可能为空，返回明确错误让模型重试
        # 而不是 Python 抛 missing positional arguments。
        if not to or not message:
            return ToolResult(
                content=(
                    "Error: SendMessage requires both 'to' (worker task_id) and "
                    "'message' (non-empty). The previous call had empty/missing input "
                    "— likely a tool_call arguments JSON truncation. Retry with a "
                    "complete JSON input."
                ),
                is_error=True,
            )
        try:
            payload = self._manager.continue_task(task_id=to, message=message)
        except ValueError as exc:
            return ToolResult(content=f"Error: {exc}", is_error=True)
        return ToolResult(content=json.dumps(payload, ensure_ascii=False))


class TaskStopTool(Tool):
    name = "TaskStop"
    description = "Stop a running worker by task_id."
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Worker task id"},
        },
        "required": ["task_id"],
    }

    def __init__(self, manager: WorkerManager):
        self._manager = manager

    def execute(self, task_id: str = "", **_) -> ToolResult:
        # 同上：JSON 截断时 task_id 可能为空，返回明确错误而不是 Python 异常。
        if not task_id:
            return ToolResult(
                content=(
                    "Error: TaskStop requires 'task_id'. The previous call had "
                    "empty/missing input — likely a tool_call arguments JSON "
                    "truncation. Retry with the worker's task_id."
                ),
                is_error=True,
            )
        try:
            payload = self._manager.stop_task(task_id=task_id)
        except ValueError as exc:
            return ToolResult(content=f"Error: {exc}", is_error=True)
        return ToolResult(content=json.dumps(payload, ensure_ascii=False))
