"""EnterPlanMode and ExitPlanMode tools."""
from __future__ import annotations

from typing import TYPE_CHECKING
from core.tool import Tool, ToolResult

if TYPE_CHECKING:
    from features.plan import PlanModeManager


class EnterPlanModeTool(Tool):
    name = "EnterPlanMode"
    description = (
        "Use this tool proactively when you're about to start a non-trivial implementation task. "
        "Transitions you into plan mode where you can explore the codebase and design an "
        "implementation approach for user approval. In plan mode you may ONLY read files and "
        "write to the plan file — no other modifications are allowed."
    )
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, plan_manager: PlanModeManager) -> None:
        self._plan_manager = plan_manager

    def is_read_only(self) -> bool:
        return True

    def get_activity_description(self, **kwargs) -> str | None:
        return "Entering plan mode…"

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(content=self._plan_manager.enter())


class ExitPlanModeTool(Tool):
    name = "ExitPlanMode"
    description = (
        "Use this tool ONLY when you have finished writing your plan to the plan file "
        "and want to notify the user that the plan is ready for review. "
        "IMPORTANT: This tool does NOT exit plan mode by itself. The user must manually "
        "press Shift+Tab after reviewing the plan to actually exit and start implementation. "
        "After calling this tool, STOP and wait for the user — do not call any more tools. "
        "If the user wants changes, refine the plan file and call this tool again when ready."
    )
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, plan_manager: PlanModeManager) -> None:
        self._plan_manager = plan_manager

    def get_activity_description(self, **kwargs) -> str | None:
        return "Awaiting user approval…"

    def execute(self, **kwargs) -> ToolResult:
    # 信号语义工具：不改变任何 engine 状态（不切换工具集、不改 system_prompt、
        # 不动 _messages），仅在终端打印通知，让用户知道 plan 已就绪。
        #
        # 真正的模式切换由用户按 Shift+Tab 触发（参见 tui/app.py:_toggle_plan_mode）。
        # 这样设计的核心动机：让"模式切换"始终是用户的显式操作，杜绝 LLM 通过工具
        # 自主修改 engine 配置而引发的一系列时序/历史一致性问题
        # （参见 commit ee0fbc7、5a7d478、4333ff8 修复过的 bug：API 400、spinner 卡死等）。
        from rich.console import Console
        Console().print(
            "\n[bold green]✓ Plan is ready for your review.[/bold green]\n"
            "[dim]Press Shift+Tab to exit plan mode and start implementation, "
            "or send a message to refine the plan.[/dim]\n"
        )
        plan_path = self._plan_manager.plan_file_path or "the plan file"
        # 返回给 LLM 的内容必须明确：本工具调用不等于已退出，必须停下等待用户。
        # 避免 LLM 误以为已在 normal 模式而尝试调用写工具（写工具仍被权限层拦截，
        # 但反复尝试会污染对话历史，降低体验）。
        return ToolResult(content=(
            f"The user has been notified that the plan ({plan_path}) is ready for review. "
            "You are STILL in plan mode. Stop now and wait — do NOT call any more tools. "
            "The user will press Shift+Tab to exit plan mode if they approve, "
            "or send a message with feedback if they want changes."
        ))
