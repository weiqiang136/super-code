from __future__ import annotations

import os
import sys
from typing import Literal, TYPE_CHECKING

from core.tool import Tool

if TYPE_CHECKING:
    from tui.keylistener import EscListener

PermissionBehavior = Literal["allow", "deny"]

_PLAN_MODE_ALLOWED_TOOLS = {
    "Read", "Glob", "Grep", "AskUserQuestion",
    "EnterPlanMode", "ExitPlanMode",
    "Agent", "SendMessage", "TaskStop",
}
_PLAN_MODE_WRITE_TOOLS = {"Edit", "Write"}


class PermissionChecker:
    """Read-only tools are auto-allowed. Bash/writes prompt the user (y/n/always)."""

    def __init__(self, auto_approve: bool = False, sandbox_manager=None):
        self._auto_approve = auto_approve                   # 是否自动批准所有工具
        self._sandbox_manager = sandbox_manager              # 沙箱管理器（用于 auto_approve_if_sandboxed）
        self._always_allow: set[str] = set()                # 用户选择“总是允许”的工具集合
        self._esc_listener: EscListener | None = None       # ESC监听器
        self._plan_manager = None                           # 计划管理器
        self._mode: str = "default"                         # 当前模式
        self._pre_plan_mode: str | None = None                  # 进入计划模式之前的模式
        self._pre_plan_always_allow: set[str] | None = None     # 进入计划模式之前的always allow备份
        self._dream_mode: bool = False                          # 是否dream模式
        self._dream_memory_dir: str | None = None               # dream模式允许写入的目录

    def set_plan_manager(self, plan_manager) -> None:
        self._plan_manager = plan_manager

    def enter_dream_mode(self, memory_dir: str) -> None:
        self._dream_mode = True
        self._dream_memory_dir = os.path.realpath(memory_dir)

    def exit_dream_mode(self) -> None:
        self._dream_mode = False
        self._dream_memory_dir = None

    def set_esc_listener(self, listener) -> None:
        self._esc_listener = listener

    @property
    def mode(self) -> str:
        return self._mode

    def enter_plan_mode(self) -> None:
        self._pre_plan_mode = self._mode
        self._pre_plan_always_allow = set(self._always_allow)
        self._mode = "plan"
        self._always_allow -= {"Bash", "Edit", "Write", "Agent"}  # 从always_allow中移除这些工具

    def exit_plan_mode(self) -> None:
        self._mode = self._pre_plan_mode or "default"
        self._pre_plan_mode = None
        if self._pre_plan_always_allow is not None:
            self._always_allow = self._pre_plan_always_allow
            self._pre_plan_always_allow = None

    def check(self, tool: Tool, inputs: dict) -> PermissionBehavior:
        if self._dream_mode:
            return self._check_dream(tool, inputs)
        if self._mode == "plan":
            return self._check_plan(tool, inputs)
        if tool.is_read_only():
            return "allow"
        if self._auto_approve:
            return "allow"
        # 沙箱激活时自动批准 Bash（命令已被黑名单保护）
        if (self._sandbox_manager is not None
                and tool.name == "Bash"
                and self._sandbox_manager._config is not None
                and self._sandbox_manager._config.auto_approve_if_sandboxed):
            return "allow"
        if tool.name in self._always_allow:
            return "allow"
        return self._prompt_user(tool, inputs)

    def _check_plan(self, tool: Tool, inputs: dict) -> PermissionBehavior:
        if tool.name in _PLAN_MODE_ALLOWED_TOOLS:
            return "allow"
        if tool.name in _PLAN_MODE_WRITE_TOOLS:
            file_path = inputs.get("file_path", "")
            plan_path = self._plan_manager.plan_file_path if self._plan_manager else None
            if plan_path and os.path.realpath(file_path) == os.path.realpath(plan_path): # 返回指定路径的规范化绝对路径，并解析所有符号链接（软链接）
                return "allow"
            from rich.console import Console
            Console().print(f"[yellow]Plan mode: can only edit the plan file ({plan_path})[/yellow]")
            return "deny"
        from rich.console import Console
        Console().print(f"[yellow]Plan mode: {tool.name} is not allowed.[/yellow]")
        return "deny"

    def _check_dream(self, tool: Tool, inputs: dict) -> PermissionBehavior:
        if tool.is_read_only():
            return "allow"
        if tool.name in ("Edit", "Write"):
            file_path = inputs.get("file_path", "")
            if (self._dream_memory_dir and isinstance(file_path, str)
                    and os.path.realpath(file_path).startswith(self._dream_memory_dir)):
                return "allow"
        return "deny"

    def _prompt_user(self, tool: Tool, inputs: dict) -> PermissionBehavior:
        from rich.console import Console
        from prompt_toolkit.application import Application as PTApp
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        console = Console()
        console.print(f"\n[bold yellow]Permission required:[/bold yellow] [bold]{tool.name}[/bold]")

        # For Edit tool, show old_string/new_string as a diff in a dashed-border Panel
        if tool.name == "Edit" and "old_string" in inputs and "new_string" in inputs:
            old_str = str(inputs.get("old_string", ""))
            new_str = str(inputs.get("new_string", ""))
            diff_lines = []
            if old_str:
                for line in old_str.splitlines():
                    diff_lines.append(f"[red]- {line}[/red]")
            if new_str:
                for line in new_str.splitlines():
                    diff_lines.append(f"[green]+ {line}[/green]")
            from rich.panel import Panel
            diff_body = "\n".join(diff_lines)
            if len(diff_body) > 500:
                diff_body = diff_body[:500] + "\n..."
            console.print(Panel(diff_body, border_style="dim", padding=(0, 1)))
            for k, v in inputs.items():
                if k in ("old_string", "new_string"):
                    continue
                val = str(v)[:200] + ("..." if len(str(v)) > 200 else "")
                console.print(f"  [dim]{k}:[/dim] {val}")
        else:
            for k, v in inputs.items():
                val = str(v)[:200] + ("..." if len(str(v)) > 200 else "")
                console.print(f"  [dim]{k}:[/dim] {val}")

        if self._esc_listener:
            self._esc_listener.pause()

        # 选项：(返回值, 显示文本)
        options = [("allow", "Yes"), ("deny", "No"), ("always", "Always allow")]
        selected = [0]

        def get_text():
            lines = []
            for i, (_, label) in enumerate(options):
                if i == selected[0]:
                    lines.append(("bold fg:ansigreen", f" \u276f {label}\n"))
                else:
                    lines.append(("", f"   {label}\n"))
            return lines

        kb = KeyBindings()

        @kb.add("up")
        def _(event):
            selected[0] = (selected[0] - 1) % len(options)
            event.app.invalidate()

        @kb.add("down")
        def _(event):
            selected[0] = (selected[0] + 1) % len(options)
            event.app.invalidate()

        @kb.add("enter")
        def _(event):
            event.app.exit(result=options[selected[0]][0])

        @kb.add("c-c")
        def _(event):
            event.app.exit(result="deny")

        try:
            app = PTApp(
                layout=Layout(Window(FormattedTextControl(get_text))),
                key_bindings=kb,
                full_screen=False,
                refresh_interval=None,
            )
            choice = app.run()
        except Exception:
            choice = "deny"
        finally:
            if self._esc_listener:
                self._esc_listener.resume()

        if choice == "always":
            self._always_allow.add(tool.name)
            return "allow"
        if choice == "allow":
            return "allow"
        return "deny"
