from __future__ import annotations

import re
import subprocess
from core.tool import Tool, ToolResult

_DEFAULT_TIMEOUT = 120

# 匹配 Bash 输出重定向的目标路径：> path, >> path, 1> path, 2> path 等
_REDIRECT_RE = re.compile(r'(?:>>|[12]?>>?)\s*([^\s|;&]+)')


class BashTool(Tool):
    name = "Bash"
    description = (
        "Executes a given bash command and returns its output.\n\n"
        "IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, "
        "`sed`, `awk`, or `echo` commands — use dedicated tools instead.\n\n"
        " - File search: use Glob\n - Content search: use Grep\n"
        " - Read files: use Read\n - Edit files: use Edit\n - Write files: use Write"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute"},
            "description": {"type": "string", "description": "What this command does"},
            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 120},
        },
        "required": ["command"],
    }

    def __init__(self, sandbox_manager=None):
        self._sandbox = sandbox_manager

    def get_activity_description(self, **kwargs) -> str | None:
        command = kwargs.get("command", "")
        preview = command[:60] + "…" if len(command) > 60 else command
        return f"Running {preview}" if command else None

    def execute(self, command: str, description: str = "",
                timeout: int = _DEFAULT_TIMEOUT, **kwargs) -> ToolResult:
        # 沙箱启用时先过滤危险命令，通过后再执行
        if self._sandbox is not None:
            allowed, reason = self._sandbox.check(command)
            if not allowed:
                return ToolResult(content=f"Error: {reason}", is_error=True)
            # 检查网络外发域名（curl/wget 等）
            allowed, reason = self._sandbox.check_network(command)
            if not allowed:
                return ToolResult(content=f"Error: {reason}", is_error=True)
            # 检查重定向目标是否指向受保护路径
            for m in _REDIRECT_RE.finditer(command):
                target = m.group(1).strip().strip("'\"")
                if target:
                    allowed, reason = self._sandbox.check_path(target, "write")
                    if not allowed:
                        return ToolResult(content=f"Error: {reason}", is_error=True)
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=timeout,
            )
            parts = []
            if result.stdout:
                stdout = result.stdout.rstrip()
                if len(stdout) > 10_000:
                    stdout = stdout[:10_000] + f"\n\n... (output truncated)"
                parts.append(stdout)
            if result.stderr:
                parts.append(f"[stderr]\n{result.stderr.rstrip()}")
            if result.returncode != 0:
                parts.append(f"[exit code: {result.returncode}]")
            return ToolResult(content="\n".join(parts) if parts else "(no output)")
        except subprocess.TimeoutExpired:
            return ToolResult(content=f"Error: Command timed out after {timeout}s", is_error=True)
        except Exception as e:
            return ToolResult(content=f"Error: {e}", is_error=True)
