from __future__ import annotations

from pathlib import Path
from core.tool import Tool, ToolResult
from core.file_state import (
    record_file_state,
    create_snippet,
)


class FileWriteTool(Tool):
    name = "Write"
    description = (
        "Writes a file to the local filesystem.\n\n"
        "Usage:\n"
        "- This tool will overwrite the existing file if one exists at the path.\n"
        "- Prefer the Edit tool for modifying existing files — it only sends the diff. "
        "Only use this tool to create new files or for complete rewrites.\n"
        "- NEVER create documentation files (*.md) or README files unless explicitly requested."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file to write"},
            "content": {"type": "string", "description": "The full content to write to the file"},
        },
        "required": ["file_path", "content"],
    }

    def __init__(self, sandbox_manager=None, session_id: str = ""):
        self._sandbox = sandbox_manager
        self._session_id = session_id

    def set_session_id(self, session_id: str) -> None:
        """Phase 3: engine 注入当前会话 ID。"""
        self._session_id = session_id

    def get_activity_description(self, **kwargs) -> str | None:
        fp = kwargs.get("file_path", "")
        return f"Writing {fp}" if fp else None

    def execute(self, file_path: str, content: str) -> ToolResult:
        # 沙箱路径保护：禁止写入受保护目录
        if self._sandbox is not None:
            allowed, reason = self._sandbox.check_path(file_path, "write")
            if not allowed:
                return ToolResult(content=f"Error: {reason}", is_error=True)

        path = Path(file_path)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            return ToolResult(content=f"Error writing file: {e}", is_error=True)

        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        lines = max(lines, 1)  # 空文件 → 1 行，避免 snippet end_line=0

        # Phase 3: 刷新文件状态 + 创建新 snippet
        meta = None
        if self._session_id:
            resolved = str(path.resolve())
            stat = path.stat()
            record_file_state(self._session_id, resolved, content, stat.st_mtime, bump_version=True)
            snippet = create_snippet(self._session_id, resolved, 1, lines, scope_type="full")
            meta = {
                "file_path": resolved,
                "snippet_id": snippet.id,
                "start_line": 1,
                "end_line": lines,
                "scope_type": "full",
            }

        return ToolResult(
            content=f"Successfully wrote {lines} lines to {file_path}",
            metadata=meta,
        )
