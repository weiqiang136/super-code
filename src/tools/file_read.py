"""Simple file read tool implementation."""
import time
from pathlib import Path
from core.tool import Tool, ToolResult
from core.file_state import (
    record_file_state,
    create_snippet,
    clear_session_state as _clear_snippets,
)


class FileReadTool(Tool):
    """A simple tool to read file contents."""

    # Phase C：最近读过的文件 → 最后一次 Read 时间戳。压缩流程取 top-N 重新 Read
    # 注入到压缩后对话，让模型不丢失文件内容上下文。
    # 类级别全局变量：worker engine 与主 engine 共享（已知限制，实际工作目录场景下
    # 通常不构成问题；如未来要严格隔离，再改为实例字段）。time.time() 为系统墙钟，
    # 不随调试或 monkey-patch 改变。
    _recent_reads: dict[str, float] = {}

    @classmethod
    def mark_recent_read(cls, file_path: str) -> None:
        """记录一次成功的 Read 调用（用绝对路径作 key，避免相对路径引发的重复）。"""
        if file_path:
            cls._recent_reads[file_path] = time.time()

    @classmethod
    def get_recent_reads(cls) -> list[tuple[str, float]]:
        """返回 (path, ts) 列表，按 ts 倒序（最近的在前）。压缩重注入用。"""
        return sorted(cls._recent_reads.items(), key=lambda kv: kv[1], reverse=True)

    @classmethod
    def clear_recent_reads(cls) -> None:
        """清空记录（测试或 /clear 命令使用）。"""
        cls._recent_reads.clear()
    
    @property
    def name(self) -> str:
        return "Read"
    
    def __init__(self, sandbox_manager=None, session_id: str = ""):
        self._sandbox = sandbox_manager
        self._session_id = session_id

    def set_session_id(self, session_id: str) -> None:
        """Phase 1/3: engine 注入当前会话 ID，用于创建 snippet。"""
        self._session_id = session_id
    
    @property
    def description(self) -> str:
        return (
            "Reads a file from the local filesystem. "
            "Usage: Provide an absolute file path to read its contents."
        )
    
    @property
    def input_schema(self) -> dict:
        """
            操作大文件的时候看，第一次可以全读，后续如果要修改文件，那么可以使用offset、limit规定只读取某个片段
        :return:
        """
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based). Default: 1."
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to read. Default: read all lines."
                },
            },
            "required": ["file_path"]
        }
    
    def is_read_only(self) -> bool:
        return True
    
    def get_activity_description(self, **kwargs) -> str | None:
        file_path = kwargs.get("file_path", "")
        return f"Reading {file_path}" if file_path else None
    
    def execute(self, file_path: str, **kwargs) -> ToolResult:
        """Execute the file read operation."""
        # 沙箱路径保护：禁止读取受保护目录
        if self._sandbox is not None:
            allowed, reason = self._sandbox.check_path(file_path, "read")
            if not allowed:
                return ToolResult(content=f"Error: {reason}", is_error=True)

        try:
            path = Path(file_path)
            
            if not path.exists():
                return ToolResult(
                    content=f"Error: File not found: {file_path}",
                    is_error=True
                )
            
            if not path.is_file():
                return ToolResult(
                    content=f"Error: Not a file: {file_path}",
                    is_error=True
                )
            
            # Read file content
            content = path.read_text(encoding="utf-8", errors="replace")
            
            # ---- offset / limit 解析 ----
            offset = _parse_positive_int(kwargs.get("offset"), "offset")
            limit = _parse_positive_int(kwargs.get("limit"), "limit")
            if isinstance(offset, str):
                return ToolResult(content=f"Error: {offset}", is_error=True)
            if isinstance(limit, str):
                return ToolResult(content=f"Error: {limit}", is_error=True)

            # Format with line numbers
            all_lines = content.splitlines(keepends=True)
            total_lines = len(all_lines) if all_lines else 1

            # 计算实际行范围
            actual_start = offset if offset else 1
            if limit:
                actual_end = min(actual_start + limit - 1, total_lines)
            else:
                actual_end = total_lines

            # 截取对应行
            selected = all_lines[actual_start - 1:actual_end]
            numbered = "".join(f"{actual_start + i}\t{line}" for i, line in enumerate(selected))

            from tools.file_edit import FileEditTool
            FileEditTool.mark_file_read(file_path)
            FileEditTool.mark_file_read(str(path.resolve()))
            # Phase C：用 resolve 后的绝对路径作 key，避免同一文件因相对/绝对路径
            # 不同被记成两条。失败路径不调用，避免压缩去 Read 一个本来就读不出来的文件。
            try:
                FileReadTool.mark_recent_read(str(path.resolve()))
            except Exception:
                try:
                    FileReadTool.mark_recent_read(file_path)
                except Exception:
                    pass

            # Phase 1: Snippet 系统 — 记录文件状态 + 创建编辑凭证
            snippet_meta = None
            if self._session_id:
                resolved = str(path.resolve())
                stat = path.stat()
                record_file_state(self._session_id, resolved, content, stat.st_mtime)
                # 根据 offset/limit 决定 snippet 的行范围和 scope_type
                is_partial = bool(offset or limit)
                snippet = create_snippet(
                    self._session_id, resolved,
                    start_line=actual_start,
                    end_line=actual_end,
                    scope_type="snippet" if is_partial else "full",
                )
                snippet_meta = {
                    "snippet_id": snippet.id,
                    "file_path": snippet.file_path,
                    "start_line": snippet.start_line,
                    "end_line": snippet.end_line,
                    "scope_type": snippet.scope_type,
                }
                # 将 snippet_id 注入 content 头部，让 LLM 能看到并用于后续 Edit 调用
                header = (
                    f"[snippet_id: {snippet_meta['snippet_id']} | "
                    f"lines: {snippet_meta['start_line']}-{snippet_meta['end_line']} | "
                    f"scope: {snippet_meta['scope_type']}]\n"
                )
                numbered = header + numbered

            return ToolResult(
                content=numbered,
                metadata=snippet_meta,
            )

        except Exception as e:
            return ToolResult(
                content=f"Error reading file: {e}",
                is_error=True
            )


def _parse_positive_int(value, label: str) -> int | None | str:
    """解析正整数参数。返回 None=未传入, int=有效值, str=错误信息。"""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        num = int(value)
    except (ValueError, TypeError):
        return f"{label} must be an integer, got: {value}"
    if num < 1:
        return f"{label} must be >= 1, got: {num}"
    return num
