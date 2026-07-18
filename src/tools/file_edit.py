"""Edit tool — scoped string replacement with snippet_id safety (Phase 2)."""
from __future__ import annotations

from pathlib import Path
from core.tool import Tool, ToolResult
from core.file_state import (
    get_snippet,
    get_file_state,
    is_snippet_stale,
    record_file_state,
    create_snippet,
)


class FileEditTool(Tool):
    name = "Edit"
    description = (
        "Replaces exact string matches in files.\n\n"
        "Usage:\n"
        "- You must use Read at least once before editing. "
        "This tool will fail if you attempt an edit without having read the file.\n"
        "- The edit will FAIL if old_string is not unique in the file. "
        "Either provide a larger string with more surrounding context or use replace_all.\n"
        "- Use replace_all to replace and rename strings across the entire file."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "snippet_id": {
                "type": "string",
                "description": (
                    "Required: snippet_id from a previous Read call. "
                    "The edit is scoped to the lines covered by that snippet."
                ),
            },
            "file_path": {
                "type": "string",
                "description": "Optional absolute path guard; must match snippet_id's file.",
            },
            "old_string": {"type": "string", "description": "Exact string to replace"},
            "new_string": {"type": "string", "description": "Replacement string (must differ from old_string)"},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)", "default": False},
        },
        "required": ["snippet_id", "old_string", "new_string"],
    }

    # Shared set of files that have been read — populated by FileReadTool
    _read_files: set[str] = set()

    @classmethod
    def mark_file_read(cls, file_path: str) -> None:
        cls._read_files.add(file_path)

    def __init__(self, sandbox_manager=None, session_id: str = ""):
        self._sandbox = sandbox_manager
        self._session_id = session_id

    def set_session_id(self, session_id: str) -> None:
        """Phase 3: engine 注入当前会话 ID。"""
        self._session_id = session_id

    def get_activity_description(self, **kwargs) -> str | None:
        fp = kwargs.get("file_path", "") or kwargs.get("snippet_id", "")
        return f"Editing {fp}" if fp else None

    def execute(self, snippet_id: str, old_string: str, new_string: str,
                file_path: str = "", replace_all: bool = False, **kwargs) -> ToolResult:
        # ── 1. snippet 校验 ──────────────────────────────────────────────
        if not snippet_id.strip():
            return ToolResult(
                content="Error: snippet_id is required. Use Read first to get a snippet_id.",
                is_error=True,
            )

        snippet = get_snippet(self._session_id, snippet_id) if self._session_id else None
        if snippet is None:
            return ToolResult(
                content=f"Error: Unknown snippet_id: {snippet_id}. The snippet may have expired "
                        f"or you may need to Read the file again to get a fresh snippet_id.",
                is_error=True,
            )

        if is_snippet_stale(self._session_id, snippet):
            return ToolResult(
                content=f"Error: The file {snippet.file_path} has been modified since snippet "
                        f"'{snippet_id}' was created. Read the file again to get a fresh snippet_id.",
                is_error=True,
            )

        # ── 2. file_path 校验 既允许用户不传路径（使用凭证中的路径），又防止用户用凭证去编辑其他文件（路径必须匹配）
        fp = file_path.strip() if file_path else snippet.file_path
        if not fp:
            return ToolResult(content="Error: file_path is required.", is_error=True)
        if file_path.strip() and Path(file_path.strip()).resolve() != Path(snippet.file_path):
            return ToolResult(
                content=f"Error: snippet_id '{snippet_id}' belongs to {snippet.file_path}, "
                        f"not {file_path}.",
                is_error=True,
            )

        # ── 3. 沙箱 & 文件存在性 ──────────────────────────────────────────
        if self._sandbox is not None:
            allowed, reason = self._sandbox.check_path(fp, "write")
            if not allowed:
                return ToolResult(content=f"Error: {reason}", is_error=True)

        path = Path(fp)
        if not path.exists():
            return ToolResult(content=f"Error: File not found: {fp}", is_error=True)
        if path.is_dir():
            return ToolResult(content=f"Error: {fp} is a directory.", is_error=True)

        # ── 4. 基本参数校验 ──────────────────────────────────────────────
        if not old_string:
            return ToolResult(
                content="Error: old_string cannot be empty.",
                is_error=True,
            )
        if old_string == new_string:
            return ToolResult(
                content="Error: new_string must differ from old_string.",
                is_error=True,
            )

        # ── 5. 读取文件 & 范围搜索 ───────────────────────────────────────
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            return ToolResult(content=f"Error reading file: {e}", is_error=True)

        lines = content.splitlines(keepends=True)
        total_lines = len(lines) if lines else 1

        # 搜索范围：snippet 行范围
        scope_start = snippet.start_line
        scope_end = min(snippet.end_line, total_lines)
        scope_text = "".join(lines[scope_start - 1:scope_end])

        # 在 scope 内搜索匹配
        match_positions = _find_all(scope_text, old_string)

        # ── 6. 匹配结果处理 ──────────────────────────────────────────────
        # 先检查文件是否被外部程序修改过（mtime 变化），给出具体原因提示；无论哪种情况，都要求 AI 重新读取文件获取新 snippet。
        if len(match_positions) == 0:
            # 试试：文件外部被修改了？
            try:
                stat = path.stat()
                state = get_file_state(self._session_id, fp)
                if state and stat.st_mtime != state.mtime:
                    return ToolResult(
                        content=f"Error: old_string not found in {fp}. The file has been modified "
                                f"externally since it was read. Read it again to get a fresh snippet_id.\n"
                                f"  snippet scope: lines {scope_start}-{scope_end}",
                        is_error=True,
                        metadata={"scope": {"file_path": fp, "start_line": scope_start,
                                              "end_line": scope_end, "snippet_id": snippet_id}},
                    )
            except Exception:
                pass

            return ToolResult(
                content=f"Error: old_string not found in {fp} within the snippet scope "
                        f"(lines {scope_start}-{scope_end}). "
                        f"Read the file again if the content has changed.",
                is_error=True,
                metadata={"scope": {"file_path": fp, "start_line": scope_start,
                                      "end_line": scope_end, "snippet_id": snippet_id}},
            )
        # 在 snippet 范围内找到多个匹配且未启用 replace_all 时，拒绝执行，防止模糊替换改错地方。
        if not replace_all and len(match_positions) > 1:
            # 非唯一匹配 → 返回候选片段
            candidates = []
            for idx, pos in enumerate(match_positions[:5]):  # 最多 5 个
                match_line = _offset_to_line(lines, scope_start, pos)
                preview_start = max(1, match_line - 1)
                preview_end = min(total_lines, match_line + 2)
                preview = "".join(
                    f"{ln}\t{lines[ln - 1]}" for ln in range(preview_start, preview_end + 1)
                )
                candidates.append({
                    "index": idx + 1,
                    "line": match_line,
                    "preview": preview,
                })

            return ToolResult(
                content=f"Error: old_string is not unique within snippet scope "
                        f"(lines {scope_start}-{scope_end}); found {len(match_positions)} matches. "
                        f"Use replace_all=true or provide more surrounding context.",
                is_error=True,
                metadata={
                    "match_count": len(match_positions),
                    "scope": {"file_path": fp, "start_line": scope_start,
                               "end_line": scope_end, "snippet_id": snippet_id},
                    "candidates": candidates,
                },
            )

        # ── 7. 执行替换 ──────────────────────────────────────────────────
        # 通过累计行长度计算出替换位置在文件中的全局字节偏移（scope_bytes_before 是范围前的字节数，global_pos 是具体匹配位置的全局偏移
        scope_bytes_before = sum(len(line) for line in lines[:scope_start - 1])
        if replace_all:
            new_scope = scope_text.replace(old_string, new_string)
            scope_bytes = sum(len(line) for line in lines[scope_start - 1:scope_end])
            new_content = content[:scope_bytes_before] + new_scope + content[scope_bytes_before + scope_bytes:]
            replaced = len(match_positions)
        else:
            pos = match_positions[0]
            # 计算全局偏移
            scope_bytes_before = sum(len(line) for line in lines[:scope_start - 1])
            global_pos = scope_bytes_before + pos
            new_content = content[:global_pos] + new_string + content[global_pos + len(old_string):]
            replaced = 1

        try:
            path.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return ToolResult(content=f"Error writing file: {e}", is_error=True)

        # ── 8. 刷新文件状态 ──────────────────────────────────────────────
        # 编辑后更新文件状态并自增版本号（bump_version=True），使所有旧 snippet 自动失效。基于修改后的完整文件内容重新生成一个 full 类型的 snippet，让 AI 可以继续编辑（同时返回 new_snippet_id）。
        if self._session_id:
            stat = path.stat()
            record_file_state(self._session_id, fp, new_content, stat.st_mtime, bump_version=True)
            # 用替换后的内容计算行数，避免多行替换导致新 snippet 的 end_line 偏小
            new_total_lines = len(new_content.splitlines(keepends=True)) or 1
            new_snippet = create_snippet(self._session_id, fp, 1, new_total_lines, scope_type="full")
            meta = {
                "file_path": fp,
                "replaced_count": replaced,
                "scope": {"file_path": fp, "start_line": scope_start,
                           "end_line": scope_end, "snippet_id": snippet_id},
                "new_snippet_id": new_snippet.id,
            }
        else:
            meta = None

        return ToolResult(
            content=f"Successfully replaced {replaced} occurrence(s) in {fp}.",
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _find_all(text: str, needle: str) -> list[int]:
    """返回 needle 在 text 中所有出现位置的偏移量列表。"""
    positions = []
    start = 0
    while True:
        pos = text.find(needle, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + len(needle)
    return positions


def _offset_to_line(lines: list[str], scope_start_line: int, scope_offset: int) -> int:
    """将 scope 内的偏移量转换为文件行号（1-based）。"""
    remaining = scope_offset
    for i in range(scope_start_line - 1, len(lines)):
        remaining -= len(lines[i])
        if remaining < 0:
            return i + 1
    return len(lines)
