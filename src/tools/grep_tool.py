from __future__ import annotations

import re
import subprocess
from pathlib import Path
import glob as glob_module

from core.tool import Tool, ToolResult

# ---------------------------------------------------------------------------
# content 模式长行截断（防 minified 单行打爆上下文，2026-09-14 esigning 事故）
# ---------------------------------------------------------------------------

GREP_LINE_WINDOW_CHARS = 2000        # 每个窗口目标长度（命中点 ±1000）
GREP_MAX_WINDOWS_PER_LINE = 3        # 单行最多输出窗口数
GREP_WINDOW_MERGE_GAP = 200          # 窗口间距 ≤ 此值合并为一个
GREP_TOTAL_CHAR_CAP = 32768          # content 模式输出总字符硬顶（对齐 PRUNE_RECENT）
GREP_SPAN_RENDER_LIMIT = 1000        # 单个匹配 span 渲染上限（防贪婪模式吞出巨 span）


def _merge_match_windows(content_len: int,
                         spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """把匹配 span 扩成 ±半窗，相邻窗口间距 ≤ GREP_WINDOW_MERGE_GAP 则合并。

    返回合并后的 [(start, end)]，有序、互不重叠、都在 [0, content_len] 内。
    """
    half = GREP_LINE_WINDOW_CHARS // 2
    windows: list[list[int]] = []
    for start, end in spans:
        if end <= start:
            continue
        ws, we = max(0, start - half), min(content_len, end + half)
        if windows and ws - windows[-1][1] <= GREP_WINDOW_MERGE_GAP:
            windows[-1][1] = max(windows[-1][1], we)
        else:
            windows.append([ws, we])
    return [(ws, we) for ws, we in windows]


_RG_PREFIX_RE = re.compile(r"^(.+?)([:-])(\d+)([:-])")


def _split_rg_prefix(line: str) -> tuple[str, str]:
    """拆 rg --no-heading 输出行的 "path:lineno:" / "path-lineno-" 前缀。

    要求分隔符间必须是数字行号，避免 minified 内容里的 "a:b" 被误拆；
    Windows 盘符 "D:" 后面不是数字，不会被误当分隔符。拆不出（如未开 -n）
    就整行当 content——锚定窗口照常工作，仅少前缀。
    """
    m = _RG_PREFIX_RE.match(line)
    if m:
        return line[:m.end()], line[m.end():]
    return "", line


def _compile_anchor_regex(pattern: str, case_insensitive: bool) -> re.Pattern | None:
    """把 pattern 编译成 Python re 用于行内锚定；rg 方言编译失败返回 None。"""
    try:
        return re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
    except re.error:
        return None


def _truncate_content_line(prefix: str, content: str,
                           regex: re.Pattern | None) -> str:
    """单行 content 截断：以匹配位置为锚点开窗，多命中合并/限额，头部兜底。

    - 短行（≤ 2000 字符）原样返回；
    - regex 为 None 或行内无匹配 → 头部截断 + 「命中位置未知」标注；
    - 命中窗口数 > GREP_MAX_WINDOWS_PER_LINE → 只显示前 3 个 + 剩余数与
      逃生路径（窄 pattern / count 模式）；
    - 匹配 span 超过 GREP_SPAN_RENDER_LIMIT（贪婪模式如 [a-z]+ 吞掉整段
      小写填充串）→ 只渲染前 1000 字符 + 标注真实 span 长度。
    """
    total = len(content)
    spans: list[tuple[int, int]] = []
    if regex is not None:
        spans = [m.span() for m in regex.finditer(content) if m.end() > m.start()]

    if not spans:
        if total <= GREP_LINE_WINDOW_CHARS:
            return prefix + content
        return (f"{prefix}{content[:GREP_LINE_WINDOW_CHARS]}"
                f"\n... [line truncated: total {total:,} chars, head only — "
                f"match position unknown, use a narrower pattern]")

    # 病态模式防护：span 本身巨长时窗口 = span ±1000 会失控（2026-09-15 实测
    # api/[a-z]+ 单窗 52K）。超限 span 只渲染前 GREP_SPAN_RENDER_LIMIT 字符。
    clamped = 0
    max_span = 0
    eff_spans: list[tuple[int, int]] = []
    for s, e in spans:
        max_span = max(max_span, e - s)
        if e - s > GREP_SPAN_RENDER_LIMIT:
            clamped += 1
            eff_spans.append((s, s + GREP_SPAN_RENDER_LIMIT))
        else:
            eff_spans.append((s, e))
    spans = eff_spans

    windows = _merge_match_windows(total, spans)
    shown = windows[:GREP_MAX_WINDOWS_PER_LINE]
    rendered = []
    for ws, we in shown:
        body = content[ws:we]
        lead = "" if ws == 0 else "…"
        tail = "" if we == total else "…"
        rendered.append(f"{prefix}{lead}{body}{tail}")
    hidden_matches = sum(1 for s, e in spans
                         if not any(ws <= s and e <= we for ws, we in shown))
    note = (f"... [line truncated: total {total:,} chars, {len(spans)} match(es), "
            f"showing {len(shown)} window(s)")
    if hidden_matches:
        note += (f", {hidden_matches} match(es) not shown — "
                 f'use a narrower pattern or output_mode="count"')
    if clamped:
        note += (f", {clamped} match(es) over {GREP_SPAN_RENDER_LIMIT:,} chars "
                 f"(longest {max_span:,}) rendered partially")
    note += "]"
    return "\n".join(rendered) + "\n" + note


def _enforce_total_cap(parts: list[str], cap: int = GREP_TOTAL_CHAR_CAP) -> str:
    """content 模式输出总字符硬顶，超限丢行并标注剩余行数。

    单个 part 自身超顶（如单行文件聚合出的巨窗）→ 硬切到 cap——
    闸门对任何输入都必须关死，不做无条件放行（2026-09-15 穿透回归）。
    """
    out: list[str] = []
    used = 0
    dropped = 0
    for p in parts:
        cost = len(p) + 1
        if out and used + cost > cap:
            dropped += 1
            continue
        if cost > cap:
            p = p[:cap] + f"\n... [part hard-truncated at {cap:,} chars]"
            cost = len(p) + 1
        out.append(p)
        used += cost
    text = "\n".join(out)
    if dropped:
        text += (f"\n\n... [output truncated at {cap:,} chars, "
                 f"{dropped} more line(s) withheld — narrow the pattern]")
    return text


class GrepTool(Tool):
    name = "Grep"
    description = (
        "Powerful search tool based on ripgrep.\n\n"
        "- Supports full regex syntax\n"
        "- Filter files with glob or type parameter\n"
        "- Output modes: \"content\" (lines), \"files_with_matches\" (paths, default), \"count\"\n"
        "- Long lines (e.g. minified bundles) are windowed around each match "
        "(~2000 chars, max 3 windows/line, 32K total) instead of returned whole\n"
        "- Falls back to Python regex if rg is not installed"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern"},
            "path": {"type": "string", "description": "Directory or file to search"},
            "glob": {"type": "string", "description": "File glob filter e.g. '*.py'"},
            "type": {"type": "string", "description": "File type filter (e.g. 'py', 'js')"},
            "output_mode": {
                "type": "string",
                "enum": ["files_with_matches", "content", "count"],
                "default": "files_with_matches",
            },
            "-i": {"type": "boolean", "description": "Case insensitive", "default": False},
            "-n": {"type": "boolean", "description": "Show line numbers", "default": True},
            "-A": {"type": "integer", "description": "Lines after each match"},
            "-B": {"type": "integer", "description": "Lines before each match"},
            "-C": {"type": "integer", "description": "Context lines around each match"},
            "multiline": {"type": "boolean", "description": "Enable multiline mode", "default": False},
            "head_limit": {"type": "integer", "description": "Limit output to first N lines", "default": 250},
            "offset": {"type": "integer", "description": "Skip first N lines", "default": 0},
        },
        "required": ["pattern"],
    }

    def is_read_only(self) -> bool:
        return True

    def get_activity_description(self, **kwargs) -> str | None:
        pattern = kwargs.get("pattern", "")
        return f"Searching for {pattern}" if pattern else None

    def execute(self, pattern: str, path: str = ".", glob: str | None = None,
                output_mode: str = "files_with_matches", **kwargs) -> ToolResult:
        cmd = ["rg", "--no-heading"]
        if kwargs.get("-i"):
            cmd.append("-i")
        if kwargs.get("multiline"):
            cmd.extend(["-U", "--multiline-dotall"])
        after = kwargs.get("-A")
        before = kwargs.get("-B")
        context = kwargs.get("-C")
        if after and output_mode == "content":
            cmd.extend(["-A", str(after)])
        if before and output_mode == "content":
            cmd.extend(["-B", str(before)])
        if context and output_mode == "content":
            cmd.extend(["-C", str(context)])
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")
        else:
            if kwargs.get("-n", True):
                cmd.append("-n")
        if glob:
            cmd.extend(["-g", glob])
        file_type = kwargs.get("type")
        if file_type:
            cmd.extend(["--type", file_type])
        cmd.extend([pattern, path])

        head_limit = kwargs.get("head_limit", 250) or 250
        offset = kwargs.get("offset", 0) or 0

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=30)
            output = result.stdout.strip()
            if not output:
                return ToolResult(content="No matches found.")
            lines = output.split("\n")
            if offset > 0:
                lines = lines[offset:]
            truncated = head_limit > 0 and len(lines) > head_limit
            if head_limit > 0:
                lines = lines[:head_limit]
            if output_mode == "content":
                # 长行锚定截断：防 minified 单行全量注入（head_limit 数行不数字符）
                regex = _compile_anchor_regex(pattern, bool(kwargs.get("-i")))
                parts = []
                for ln in lines:
                    prefix, content = _split_rg_prefix(ln)
                    parts.append(_truncate_content_line(prefix, content, regex))
                result_text = _enforce_total_cap(parts)
            else:
                result_text = "\n".join(lines)
            if truncated:
                result_text += f"\n\n... (results truncated, showing {head_limit} entries)"
            return ToolResult(content=result_text)
        except FileNotFoundError:
            return self._python_grep(pattern, path, glob, kwargs.get("-i", False), output_mode)
        except subprocess.TimeoutExpired:
            return ToolResult(content="Error: Search timed out.", is_error=True)

    def _python_grep(self, pattern: str, path: str, glob_filter: str | None,
                     case_insensitive: bool, output_mode: str = "files_with_matches") -> ToolResult:
        base = Path(path)
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern, flags)

        if base.is_file():
            files = [base]
        else:
            pat = glob_filter or "**/*"
            files = [base / p for p in glob_module.glob(pat, root_dir=str(base), recursive=True)]

        matched = []
        for f in files:
            if not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                if output_mode == "content":
                    for lineno, line in enumerate(text.splitlines(), 1):
                        if regex.search(line):
                            matched.append(
                                _truncate_content_line(f"{f}:{lineno}:", line, regex))
                else:
                    if regex.search(text):
                        matched.append(str(f))
            except OSError:
                pass

        if output_mode == "content":
            return ToolResult(content=_enforce_total_cap(matched)
                              if matched else "No matches found.")
        return ToolResult(content="\n".join(matched) if matched else "No matches found.")
