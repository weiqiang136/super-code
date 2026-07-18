"""
test_memory_truncate.py — Step 3 MEMORY.md 双重截断 + WARNING 测试。

覆盖：
1. 空字符串 → 空内容、无 WARNING、各计数为 0
2. 未超 cap → 原样返回（仅 strip）、无 WARNING
3. 仅超行数 cap → 截到 200 行 + 行数原因 WARNING
4. 仅超字节 cap（行少但行很长）→ 在最后一个 \\n 处下刀 + 字节原因 WARNING
5. 双超（行多且总字节超）→ WARNING 文案包含双原因
6. load_memory_index：
   - 文件不存在 → 空字符串（与旧行为一致）
   - 文件正常 → 调用 truncate_entrypoint_content 后的 content
   - 字符总数小于旧 MAX_MEMORY_INDEX_CHARS 但行数超 200 → 必须被截断（验证新阈值生效）
"""
from pathlib import Path

from features.memory import (
    ENTRYPOINT_NAME,
    MAX_ENTRYPOINT_BYTES,
    MAX_ENTRYPOINT_LINES,
    EntrypointTruncation,
    load_memory_index,
    truncate_entrypoint_content,
)


# ---------------------------------------------------------------------------
# truncate_entrypoint_content
# ---------------------------------------------------------------------------

def test_empty_string():
    r = truncate_entrypoint_content("")
    assert isinstance(r, EntrypointTruncation)
    assert r.content == ""
    assert r.line_count == 0
    assert r.byte_count == 0
    assert r.was_line_truncated is False
    assert r.was_byte_truncated is False


def test_under_cap_unchanged():
    raw = "- [Title](file.md) — hook\n- [Other](other.md) — hook"
    r = truncate_entrypoint_content(raw)
    # 内容原样返回（外围 strip 不变内文）
    assert r.content == raw
    assert "WARNING" not in r.content
    assert r.line_count == 2
    assert r.was_line_truncated is False
    assert r.was_byte_truncated is False


def test_strips_surrounding_whitespace():
    """前后空白应被 trim；不算"被截断"。"""
    raw = "\n\n- a\n- b\n\n"
    r = truncate_entrypoint_content(raw)
    assert r.content == "- a\n- b"
    assert "WARNING" not in r.content


def test_line_cap_truncation():
    # 250 行，每行只有 'x'，远不超字节 cap
    raw = "\n".join(f"x{i}" for i in range(250))
    r = truncate_entrypoint_content(raw)

    assert r.was_line_truncated is True
    assert r.was_byte_truncated is False
    assert r.line_count == 250
    # 截断到 200 行
    body, _, warning = r.content.partition("\n\n> WARNING:")
    assert body.count("\n") == MAX_ENTRYPOINT_LINES - 1  # 200 行 → 199 个换行
    # WARNING 文案包含触发原因
    assert "250 lines" in warning
    assert f"limit: {MAX_ENTRYPOINT_LINES}" in warning
    # 索引建议提示存在
    assert "index entries" in warning
    assert ENTRYPOINT_NAME in r.content


def test_byte_cap_truncation_with_short_line_count():
    """行数没超但单行非常长 → 应触发字节截断、且在 \\n 处下刀。"""
    # 10 行，每行 5000 字节 → 总 ~50KB，行数 < 200 但字节 > 25KB
    long_line = "y" * 5000
    raw = "\n".join([long_line] * 10)
    r = truncate_entrypoint_content(raw)

    assert r.was_line_truncated is False
    assert r.was_byte_truncated is True
    # 切点必须在某个 \n 处：truncated body 末尾应是 'y'（最后一行的内容），不是半行
    body, sep, warning = r.content.partition("\n\n> WARNING:")
    assert sep  # WARNING 一定被拼上
    # 切完的 body 字节数应不超过 cap
    assert len(body.encode("utf-8")) <= MAX_ENTRYPOINT_BYTES
    # body 内部不应包含半截单行末尾的非 'y'（除非是合法的整行 'y' * 5000）
    # 简化断言：body 应以完整一行结束（即最后一段是若干个 'y'）
    last_segment = body.split("\n")[-1]
    assert last_segment == "" or set(last_segment) <= {"y"}
    # 文案
    assert "bytes" in warning
    assert "index entries are too long" in warning


def test_both_caps_triggered():
    """行多 + 字节多 → 双原因 WARNING。"""
    long_line = "z" * 200  # 单行 200 字节
    # 250 行 × 200 字节 ≈ 50KB，两个 cap 都触发
    raw = "\n".join([long_line] * 250)
    r = truncate_entrypoint_content(raw)

    assert r.was_line_truncated is True
    assert r.was_byte_truncated is True
    _, _, warning = r.content.partition("\n\n> WARNING:")
    assert "lines" in warning and "bytes" in warning


def test_byte_truncation_preserves_multibyte_safe():
    """硬切场景下不应输出非法 UTF-8（errors='ignore' 兜底）。"""
    # 单行超大 + 含多字节字符
    raw = "你好" * 20_000  # 远超 25KB
    r = truncate_entrypoint_content(raw)
    assert r.was_byte_truncated is True
    # 必须能正常解码，且不抛异常
    r.content.encode("utf-8")  # 不抛即可


# ---------------------------------------------------------------------------
# load_memory_index
# ---------------------------------------------------------------------------

def test_load_memory_index_missing_file(tmp_path: Path):
    """文件不存在 → 空字符串（保留旧行为，让上层走 no-memories 分支）。"""
    assert load_memory_index(tmp_path) == ""


def test_load_memory_index_under_cap(tmp_path: Path):
    (tmp_path / ENTRYPOINT_NAME).write_text("- a\n- b\n", encoding="utf-8")
    out = load_memory_index(tmp_path)
    assert out == "- a\n- b"
    assert "WARNING" not in out


def test_load_memory_index_triggers_line_cap(tmp_path: Path):
    """构造 250 行短文件 → 总字符数远小于旧 MAX_MEMORY_INDEX_CHARS (10_000)，
    但新行 cap 必须把它截断 —— 验证新阈值替换了旧的 char cap。"""
    raw = "\n".join(f"r{i}" for i in range(250))  # 约 1KB
    (tmp_path / ENTRYPOINT_NAME).write_text(raw, encoding="utf-8")

    out = load_memory_index(tmp_path)
    assert "WARNING" in out
    assert "250 lines" in out
