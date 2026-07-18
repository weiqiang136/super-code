"""
test_memory_scan.py — 验证 scan_memory_files / format_memory_manifest 行为。

覆盖：
1. 不存在的目录 → 空列表（不抛异常）
2. MEMORY.md 被排除（即使带 frontmatter）
3. logs/YYYY/MM/*.md 被排除（KAIROS 日志）
4. 完整 frontmatter 文件 → description / type 正确解析
5. 无 frontmatter 的旧文件 → 仍返回，description / type 为 None
6. 多个文件 → 按 mtime 倒序
7. 未知 type 字段 → type 字段为 None（不丢弃整条）
8. format_memory_manifest 输出格式（带 / 不带 type 与 description）

为什么 mtime 用 os.utime 而不是 sleep：
    确定性 + 快。给文件强制设定 mtime 后顺序行为完全可预期。
"""
import os
from pathlib import Path

from features.memory_scan import (
    MAX_MEMORY_FILES,
    MemoryHeader,
    format_memory_manifest,
    scan_memory_files,
)


def _write(path: Path, text: str, mtime: float | None = None) -> None:
    """工具：写文件并按需打 mtime。父目录自动创建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_missing_dir_returns_empty(tmp_path: Path):
    # 目录不存在不应抛异常（首次启动场景）
    assert scan_memory_files(tmp_path / "does_not_exist") == []


def test_skip_memory_md(tmp_path: Path):
    """MEMORY.md 即使带 frontmatter 也必须被排除（它是索引而非记忆）。"""
    _write(
        tmp_path / "MEMORY.md",
        "---\ntype: user\ndescription: should not appear\n---\nbody\n",
    )
    _write(
        tmp_path / "real_mem.md",
        "---\ntype: user\ndescription: real one\n---\nbody\n",
    )

    headers = scan_memory_files(tmp_path)
    filenames = {h.filename for h in headers}
    assert "MEMORY.md" not in filenames
    assert "real_mem.md" in filenames


def test_skip_logs_subtree(tmp_path: Path):
    """logs/ 子树（KAIROS 日志）必须被整体排除。"""
    _write(tmp_path / "logs" / "2026" / "06" / "2026-06-10.md", "- [10:00] hi\n")
    _write(
        tmp_path / "topic.md",
        "---\ntype: project\ndescription: topic file\n---\nbody\n",
    )

    headers = scan_memory_files(tmp_path)
    filenames = {h.filename for h in headers}
    # POSIX 风格相对路径
    assert "topic.md" in filenames
    assert not any(f.startswith("logs/") for f in filenames)


def test_parses_frontmatter_fields(tmp_path: Path):
    _write(
        tmp_path / "feedback_pytest.md",
        "---\nname: feedback-pytest\ndescription: use pytest, not unittest\ntype: feedback\n---\nbody text\n",
    )

    headers = scan_memory_files(tmp_path)
    assert len(headers) == 1
    h = headers[0]
    assert h.filename == "feedback_pytest.md"
    assert h.description == "use pytest, not unittest"
    assert h.type == "feedback"
    assert h.mtime_ms > 0


def test_no_frontmatter_graceful(tmp_path: Path):
    """旧记忆没有 frontmatter，也必须出现在结果中（type / description = None）。"""
    _write(tmp_path / "legacy.md", "just plain markdown, no yaml header\n")

    headers = scan_memory_files(tmp_path)
    assert len(headers) == 1
    assert headers[0].filename == "legacy.md"
    assert headers[0].description is None
    assert headers[0].type is None


def test_sort_by_mtime_desc(tmp_path: Path):
    """三个文件设不同 mtime → 结果严格按 mtime 倒序。"""
    _write(tmp_path / "old.md", "old\n", mtime=1_000_000.0)
    _write(tmp_path / "mid.md", "mid\n", mtime=2_000_000.0)
    _write(tmp_path / "new.md", "new\n", mtime=3_000_000.0)

    headers = scan_memory_files(tmp_path)
    assert [h.filename for h in headers] == ["new.md", "mid.md", "old.md"]


def test_unknown_type_becomes_none(tmp_path: Path):
    """type 字段非法不应丢弃记忆，只把 type 字段降级为 None。"""
    _write(
        tmp_path / "weird.md",
        "---\ntype: bogus\ndescription: still has desc\n---\nbody\n",
    )

    headers = scan_memory_files(tmp_path)
    assert len(headers) == 1
    assert headers[0].type is None
    assert headers[0].description == "still has desc"


def test_recursive_scan_finds_nested(tmp_path: Path):
    """rglob 应找到子目录内的 .md（除 logs/ 之外）。"""
    _write(
        tmp_path / "sub" / "deep.md",
        "---\ntype: project\ndescription: nested\n---\n",
    )
    headers = scan_memory_files(tmp_path)
    assert len(headers) == 1
    # POSIX 路径，跨平台稳定
    assert headers[0].filename == "sub/deep.md"


def test_max_files_cap(tmp_path: Path, monkeypatch):
    """超过上限时按 mtime 倒序截断。"""
    # 把上限调小避免造文件过多
    monkeypatch.setattr("features.memory_scan.MAX_MEMORY_FILES", 3)

    for i in range(5):
        _write(tmp_path / f"m{i}.md", f"#{i}\n", mtime=1_000_000.0 + i)

    headers = scan_memory_files(tmp_path)
    assert len(headers) == 3
    # mtime 最大的三个（i=4,3,2）
    assert [h.filename for h in headers] == ["m4.md", "m3.md", "m2.md"]


def test_format_manifest_with_type_and_description():
    h = MemoryHeader(
        filename="feedback_pytest.md",
        file_path=Path("/tmp/feedback_pytest.md"),
        # 1700000000000 ms = 2023-11-14T22:13:20+00:00
        mtime_ms=1_700_000_000_000.0,
        description="use pytest, not unittest",
        type="feedback",
    )
    out = format_memory_manifest([h])
    assert out == "- [feedback] feedback_pytest.md (2023-11-14T22:13:20+00:00): use pytest, not unittest"


def test_format_manifest_without_type_and_description():
    h = MemoryHeader(
        filename="legacy.md",
        file_path=Path("/tmp/legacy.md"),
        mtime_ms=1_700_000_000_000.0,
        description=None,
        type=None,
    )
    out = format_memory_manifest([h])
    # 无 type → 无前缀；无 description → 无尾冒号
    assert out == "- legacy.md (2023-11-14T22:13:20+00:00)"


def test_format_manifest_empty_list():
    assert format_memory_manifest([]) == ""


def test_module_default_cap_is_200():
    """避免后续无意把上限改小（接口契约）。"""
    assert MAX_MEMORY_FILES == 200
