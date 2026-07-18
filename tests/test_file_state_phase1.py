"""Phase 1: Snippet 数据模型 + Read 集成测试。

验证点：
1. record_file_state 首次记录版本号为 1
2. create_snippet 绑定正确版本号
3. is_snippet_stale 版本升级后返回 True
4. 多文件隔离
5. clear_session_state 清空
6. FileReadTool 返回 snippet 元信息
7. Read 输出 content 中包含 snippet_id（LLM 可见）
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from core.file_state import (
    record_file_state,
    create_snippet,
    get_snippet,
    is_snippet_stale,
    get_file_version,
    clear_session_state,
    invalidate_all_snippets,
    FileState,
    FileSnippet,
    _file_states,
    _snippets,
    _file_versions,
)
from tools.file_read import FileReadTool


@pytest.fixture(autouse=True)
def clean_state():
    """每个测试前清空全局状态。"""
    _file_states.clear()
    _snippets.clear()
    _file_versions.clear()
    yield
    _file_states.clear()
    _snippets.clear()
    _file_versions.clear()


# ============================================================================
# 数据模型测试
# ============================================================================


class TestRecordFileState:
    def test_first_record_version_is_1(self):
        record_file_state("s1", "/tmp/a.py", "content", 1000.0)
        assert get_file_version("s1", "/tmp/a.py") == 1

    def test_bump_version_increments(self):
        record_file_state("s1", "/tmp/a.py", "v1", 1000.0)
        assert get_file_version("s1", "/tmp/a.py") == 1
        record_file_state("s1", "/tmp/a.py", "v2", 2000.0, bump_version=True)
        assert get_file_version("s1", "/tmp/a.py") == 2

    def test_session_isolation(self):
        record_file_state("s1", "/tmp/a.py", "x", 1.0)
        record_file_state("s2", "/tmp/a.py", "y", 2.0)
        assert get_file_version("s1", "/tmp/a.py") == 1
        assert get_file_version("s2", "/tmp/a.py") == 1
        # bump s1 only
        record_file_state("s1", "/tmp/a.py", "x2", 3.0, bump_version=True)
        assert get_file_version("s1", "/tmp/a.py") == 2
        assert get_file_version("s2", "/tmp/a.py") == 1


class TestSnippet:
    def test_create_and_retrieve(self):
        record_file_state("s1", "/tmp/b.py", "hello", 1.0)
        s = create_snippet("s1", "/tmp/b.py", 1, 3, scope_type="full")
        assert s.id.startswith("full_")
        assert s.file_version == 1
        assert s.scope_type == "full"
        assert s.start_line == 1
        assert s.end_line == 3

        found = get_snippet("s1", s.id)
        assert found is not None
        assert found.file_path.endswith("b.py")

    def test_full_vs_snippet_id_prefix(self):
        record_file_state("s1", "/tmp/c.py", "data", 1.0)
        full = create_snippet("s1", "/tmp/c.py", 1, 10, scope_type="full")
        snip = create_snippet("s1", "/tmp/c.py", 1, 5, scope_type="snippet")
        assert full.id.startswith("full_")
        assert snip.id.startswith("snp_")

    def test_stale_after_bump(self):
        record_file_state("s1", "/tmp/d.py", "v1", 1.0)
        s = create_snippet("s1", "/tmp/d.py", 1, 1, scope_type="full")
        assert not is_snippet_stale("s1", s)

        record_file_state("s1", "/tmp/d.py", "v2", 2.0, bump_version=True)
        assert is_snippet_stale("s1", s)

    def test_not_stale_same_version(self):
        record_file_state("s1", "/tmp/e.py", "v1", 1.0)
        s = create_snippet("s1", "/tmp/e.py", 1, 1)
        # 再次 record（不 bump），version 不变
        record_file_state("s1", "/tmp/e.py", "v1-again", 1.5)
        assert not is_snippet_stale("s1", s)

    def test_multi_file_isolation(self):
        record_file_state("s1", "/tmp/f.py", "f", 1.0)
        record_file_state("s1", "/tmp/g.py", "g", 1.0)
        sf = create_snippet("s1", "/tmp/f.py", 1, 1)
        sg = create_snippet("s1", "/tmp/g.py", 1, 1)
        assert sf.file_path.endswith("f.py")
        assert sg.file_path.endswith("g.py")

        # bump f only
        record_file_state("s1", "/tmp/f.py", "f2", 2.0, bump_version=True)
        assert is_snippet_stale("s1", sf)
        assert not is_snippet_stale("s1", sg)


class TestClearSession:
    def test_clear_removes_all(self):
        record_file_state("s1", "/tmp/h.py", "h", 1.0)
        create_snippet("s1", "/tmp/h.py", 1, 1)
        assert get_file_version("s1", "/tmp/h.py") == 1

        clear_session_state("s1")
        assert "s1" not in _file_states
        assert "s1" not in _snippets
        assert "s1" not in _file_versions


class TestUnknownSnippet:
    def test_get_nonexistent_snippet(self):
        assert get_snippet("s1", "nonexistent") is None

    def test_stale_check_unknown_file(self):
        s = FileSnippet(id="x", file_path="/tmp/unknown.py",
                        start_line=1, end_line=1, file_version=0, scope_type="full")
        # never recorded → version 0 → snippet file_version 0 → not stale (0 ≤ 0)
        assert not is_snippet_stale("s1", s)


# ============================================================================
# Read 工具集成测试
# ============================================================================


class TestReadToolSnippet:
    """测试 FileReadTool 是否正确生成 snippet 元信息。"""

    def test_read_creates_snippet_metadata(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")

        tool = FileReadTool(session_id="test-session")
        result = tool.execute(file_path=str(f))

        assert not result.is_error
        assert result.metadata is not None
        assert "snippet_id" in result.metadata
        assert result.metadata["scope_type"] == "full"
        assert result.metadata["start_line"] == 1
        assert result.metadata["end_line"] == 3
        assert result.metadata["file_path"].endswith("test.py")

    def test_read_content_includes_snippet_header(self, tmp_path):
        """LLM 可见的 content 中包含 snippet_id，确保能传给 Edit。"""
        f = tmp_path / "header.py"
        f.write_text("hello\nworld\n", encoding="utf-8")

        tool = FileReadTool(session_id="test-session")
        result = tool.execute(file_path=str(f))

        assert not result.is_error
        # content 头部应有 snippet 信息
        assert result.content.startswith("[snippet_id:")
        assert "| lines:" in result.content
        assert "| scope: full]" in result.content
        # snippet_id 具体值
        assert result.metadata["snippet_id"] in result.content

    def test_read_twice_same_file_two_snippets(self, tmp_path):
        f = tmp_path / "twice.py"
        f.write_text("a\nb\nc\n", encoding="utf-8")

        tool = FileReadTool(session_id="test-session")
        r1 = tool.execute(file_path=str(f))
        r2 = tool.execute(file_path=str(f))

        sid1 = r1.metadata["snippet_id"]
        sid2 = r2.metadata["snippet_id"]
        # 同一个 version，两个 snippet 的 version 相同
        assert sid1 != sid2  # 不同 id
        s1 = get_snippet("test-session", sid1)
        s2 = get_snippet("test-session", sid2)
        assert s1.file_version == s2.file_version == 1

    def test_read_without_session_id_no_snippet(self, tmp_path):
        f = tmp_path / "nosession.py"
        f.write_text("data\n", encoding="utf-8")

        tool = FileReadTool()  # no session_id
        result = tool.execute(file_path=str(f))
        assert not result.is_error
        assert result.metadata is None
        # 无 session_id 时也不应有 snippet header
        assert not result.content.startswith("[snippet_id:")

    def test_read_error_paths_no_snippet(self, tmp_path):
        tool = FileReadTool(session_id="test-session")

        # 不存在的文件
        r = tool.execute(file_path=str(tmp_path / "does_not_exist.py"))
        assert r.is_error
        assert r.metadata is None

        # 目录不是文件
        r = tool.execute(file_path=str(tmp_path))
        assert r.is_error
        assert r.metadata is None

    def test_read_with_offset_limit_partial_snippet(self, tmp_path):
        """Bug fix: partial read 的 snippet 行范围必须与 offset/limit 一致。"""
        f = tmp_path / "big.py"
        f.write_text("\n".join(str(i) for i in range(1, 101)), encoding="utf-8")  # 100 lines

        tool = FileReadTool(session_id="test-session")
        # 读第 50-59 行
        result = tool.execute(file_path=str(f), offset=50, limit=10)

        assert not result.is_error
        assert result.metadata is not None
        assert result.metadata["scope_type"] == "snippet"
        assert result.metadata["start_line"] == 50
        assert result.metadata["end_line"] == 59

    def test_read_offset_only(self, tmp_path):
        """只有 offset 没有 limit → 从 offset 读到文件尾，scope_type=snippet。"""
        f = tmp_path / "offset_only.py"
        f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")

        tool = FileReadTool(session_id="test-session")
        result = tool.execute(file_path=str(f), offset=3)

        assert not result.is_error
        assert result.metadata["scope_type"] == "snippet"
        assert result.metadata["start_line"] == 3
        assert result.metadata["end_line"] == 5  # 读到末尾

    def test_read_limit_only(self, tmp_path):
        """只有 limit 没有 offset → 从第 1 行开始，scope_type=snippet。"""
        f = tmp_path / "limit_only.py"
        f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")

        tool = FileReadTool(session_id="test-session")
        result = tool.execute(file_path=str(f), limit=2)

        assert not result.is_error
        assert result.metadata["scope_type"] == "snippet"
        assert result.metadata["start_line"] == 1
        assert result.metadata["end_line"] == 2

    def test_snippet_counter_starts_from_0(self):
        """Counter 对称性：full_0 / snp_0 起步。"""
        record_file_state("s1", "/tmp/cnt.py", "x", 1.0)
        full = create_snippet("s1", "/tmp/cnt.py", 1, 1, scope_type="full")
        snip = create_snippet("s1", "/tmp/cnt.py", 1, 1, scope_type="snippet")
        assert full.id == "full_0"
        assert snip.id == "snp_0"


# ============================================================================
# FileState / invalidate_all 测试
# ============================================================================


class TestFileStateFields:
    def test_line_endings_default(self):
        fs = FileState(file_path="/tmp/x.py", content="", mtime=0.0)
        assert fs.line_endings == "LF"


class TestInvalidateAll:
    def test_invalidate_all_makes_all_stale(self):
        record_file_state("s1", "/tmp/a.py", "a", 1.0)
        record_file_state("s1", "/tmp/b.py", "b", 1.0)
        sa = create_snippet("s1", "/tmp/a.py", 1, 1, scope_type="full")
        sb = create_snippet("s1", "/tmp/b.py", 1, 1, scope_type="full")

        assert not is_snippet_stale("s1", sa)
        assert not is_snippet_stale("s1", sb)

        invalidate_all_snippets("s1")

        assert is_snippet_stale("s1", sa)
        assert is_snippet_stale("s1", sb)
