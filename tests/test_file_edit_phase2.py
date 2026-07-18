"""Phase 2: Edit 工具 snippet_id 接入测试。

验证点：
1. 无 snippet_id → 报错
2. 无效 snippet_id → 报错
3. snippet stale（版本升级） → 报错
4. snippet_id/file_path 不匹配 → 报错
5. 正常 read → edit 成功
6. 范围内匹配不到 → 报错（附带 scope 信息）
7. 范围内多处匹配 → 报错（附带 candidates）
8. replace_all 全替换成功
9. edit 后 old snippet stale
10. edit 后返回 new_snippet_id
"""

import sys
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
    _file_states,
    _snippets,
    _file_versions,
    _snippet_counters,
    _full_counters,
)
from tools.file_read import FileReadTool
from tools.file_edit import FileEditTool


@pytest.fixture(autouse=True)
def clean_state():
    """每个测试前清空全局状态。"""
    for d in [_file_states, _snippets, _file_versions, _snippet_counters, _full_counters]:
        d.clear()
    yield
    for d in [_file_states, _snippets, _file_versions, _snippet_counters, _full_counters]:
        d.clear()


# ============================================================================
# Schedule
# ============================================================================


def _read(tmp_path, filename: str, content: str, session_id="s1",
          offset=None, limit=None) -> tuple[FileReadTool, dict]:
    """Helper: 创建文件 + Read + 返回 (tool, result)。"""
    f = tmp_path / filename
    f.write_text(content, encoding="utf-8")
    tool = FileReadTool(session_id=session_id)
    kwargs = {"file_path": str(f)}
    if offset is not None:
        kwargs["offset"] = offset
    if limit is not None:
        kwargs["limit"] = limit
    result = tool.execute(**kwargs)
    return tool, result


def _edit(tmp_path, session_id, snippet_id, old_string, new_string,
          file_path="", replace_all=False) -> tuple[FileEditTool, dict]:
    """Helper: Edit 操作。"""
    tool = FileEditTool(session_id=session_id)
    kwargs = {
        "snippet_id": snippet_id,
        "old_string": old_string,
        "new_string": new_string,
        "replace_all": replace_all,
    }
    if file_path:
        kwargs["file_path"] = file_path
    result = tool.execute(**kwargs)
    return tool, result


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ============================================================================
# 签名校验
# ============================================================================


class TestEditRejectsBadInput:
    def test_no_snippet_id(self, tmp_path):
        f = tmp_path / "t.py"
        f.write_text("hello", encoding="utf-8")
        tool = FileEditTool(session_id="s1")
        r = tool.execute(snippet_id="", old_string="h", new_string="x")
        assert r.is_error
        assert "snippet_id is required" in r.content

    def test_unknown_snippet_id(self, tmp_path):
        tool = FileEditTool(session_id="s1")
        r = tool.execute(snippet_id="snp_999", old_string="x", new_string="y")
        assert r.is_error
        assert "Unknown snippet_id" in r.content

    def test_stale_snippet(self, tmp_path):
        _, rd = _read(tmp_path, "stale.py", "line1\nline2\nline3\n")
        sid = rd.metadata["snippet_id"]

        # bump version
        fp = rd.metadata["file_path"]
        record_file_state("s1", fp, "line1\nline2-CHANGED\nline3\n", 9999.0, bump_version=True)

        _, ed = _edit(tmp_path, "s1", sid, "line2", "NEW")
        assert ed.is_error
        assert "modified since snippet" in ed.content

    def test_snippet_file_path_mismatch(self, tmp_path):
        _, rd = _read(tmp_path, "a.py", "hello\n")
        sid = rd.metadata["snippet_id"]

        wrong_path = str(tmp_path / "b.py")
        (tmp_path / "b.py").write_text("hello\n", encoding="utf-8")

        _, ed = _edit(tmp_path, "s1", sid, "hello", "bye", file_path=wrong_path)
        assert ed.is_error
        assert "belongs to" in ed.content.lower()

    def test_old_equals_new(self, tmp_path):
        _, rd = _read(tmp_path, "x.py", "hello\n")
        sid = rd.metadata["snippet_id"]
        _, ed = _edit(tmp_path, "s1", sid, "hello", "hello")
        assert ed.is_error
        assert "must differ" in ed.content


# ============================================================================
# 正常编辑流程
# ============================================================================


class TestEditSuccess:
    def test_basic_edit(self, tmp_path):
        f = tmp_path / "basic.py"
        f.write_text("aaa\nbbb\nccc\n", encoding="utf-8")

        _, rd = _read(tmp_path, "basic.py", "aaa\nbbb\nccc\n")
        sid = rd.metadata["snippet_id"]

        _, ed = _edit(tmp_path, "s1", sid, "bbb", "BBB")
        assert not ed.is_error
        assert "Successfully replaced 1" in ed.content
        assert _read_file(f) == "aaa\nBBB\nccc\n"

        # metadata 中有 new_snippet_id
        assert ed.metadata is not None
        assert "new_snippet_id" in ed.metadata

    def test_edit_middle_of_file(self, tmp_path):
        content = "\n".join(str(i) for i in range(1, 21))  # lines 1-20
        f = tmp_path / "mid.py"
        f.write_text(content, encoding="utf-8")

        # 只读第 8-12 行
        _, rd = _read(tmp_path, "mid.py", content, offset=8, limit=5)
        sid = rd.metadata["snippet_id"]
        assert rd.metadata["scope_type"] == "snippet"
        assert rd.metadata["start_line"] == 8
        assert rd.metadata["end_line"] == 12

        # edit 只能在这 5 行里搜索
        _, ed = _edit(tmp_path, "s1", sid, "10", "TEN")
        assert not ed.is_error
        result = _read_file(f)
        lines = result.split("\n")
        assert lines[9] == "TEN"  # line 10 = index 9

    def test_replace_all(self, tmp_path):
        content = "X\nX\nX\nY\nX\n"
        f = tmp_path / "ra.py"
        f.write_text(content, encoding="utf-8")

        _, rd = _read(tmp_path, "ra.py", content)
        sid = rd.metadata["snippet_id"]

        _, ed = _edit(tmp_path, "s1", sid, "X", "Z", replace_all=True)
        assert not ed.is_error
        assert "Successfully replaced 4" in ed.content
        assert _read_file(f) == "Z\nZ\nZ\nY\nZ\n"

    def test_edit_makes_old_snippet_stale(self, tmp_path):
        f = tmp_path / "stale2.py"
        f.write_text("v1\n", encoding="utf-8")

        _, rd = _read(tmp_path, "stale2.py", "v1\n")
        sid1 = rd.metadata["snippet_id"]

        # edit 用 sid1
        _, ed = _edit(tmp_path, "s1", sid1, "v1", "v2")
        assert not ed.is_error

        # sid1 现在 stale
        import core.file_state as fs
        snip = fs.get_snippet("s1", sid1)
        assert fs.is_snippet_stale("s1", snip)

        # new_snippet_id 有效
        new_sid = ed.metadata["new_snippet_id"]
        snip2 = fs.get_snippet("s1", new_sid)
        assert snip2 is not None
        assert not fs.is_snippet_stale("s1", snip2)


# ============================================================================
# 范围搜索 & 错误信息
# ============================================================================


class TestEditScopeSearch:
    def test_not_found_in_scope(self, tmp_path):
        content = "line1\nline2\nline3\nline4\nline5\n"
        f = tmp_path / "scope.py"
        f.write_text(content, encoding="utf-8")

        # 只读第 1-2 行
        _, rd = _read(tmp_path, "scope.py", content, offset=1, limit=2)
        sid = rd.metadata["snippet_id"]

        # 尝试改 line4，但 snippet 只覆盖 1-2
        _, ed = _edit(tmp_path, "s1", sid, "line4", "NEW")
        assert ed.is_error
        assert "not found" in ed.content or "within the snippet scope" in ed.content
        # metadata 中应有 scope 信息
        assert ed.metadata is not None
        assert "scope" in ed.metadata

    def test_not_found_anywhere(self, tmp_path):
        _, rd = _read(tmp_path, "nf.py", "hello\nworld\n")
        sid = rd.metadata["snippet_id"]

        _, ed = _edit(tmp_path, "s1", sid, "nonexistent", "x")
        assert ed.is_error
        assert "not found" in ed.content

    def test_multiple_matches_returns_candidates(self, tmp_path):
        content = "dup\nunique\ndup\n"
        f = tmp_path / "dup.py"
        f.write_text(content, encoding="utf-8")

        _, rd = _read(tmp_path, "dup.py", content)
        sid = rd.metadata["snippet_id"]

        _, ed = _edit(tmp_path, "s1", sid, "dup", "NEW")
        assert ed.is_error
        assert "not unique" in ed.content
        assert ed.metadata is not None
        assert "candidates" in ed.metadata
        assert ed.metadata["match_count"] == 2

    def test_scope_isolates_from_other_matches(self, tmp_path):
        """相同的 old_string 在文件其他地方出现，但 snippet 范围限制后只命中 1 次 → 成功。"""
        content = "dup\nAAA\ndup\nBBB\ndup\n"
        f = tmp_path / "iso.py"
        f.write_text(content, encoding="utf-8")

        # 只读第 3-4 行（即 "dup\nBBB"）
        _, rd = _read(tmp_path, "iso.py", content, offset=3, limit=2)
        sid = rd.metadata["snippet_id"]
        assert rd.metadata["start_line"] == 3
        assert rd.metadata["end_line"] == 4

        # 在 snippet 范围内（3-4行），"dup" 只出现 1 次 → 成功
        _, ed = _edit(tmp_path, "s1", sid, "dup", "UNIQUE")
        assert not ed.is_error
        assert "Successfully replaced 1" in ed.content
        result = _read_file(f)
        lines = result.split("\n")
        assert lines[2] == "UNIQUE"  # line 3 = index 2
        assert lines[0] == "dup"     # line 1 没被改
        assert lines[4] == "dup"     # line 5 没被改


# ============================================================================
# 向后兼容（无 session_id）
# ============================================================================


class TestEditWithoutSessionId:
    def test_empty_old_string_blocked(self, tmp_path):
        _, rd = _read(tmp_path, "empty_str.py", "hello\n")
        sid = rd.metadata["snippet_id"]
        _, ed = _edit(tmp_path, "s1", sid, "", "x")
        assert ed.is_error
        assert "cannot be empty" in ed.content


# ============================================================================
# Bug 修复验证
# ============================================================================


class TestReplaceAllScope:
    """Bug 1 修复：replace_all 必须限制在 snippet 范围内。"""

    def test_replace_all_scoped_to_snippet(self, tmp_path):
        """scope 外的相同字符串不被 replace_all 修改。"""
        content = "X\nAAA\nX\nBBB\nX\n"
        f = tmp_path / "ra_scope.py"
        f.write_text(content, encoding="utf-8")

        # 只读第 2-4 行（"AAA\nX\nBBB"），X 只出现在第 3 行
        _, rd = _read(tmp_path, "ra_scope.py", content, offset=2, limit=3)
        sid = rd.metadata["snippet_id"]

        _, ed = _edit(tmp_path, "s1", sid, "X", "Z", replace_all=True)
        assert not ed.is_error
        assert "Successfully replaced 1" in ed.content  # scope 内只有 1 个 X
        result = _read_file(f)
        lines = result.split("\n")
        assert lines[0] == "X"    # line 1: 未被改（在 scope 外）
        assert lines[2] == "Z"    # line 3: 被替换
        assert lines[4] == "X"    # line 5: 未被改（在 scope 外）

    def test_replace_all_multi_line_expansion(self, tmp_path):
        """Bug 2 修复：多行替换后新 snippet 行数正确。"""
        content = "a\nOLD1\nOLD2\nb\n"
        f = tmp_path / "expand.py"
        f.write_text(content, encoding="utf-8")

        _, rd = _read(tmp_path, "expand.py", content)
        sid = rd.metadata["snippet_id"]

        # old_string 是 2 行，new_string 是 4 行 → 文件增加 2 行
        _, ed = _edit(tmp_path, "s1", sid, "OLD1\nOLD2", "NEW1\nNEW2\nNEW3\nNEW4")
        assert not ed.is_error

        import core.file_state as fs
        new_sid = ed.metadata["new_snippet_id"]
        new_snip = fs.get_snippet("s1", new_sid)
        assert new_snip is not None
        # 原来是 4 行，old 占 2 行被删，new 占 4 行插入 → 最终 6 行
        assert new_snip.end_line == 6
