"""Phase 4: Compact 集成 + 边界情况测试。

验证点：
1. invalidate_all_snippets 使所有 snippet 失效
2. compact 后旧 snippet stale → edit 被拒绝
3. compact 后重新 read → 新 snippet 有效
4. 已删除文件 read → 不创建 snippet
5. 空文件 read → snippet 有效
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from core.file_state import (
    record_file_state, create_snippet, get_snippet, is_snippet_stale,
    get_file_version, clear_session_state, invalidate_all_snippets,
    _file_states, _snippets, _file_versions, _snippet_counters, _full_counters,
)
from tools.file_read import FileReadTool
from tools.file_edit import FileEditTool
from tools.file_write import FileWriteTool


@pytest.fixture(autouse=True)
def clean_state():
    for d in [_file_states, _snippets, _file_versions, _snippet_counters, _full_counters]:
        d.clear()
    yield
    for d in [_file_states, _snippets, _file_versions, _snippet_counters, _full_counters]:
        d.clear()


# ============================================================================
# Invalidate All
# ============================================================================


class TestInvalidateAll:
    def test_all_snippets_stale_after_invalidate(self):
        """invalidate_all_snippets 后，所有已存在 snippet 全部 stale。"""
        record_file_state("s1", "/tmp/a.py", "a", 1.0)
        record_file_state("s1", "/tmp/b.py", "b", 1.0)
        sa = create_snippet("s1", "/tmp/a.py", 1, 1, scope_type="full")
        sb = create_snippet("s1", "/tmp/b.py", 1, 1, scope_type="full")

        assert not is_snippet_stale("s1", sa)
        assert not is_snippet_stale("s1", sb)

        invalidate_all_snippets("s1")

        assert is_snippet_stale("s1", sa)
        assert is_snippet_stale("s1", sb)

    def test_new_snippet_after_invalidate_is_valid(self):
        """invalidate 后重新 read 得到的新 snippet 仍然有效。"""
        record_file_state("s1", "/tmp/c.py", "c", 1.0)
        s_old = create_snippet("s1", "/tmp/c.py", 1, 1, scope_type="full")

        invalidate_all_snippets("s1")
        assert is_snippet_stale("s1", s_old)

        # 重新 record（模拟重新 read）→ 新 snippet 有效
        record_file_state("s1", "/tmp/c.py", "c", 2.0)
        s_new = create_snippet("s1", "/tmp/c.py", 1, 1, scope_type="full")
        assert not is_snippet_stale("s1", s_new)


# ============================================================================
# Compact 模拟
# ============================================================================


class TestCompactFlow:
    """模拟 compact 后 snippet 行为。"""

    def test_compact_makes_old_snippet_stale(self, tmp_path):
        """compact → invalidate_all → 旧 snippet stale，edit 被拒。"""
        f = tmp_path / "comp.py"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")

        # read → snippet
        rt = FileReadTool(session_id="comp-test")
        rr = rt.execute(file_path=str(f))
        sid = rr.metadata["snippet_id"]

        # 模拟 compact: invalidate_all
        invalidate_all_snippets("comp-test")

        # edit 旧 snippet → 拒绝
        et = FileEditTool(session_id="comp-test")
        er = et.execute(snippet_id=sid, old_string="line2", new_string="LINE2")
        assert er.is_error
        assert "modified since snippet" in er.content

        clear_session_state("comp-test")

    def test_after_compact_edit_works_after_reread(self, tmp_path):
        """compact → re-read → edit 成功。"""
        f = tmp_path / "comp2.py"
        f.write_text("v1\nv2\nv3\n", encoding="utf-8")

        rt = FileReadTool(session_id="comp2-test")
        rr = rt.execute(file_path=str(f))
        sid = rr.metadata["snippet_id"]

        # compact
        invalidate_all_snippets("comp2-test")

        # re-read
        rr2 = rt.execute(file_path=str(f))
        sid2 = rr2.metadata["snippet_id"]

        # edit 成功
        et = FileEditTool(session_id="comp2-test")
        er = et.execute(snippet_id=sid2, old_string="v2", new_string="V2")
        assert not er.is_error

        clear_session_state("comp2-test")


# ============================================================================
# 边界情况
# ============================================================================


class TestEdgeCases:
    def test_read_deleted_file_no_snippet(self, tmp_path):
        """已删除的文件 read → error，无 snippet。"""
        f = tmp_path / "ghost.py"
        f.write_text("data\n", encoding="utf-8")
        f.unlink()  # 删除

        rt = FileReadTool(session_id="edge-test")
        r = rt.execute(file_path=str(f))
        assert r.is_error
        assert r.metadata is None

    def test_read_empty_file_creates_valid_snippet(self, tmp_path):
        """空文件 read → snippet 有效（end_line >= 1）。"""
        f = tmp_path / "empty.py"
        f.write_text("", encoding="utf-8")

        rt = FileReadTool(session_id="edge-test")
        r = rt.execute(file_path=str(f))
        assert not r.is_error
        assert r.metadata is not None
        assert r.metadata["start_line"] == 1
        assert r.metadata["end_line"] >= 1

    def test_write_then_read_snippet_chain(self, tmp_path):
        """write → read → edit 完整链路。"""
        f = tmp_path / "chain.py"

        # write
        wt = FileWriteTool(session_id="chain")
        wr = wt.execute(file_path=str(f), content="aaa\nbbb\nccc\n")

        # read
        rt = FileReadTool(session_id="chain")
        rr = rt.execute(file_path=str(f))
        sid = rr.metadata["snippet_id"]

        # edit
        et = FileEditTool(session_id="chain")
        er = et.execute(snippet_id=sid, old_string="bbb", new_string="BBB")
        assert not er.is_error
        assert f.read_text(encoding="utf-8") == "aaa\nBBB\nccc\n"

        clear_session_state("chain")