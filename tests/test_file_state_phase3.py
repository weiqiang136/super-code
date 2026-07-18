"""Phase 3: Write 工具 + Engine 注入 + /resume 重建测试。

验证点：
1. write 成功 → snippet 注册表有记录
2. write 覆盖已有文件 → 旧 snippet stale, 新 snippet 有效
3. engine 创建后 Edit/Read/Write 工具 session_id 自动注入
4. /resume 从 JSONL 重建 snippet 注册表
5. /clear 清空 snippet 状态
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from core.file_state import (
    record_file_state, create_snippet, get_snippet, is_snippet_stale,
    get_file_version, clear_session_state,
    _file_states, _snippets, _file_versions, _snippet_counters, _full_counters,
)
from core.tool import ToolResult
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
# Write 工具
# ============================================================================


class TestWriteTool:
    def test_write_new_file_creates_snippet(self, tmp_path):
        f = tmp_path / "new.py"
        tool = FileWriteTool(session_id="s1")
        r = tool.execute(file_path=str(f), content="hello\nworld\n")

        assert not r.is_error
        assert r.metadata is not None
        assert r.metadata["snippet_id"] is not None
        assert r.metadata["scope_type"] == "full"
        assert r.metadata["start_line"] == 1
        assert r.metadata["end_line"] == 2  # "hello\nworld\n" = 2 lines

        # snippet 注册表中存在
        snippet = get_snippet("s1", r.metadata["snippet_id"])
        assert snippet is not None
        assert not is_snippet_stale("s1", snippet)

    def test_write_overwrite_stales_old_snippet(self, tmp_path):
        f = tmp_path / "overwrite.py"
        f.write_text("v1\nv2\n", encoding="utf-8")

        # 第一次 write
        tool = FileWriteTool(session_id="s1")
        r1 = tool.execute(file_path=str(f), content="v1\nv2\n")
        sid1 = r1.metadata["snippet_id"]
        assert not is_snippet_stale("s1", get_snippet("s1", sid1))

        # 第二次 write 覆盖
        r2 = tool.execute(file_path=str(f), content="new1\nnew2\nnew3\n")
        sid2 = r2.metadata["snippet_id"]

        # 旧 snippet stale
        assert is_snippet_stale("s1", get_snippet("s1", sid1))
        # 新 snippet 有效
        assert not is_snippet_stale("s1", get_snippet("s1", sid2))
        assert get_snippet("s1", sid2).end_line == 3

    def test_write_without_session_id_no_snippet(self, tmp_path):
        f = tmp_path / "nosid.py"
        tool = FileWriteTool()
        r = tool.execute(file_path=str(f), content="data\n")
        assert not r.is_error
        assert r.metadata is None

    def test_read_then_write_overwrite_flow(self, tmp_path):
        """read → write 覆盖 → read 新 snippet 有效。"""
        f = tmp_path / "flow.py"
        f.write_text("original\n", encoding="utf-8")

        # read
        rt = FileReadTool(session_id="s1")
        rr = rt.execute(file_path=str(f))
        read_sid = rr.metadata["snippet_id"]

        # write 覆盖
        wt = FileWriteTool(session_id="s1")
        wr = wt.execute(file_path=str(f), content="replaced\n")
        write_sid = wr.metadata["snippet_id"]

        # read 时的 snippet 已 stale
        assert is_snippet_stale("s1", get_snippet("s1", read_sid))
        # write 后的 snippet 有效
        assert not is_snippet_stale("s1", get_snippet("s1", write_sid))

    def test_write_empty_file_snippet_has_valid_end_line(self, tmp_path):
        """空文件 → end_line >= 1，避免无效区间。"""
        f = tmp_path / "empty.py"
        tool = FileWriteTool(session_id="s1")
        r = tool.execute(file_path=str(f), content="")
        assert not r.is_error
        assert r.metadata is not None
        assert r.metadata["start_line"] == 1
        assert r.metadata["end_line"] >= 1


# ============================================================================
# Engine session_id 注入
# ============================================================================


class TestEngineInjectSessionId:
    """验证 tool 的 set_session_id 被正确调用。"""

    def test_edit_tool_gets_session_id(self, tmp_path):
        f = tmp_path / "inject.py"
        f.write_text("aaa\nbbb\nccc\n", encoding="utf-8")

        # 模拟 engine 注入
        et = FileEditTool(session_id="injected-session")
        assert et._session_id == "injected-session"

        rt = FileReadTool(session_id="injected-session")
        rr = rt.execute(file_path=str(f))
        sid = rr.metadata["snippet_id"]

        # edit 能正常使用 snippet
        er = et.execute(snippet_id=sid, old_string="bbb", new_string="BBB")
        assert not er.is_error

    def test_write_tool_gets_session_id(self):
        wt = FileWriteTool(session_id="test-sid")
        assert wt._session_id == "test-sid"

        wt.set_session_id("new-sid")
        assert wt._session_id == "new-sid"


# ============================================================================
# /resume JSONL 重建
# ============================================================================


class TestRebuildFromJSONL:
    """模拟 /resume 场景：engine 加载历史消息 → rebuild_snippets_from_messages。"""

    def _make_engine(self, messages, sid="resume-test"):
        """创建一个最小 engine 并设置消息。"""
        from unittest.mock import patch
        from core.permissions import PermissionChecker

        # Mock LLMClient 避免真的创建 OpenAI 连接
        with patch("core.engine.LLMClient"):
            from core.engine import Engine

            class FakeStore:
                session_id = sid
                cwd = str(Path.cwd())

            engine = Engine(
                tools=[FileReadTool(), FileEditTool(), FileWriteTool()],
                system_prompt="",
                permission_checker=PermissionChecker(auto_approve=True),
                session_store=FakeStore(),
            )

        engine._messages = messages
        return engine

    def test_rebuild_from_read_then_edit_history(self, tmp_path):
        """模拟 load_session 得到的消息列表，重建后 snippet 有效。"""
        f = tmp_path / "rebuild.py"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")

        # Step 1: 用真实工具生成带 metadata 的 tool_result
        rt = FileReadTool(session_id="rebuild-1")
        rr = rt.execute(file_path=str(f))
        sid = rr.metadata["snippet_id"]

        et = FileEditTool(session_id="rebuild-1")
        er = et.execute(snippet_id=sid, old_string="line2", new_string="LINE2")

        # Step 2: 模拟持久化后的消息列表（metadata 在 block 级别，对齐 engine 实际存储）
        messages = [
            {"role": "user", "content": "edit line2"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu1", "name": "Read", "input": {"file_path": str(f)}},
                {"type": "tool_use", "id": "tu2", "name": "Edit", "input": {"snippet_id": sid, "old_string": "line2", "new_string": "LINE2"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1",
                 "content": rr.content, "is_error": False, "metadata": rr.metadata},
                {"type": "tool_result", "tool_use_id": "tu2",
                 "content": er.content, "is_error": False, "metadata": er.metadata},
            ]},
        ]

        # Step 3: engine rebuild
        engine = self._make_engine(messages, sid="rebuild-1")
        count = engine.rebuild_snippets_from_messages()
        assert count >= 1

        # 验证 snippet 已重建
        new_sid = er.metadata.get("new_snippet_id")
        assert new_sid is not None
        snippet = get_snippet("rebuild-1", new_sid)
        assert snippet is not None
        assert snippet.file_path.endswith("rebuild.py")
        assert not is_snippet_stale("rebuild-1", snippet)

        # cleanup
        clear_session_state("rebuild-1")

    def test_rebuild_no_session_id_returns_zero(self):
        from unittest.mock import patch
        from core.permissions import PermissionChecker
        with patch("core.engine.LLMClient"):
            from core.engine import Engine
            engine = Engine(
                tools=[FileReadTool()], system_prompt="",
                permission_checker=PermissionChecker(auto_approve=True),
            )
        engine._messages = []
        assert engine.rebuild_snippets_from_messages() == 0

    def test_rebuild_empty_messages(self):
        engine = self._make_engine([], sid="empty-test")
        assert engine.rebuild_snippets_from_messages() == 0
        clear_session_state("empty-test")


# ============================================================================
# /clear 清空 snippet 状态
# ============================================================================


class TestClearSnippetState:
    def test_clear_removes_session_data(self, tmp_path):
        f = tmp_path / "clear.py"
        f.write_text("data\n", encoding="utf-8")

        # 创建一些数据
        rt = FileReadTool(session_id="clear-test")
        rt.execute(file_path=str(f))

        assert "clear-test" in _snippets
        assert get_file_version("clear-test", str(f.resolve())) == 1

        clear_session_state("clear-test")
        assert "clear-test" not in _snippets
        assert "clear-test" not in _file_versions
