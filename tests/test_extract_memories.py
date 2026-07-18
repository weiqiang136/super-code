"""
test_extract_memories.py — Step 5 后台抽取闭包逻辑测试。

只测决策层（游标 / 互斥 / 跳过条件）与 prompt 构造。
**不**真调 LLM —— 用 monkeypatch 把 _run_extract_agent 替换成固定返回值的桩。

每个测试都先 reset_extract_memories() 以拿到干净的闭包状态。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from features import extract_memories as em
from features.extract_memories import (
    MIN_NEW_MESSAGES,
    _format_recent_excerpt,
    _last_assistant_wrote_memory,
    build_extract_prompt,
    execute_extract_memories,
    reset_extract_memories,
)


# 一个轻量 app_config 桩：execute 内部只 forward 给被 mock 掉的 _run_extract_agent，
# 字段不会被真的读到（除非测试漏 mock），但仍按 AppConfig 字段名给齐。
_FAKE_CONFIG = SimpleNamespace(
    provider="openai",
    api_key=None,
    base_url=None,
    model="gpt-4o",
    max_tokens=2000,
)


@pytest.fixture(autouse=True)
def _reset_state():
    """每个用例独立闭包状态，防止上一个测试残留游标。"""
    reset_extract_memories()
    yield
    reset_extract_memories()


def _wait_until(predicate, timeout=2.0):
    """等待后台线程完成（用谓词轮询，最长 2s）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# _format_recent_excerpt
# ---------------------------------------------------------------------------

def test_excerpt_skips_non_user_assistant():
    """tool / system / 空 role 都不应进入 excerpt。"""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "tool result junk"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "sys"},
    ]
    out = _format_recent_excerpt(msgs)
    assert "[user]" in out and "hi" in out
    assert "[assistant]" in out and "hello" in out
    assert "tool result junk" not in out
    assert "[system]" not in out


def test_excerpt_handles_list_content_text_blocks():
    """list 形式内容只保留 text block，忽略 tool_use / tool_result。"""
    msgs = [
        {"role": "assistant", "content": [
            {"type": "text", "text": "plain answer"},
            {"type": "tool_use", "id": "x", "name": "Read", "input": {"file_path": "a.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "x", "content": "file contents"},
        ]},
    ]
    out = _format_recent_excerpt(msgs)
    assert "plain answer" in out
    # tool_use 名 / tool_result 内容都不该出现
    assert "Read" not in out
    assert "file contents" not in out


def test_excerpt_takes_only_last_N(monkeypatch):
    """超出 RECENT_MESSAGES_FOR_EXTRACT 的早期消息应被截掉。"""
    monkeypatch.setattr(em, "RECENT_MESSAGES_FOR_EXTRACT", 3)
    msgs = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
    out = _format_recent_excerpt(msgs, limit=3)
    # 只保留 msg7/8/9
    assert "msg9" in out and "msg8" in out and "msg7" in out
    assert "msg6" not in out


def test_excerpt_truncates_oversized_single_message():
    big = "x" * 6000
    msgs = [{"role": "assistant", "content": big}]
    out = _format_recent_excerpt(msgs)
    assert "…[truncated]" in out


# ---------------------------------------------------------------------------
# build_extract_prompt
# ---------------------------------------------------------------------------

def test_prompt_contains_required_sections(tmp_path: Path):
    p = build_extract_prompt("[user]\nhi", "- foo.md (...)", tmp_path)
    # 工具白名单 / 沙箱声明
    assert "Edit/Write" in p and "memory directory" in p
    # 类型约束
    assert "user`, `feedback`, `project`, `reference" in p
    # frontmatter 模板
    assert "type: <user|feedback|project|reference>" in p
    # 已有清单注入
    assert "Existing memory files" in p
    assert "foo.md" in p
    # 对话片段
    assert "[user]" in p and "hi" in p
    # memory_dir 插值
    assert str(tmp_path) in p


def test_prompt_omits_manifest_when_empty(tmp_path: Path):
    p = build_extract_prompt("[user]\nhi", "", tmp_path)
    assert "Existing memory files" not in p


# ---------------------------------------------------------------------------
# _last_assistant_wrote_memory
# ---------------------------------------------------------------------------

def test_assistant_write_inside_memory_dir_detected(tmp_path: Path):
    target = tmp_path / "feedback_x.md"
    msgs = [
        {"role": "user", "content": "save this"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Write", "input": {"file_path": str(target)}}
        ]},
    ]
    assert _last_assistant_wrote_memory(msgs, 0, tmp_path) is True


def test_assistant_write_outside_memory_dir_ignored(tmp_path: Path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    target = other / "src.py"
    target.write_text("x")
    msgs = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Write", "input": {"file_path": str(target)}}
        ]},
    ]
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    assert _last_assistant_wrote_memory(msgs, 0, mem_dir) is False


def test_assistant_write_since_index_respected(tmp_path: Path):
    """游标之前的 assistant 写入不算（避免老旧记录触发跳过）。"""
    target = tmp_path / "feedback_x.md"
    msgs = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Write", "input": {"file_path": str(target)}}
        ]},
        {"role": "user", "content": "next turn"},
    ]
    # since_index=1 → 只看第二条之后，第一条的 Write 不算
    assert _last_assistant_wrote_memory(msgs, 1, tmp_path) is False


# ---------------------------------------------------------------------------
# execute_extract_memories —— 决策层（用桩替代 LLM）
# ---------------------------------------------------------------------------

def _install_agent_stub(monkeypatch, written_paths: list[str], done: threading.Event):
    """把 _run_extract_agent 替换为同步返回 written_paths 的桩，并 set done。"""
    def stub(prompt: str, memory_dir: Path, app_config: Any) -> list[str]:
        try:
            return list(written_paths)
        finally:
            done.set()
    monkeypatch.setattr(em, "_run_extract_agent", stub)


def test_execute_returns_true_and_runs_worker(tmp_path: Path, monkeypatch):
    """execute 起线程后，stub 必须真正被调用一次（done 被 set 即证明 worker 跑完）。"""
    done = threading.Event()
    written = [str(tmp_path / "feedback_x.md")]
    _install_agent_stub(monkeypatch, written, done)

    msgs = [{"role": "user", "content": "use uv not pip"},
            {"role": "assistant", "content": "ok"}]

    started = execute_extract_memories(msgs, _FAKE_CONFIG, tmp_path)
    assert started is True
    assert _wait_until(done.is_set), "extract worker timed out"


def test_execute_skips_when_no_new_messages(tmp_path: Path, monkeypatch):
    """空 messages → 不应起线程。"""
    called = {"hit": False}
    monkeypatch.setattr(em, "_run_extract_agent",
                        lambda *a, **kw: (called.__setitem__("hit", True), [])[1])

    started = execute_extract_memories([], _FAKE_CONFIG, tmp_path)
    assert started is False
    time.sleep(0.05)
    assert called["hit"] is False


def test_execute_skips_when_main_agent_wrote_memory(tmp_path: Path, monkeypatch):
    """主智能体本轮已写过 memory_dir 内的文件 → 跳过抽取。"""
    target = tmp_path / "feedback_main.md"
    msgs = [
        {"role": "user", "content": "remember this"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Write", "input": {"file_path": str(target)}}
        ]},
    ]
    called = {"hit": False}
    monkeypatch.setattr(em, "_run_extract_agent",
                        lambda *a, **kw: (called.__setitem__("hit", True), [])[1])

    started = execute_extract_memories(msgs, _FAKE_CONFIG, tmp_path)
    assert started is False
    time.sleep(0.05)
    assert called["hit"] is False


def test_execute_advances_cursor_after_completion(tmp_path: Path, monkeypatch):
    """worker 跑完后游标必须前移；下一次同样 messages 应被节流跳过。"""
    done = threading.Event()
    _install_agent_stub(monkeypatch, [], done)

    msgs = [{"role": "user", "content": "msg1"}]
    assert execute_extract_memories(msgs, _FAKE_CONFIG, tmp_path) is True
    assert _wait_until(done.is_set)
    # 等 worker 的 finally 推进游标
    time.sleep(0.05)

    # 相同长度的 messages → MIN_NEW_MESSAGES 节流跳过
    assert execute_extract_memories(msgs, _FAKE_CONFIG, tmp_path) is False


def test_execute_in_progress_mutex(tmp_path: Path, monkeypatch):
    """worker 跑期间再来一发 → 必须被互斥跳过。"""
    block = threading.Event()
    started_event = threading.Event()

    def slow_stub(prompt, memory_dir, app_config):
        started_event.set()
        block.wait(timeout=2.0)  # 阻塞直到测试 set
        return []

    monkeypatch.setattr(em, "_run_extract_agent", slow_stub)

    first = execute_extract_memories(
        [{"role": "user", "content": "x"}], _FAKE_CONFIG, tmp_path)
    assert first is True
    # 等 worker 真的进入 stub
    assert started_event.wait(timeout=1.0)

    # 此时第二发应被 in_progress 跳过
    second = execute_extract_memories(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        _FAKE_CONFIG, tmp_path)
    assert second is False

    # 放开 worker，让游标推进
    block.set()
    # 等 in_progress 复位
    time.sleep(0.2)
