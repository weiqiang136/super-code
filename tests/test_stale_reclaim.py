"""Tests for stale read-result reclamation (过期 Read 结果回收).

Covers:
- Files NOT modified after read → not reclaimed (identity / same reference)
- Files modified (version bumped) after read → reclaimed: content replaced with
  short marker, block structure / tool_use_id / metadata preserved
- Reclamation preserves tool_result → role:"tool" mapping for the OpenAI API
- No-metadata tool_results (Bash etc.) → untouched
- Unknown snippet_id → fail-closed, untouched
- Stats correctness (reclaimed count, chars_removed)
- Mixed scenario: stale reclaimed, current kept
"""
import sys
sys.path.insert(0, "src")

from features.compact import reclaim_stale_read_results
from core.file_state import (
    clear_session_state,
    create_snippet,
    get_file_version,
    record_file_state,
)
from core.llm import _to_openai_messages

SID = "test-stale-reclaim"
PATH = "D:/repo/src/app.py"


def _reset():
    clear_session_state(SID)


def _read_result_user(tid, text, meta):
    """构造一条 Read 工具结果的 user 消息（单 tool_result block）。"""
    return {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tid, "content": text,
         "is_error": False, "metadata": meta},
    ]}


def _tool_use_assistant(tid):
    return {"role": "assistant", "content": [
        {"type": "tool_use", "id": tid, "name": "Read", "input": {"file_path": PATH}},
    ]}


def _snippet_meta(snippet):
    return {
        "snippet_id": snippet.id,
        "file_path": snippet.file_path,
        "start_line": snippet.start_line,
        "end_line": snippet.end_line,
        "scope_type": snippet.scope_type,
    }


def _has_orphan_tool_use(msgs):
    """任何 assistant tool_use 缺对应 tool_result → True。"""
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and isinstance(m.get("content", ""), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tid = b["id"]
                    found = any(
                        isinstance(nm.get("content", ""), list)
                        and any(
                            bb.get("type") == "tool_result"
                            and bb.get("tool_use_id") == tid
                            for bb in nm.get("content", [])
                        )
                        for nm in msgs[i + 1:]
                    )
                    if not found:
                        return True
    return False


# ---------------------------------------------------------------------------
# 未被修改 → 不回收
# ---------------------------------------------------------------------------

def test_file_not_modified_not_reclaimed():
    """Read 后文件版本未变 → 结果原样保留（同一对象引用）。"""
    _reset()
    record_file_state(SID, PATH, "content v1", 123.0)          # version=1
    snip = create_snippet(SID, PATH, 1, 100, scope_type="full")  # file_version=1
    msg = _read_result_user("t1", "BIG CONTENT " * 500, _snippet_meta(snip))
    msgs = [_tool_use_assistant("t1"), msg]

    out, stats = reclaim_stale_read_results(msgs, SID)
    assert stats == {"reclaimed": 0, "chars_removed": 0}
    assert out[1] is msg, "未过期消息应保持同一引用（零拷贝）"
    assert out[1]["content"][0]["content"] == "BIG CONTENT " * 500


# ---------------------------------------------------------------------------
# 已被修改 → 回收
# ---------------------------------------------------------------------------

def test_file_modified_after_read_reclaimed():
    """Read 后 Edit（version+1）→ 结果被替换为短标记，结构字段保留。"""
    _reset()
    record_file_state(SID, PATH, "content v1", 123.0)            # version=1
    snip = create_snippet(SID, PATH, 1, 100, scope_type="full")   # file_version=1
    # 模拟 Edit：bump version → 2
    record_file_state(SID, PATH, "content v2", 124.0, bump_version=True)
    assert get_file_version(SID, PATH) == 2

    original = "BIG CONTENT " * 500
    msg = _read_result_user("t1", original, _snippet_meta(snip))
    msgs = [_tool_use_assistant("t1"), msg]

    out, stats = reclaim_stale_read_results(msgs, SID)
    assert stats["reclaimed"] == 1
    assert stats["chars_removed"] == len(original)

    block = out[1]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "t1", "tool_use_id 必须保留（配对）"
    assert block["is_error"] is False, "is_error 字段必须保留"
    assert block["metadata"] == _snippet_meta(snip), "metadata 必须保留（供重读/重建）"
    new_text = block["content"]
    assert len(new_text) < len(original) // 10, "替换后应远短于原文"
    assert "snippet_id" in new_text, "标记应保留 snippet_id 定位信息"
    assert "旧版本" in new_text, "标记应提示内容为旧版本"
    assert "offset/limit" in new_text, "标记应提示重读方式"


def test_reclaimed_block_is_new_dict_original_untouched():
    """回收不污染输入消息（重建 block，不改原对象）。"""
    _reset()
    record_file_state(SID, PATH, "v1", 1.0)
    snip = create_snippet(SID, PATH, 1, 10, scope_type="snippet")
    record_file_state(SID, PATH, "v2", 2.0, bump_version=True)
    msg = _read_result_user("t1", "OLD" * 100, _snippet_meta(snip))
    msgs = [_tool_use_assistant("t1"), msg]

    out, _ = reclaim_stale_read_results(msgs, SID)
    assert out[1] is not msg, "回收后 user 消息应重建"
    assert msg["content"][0]["content"] == "OLD" * 100, "原始输入不得被修改"


# ---------------------------------------------------------------------------
# 无 metadata / 未知 snippet → fail-closed 不回收
# ---------------------------------------------------------------------------

def test_no_metadata_tool_result_not_reclaimed():
    """Bash 等无 snippet metadata 的 tool_result → 原样保留。"""
    _reset()
    msg = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "b1", "content": "ls output" * 100,
         "is_error": False, "metadata": None},
    ]}
    out, stats = reclaim_stale_read_results([msg], SID)
    assert stats["reclaimed"] == 0
    assert out[0] is msg


def test_unknown_snippet_id_fail_closed():
    """snippet_id 查不到（/resume 未重建等）→ 不回收、不抛异常。"""
    _reset()
    meta = {"snippet_id": "full_999", "file_path": PATH,
            "start_line": 1, "end_line": 10, "scope_type": "full"}
    msg = _read_result_user("t1", "content" * 100, meta)
    out, stats = reclaim_stale_read_results([msg], SID)
    assert stats["reclaimed"] == 0
    assert out[0] is msg


def test_empty_session_id_fail_closed():
    """session_id 为空 → 查不到任何 snippet → 不回收。"""
    _reset()
    record_file_state(SID, PATH, "v1", 1.0)
    snip = create_snippet(SID, PATH, 1, 10, scope_type="full")
    record_file_state(SID, PATH, "v2", 2.0, bump_version=True)
    msg = _read_result_user("t1", "content" * 100, _snippet_meta(snip))
    out, stats = reclaim_stale_read_results([msg], "")
    assert stats["reclaimed"] == 0


# ---------------------------------------------------------------------------
# 混合场景 + 配对完整性
# ---------------------------------------------------------------------------

def test_mixed_scenario_stale_reclaimed_current_kept():
    """同会话两个文件：被修改的回收、未修改的保留。"""
    _reset()
    path2 = "D:/repo/src/util.py"
    # 文件 A：Read 后被修改 → 回收
    record_file_state(SID, PATH, "a v1", 1.0)
    snip_a = create_snippet(SID, PATH, 1, 50, scope_type="full")
    record_file_state(SID, PATH, "a v2", 2.0, bump_version=True)
    # 文件 B：Read 后未修改 → 保留
    record_file_state(SID, path2, "b v1", 3.0)
    snip_b = create_snippet(SID, path2, 1, 80, scope_type="full")

    msgs = [
        _tool_use_assistant("t_a"),
        _read_result_user("t_a", "AAA" * 200, _snippet_meta(snip_a)),
        _tool_use_assistant("t_b"),
        _read_result_user("t_b", "BBB" * 200, _snippet_meta(snip_b)),
    ]
    out, stats = reclaim_stale_read_results(msgs, SID)
    assert stats["reclaimed"] == 1
    assert "旧版本" in out[1]["content"][0]["content"], "文件 A 被回收"
    assert out[3]["content"][0]["content"] == "BBB" * 200, "文件 B 原样保留"


def test_reclaim_preserves_pairing_no_orphan():
    """回收后无孤儿 tool_use（端到端配对完整性）。"""
    _reset()
    record_file_state(SID, PATH, "v1", 1.0)
    snip = create_snippet(SID, PATH, 1, 100, scope_type="full")
    record_file_state(SID, PATH, "v2", 2.0, bump_version=True)
    msgs = [
        {"role": "user", "content": "please fix"},
        _tool_use_assistant("t1"),
        _read_result_user("t1", "OLD CONTENT" * 300, _snippet_meta(snip)),
        {"role": "assistant", "content": "done"},
    ]
    out, stats = reclaim_stale_read_results(msgs, SID)
    assert stats["reclaimed"] == 1
    assert not _has_orphan_tool_use(out), "回收后不得产生孤儿 tool_use"


# ---------------------------------------------------------------------------
# 端到端：OpenAI 消息转换
# ---------------------------------------------------------------------------

def test_reclaimed_messages_convert_to_openai():
    """回收后的消息列表经 _to_openai_messages 转换无异常、tool 配对正确。"""
    _reset()
    record_file_state(SID, PATH, "v1", 1.0)
    snip = create_snippet(SID, PATH, 1, 100, scope_type="full")
    record_file_state(SID, PATH, "v2", 2.0, bump_version=True)
    msgs = [
        {"role": "user", "content": "fix it"},
        _tool_use_assistant("t1"),
        _read_result_user("t1", "OLD" * 500, _snippet_meta(snip)),
        {"role": "assistant", "content": "done"},
    ]
    out, _ = reclaim_stale_read_results(msgs, SID)
    converted = _to_openai_messages("sys", out)
    tool_msgs = [m for m in converted if m.get("role") == "tool"]
    assert len(tool_msgs) == 1, "应恰有一条 tool 消息（配对完整）"
    assert tool_msgs[0]["tool_call_id"] == "t1"
    assert "旧版本" in tool_msgs[0]["content"]


def test_non_tool_result_messages_untouched():
    """不含 tool_result 的消息原样保留（同一引用）。"""
    _reset()
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    out, stats = reclaim_stale_read_results(msgs, SID)
    assert stats == {"reclaimed": 0, "chars_removed": 0}
    assert out == msgs
