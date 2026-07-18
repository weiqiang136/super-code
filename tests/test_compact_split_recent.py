"""Tests for compact message splitting and alternation fixes.

Covers:
- _split_recent tool_use/tool_result pair preservation
- _fix_alternation not corrupting tool blocks when merging consecutive same-role messages
- End-to-end: _fix_alternation → _to_openai_messages preserves tool_result→tool mapping
"""
import sys
sys.path.insert(0, "src")
from features.compact import _split_recent, _fix_alternation
from core.llm import _to_openai_messages


def mu(blocks):
    return {"role": "user", "content": blocks}


def ma(blocks):
    return {"role": "assistant", "content": blocks}


def tr(tid):
    return {"type": "tool_result", "tool_use_id": tid, "content": "r"}


def tu(tid):
    return {"type": "tool_use", "id": tid, "name": "t", "input": {}}


def ut(text):
    return {"role": "user", "content": text}


def at(text):
    return {"role": "assistant", "content": text}


def has_orphan_tool_use(msgs):
    """Return True if any assistant with tool_use lacks tool_result follow-up."""
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
                        for nm in msgs[i + 1 :]
                    )
                    if not found:
                        return True
    return False


# Padding to make messages exceed MIN_RECENT limits and force a real split
PAD_U = ut("padding " * 1000)
PAD_A = at("padding " * 1000)
TAIL = [at("recent_a"), ut("recent_u")]
PADDING = [PAD_U, PAD_A] * 10


def test_pure_tool_result_user_at_keep_start():
    """keep_start lands on a user with all-tool_result blocks (original case)."""
    msgs = PADDING + [ma([tu("c1")]), mu([tr("c1")])] + TAIL
    h, r = _split_recent(msgs)
    assert not has_orphan_tool_use(h), "history has orphan tool_use"
    assert not has_orphan_tool_use(r), "recent has orphan tool_use"


def test_mixed_block_user_at_keep_start():
    """keep_start lands on a user with mixed (tool_result + text) blocks."""
    msgs = (
        PADDING
        + [ma([tu("c2")]), mu([tr("c2"), {"type": "text", "text": "mixed"}])]
        + TAIL
    )
    h, r = _split_recent(msgs)
    assert not has_orphan_tool_use(h), "history has orphan tool_use (mixed block missed)"
    assert not has_orphan_tool_use(r), "recent has orphan tool_use"


def test_tool_use_stays_out_of_history_even_with_string_user():
    """When tool_use followed by string-content user (degenerate), pair stays in recent.

    The while loop detects tool_use in the preceding assistant and moves keep_start
    back, so the assistant+user pair stays together in recent.  history must be
    orphan-free (it's sent to the LLM API).  The orphan in recent is a data problem
    (engine always creates tool_result blocks), not a split problem.
    """
    msgs = PADDING + [ma([tu("c3")]), ut("plain string")] + TAIL
    h, r = _split_recent(msgs)
    assert not has_orphan_tool_use(h), "history must never have orphan tool_use"


def test_consecutive_tool_use_pairs():
    """Two consecutive tool_use/tool_result pairs at split boundary."""
    msgs = (
        PADDING
        + [
            ma([tu("c4a")]),
            mu([tr("c4a")]),
            ma([tu("c4b")]),
            mu([tr("c4b")]),
        ]
        + TAIL
    )
    h, r = _split_recent(msgs)
    assert not has_orphan_tool_use(h), "history has orphan tool_use"
    assert not has_orphan_tool_use(r), "recent has orphan tool_use"


def test_regular_assistant_no_tool_use():
    """Assistant without tool_use at boundary should not trigger loop."""
    msgs = PADDING + [at("text only"), ut("u")] + TAIL
    h, r = _split_recent(msgs)
    assert len(h) > 0 and len(r) > 0, "split produced empty segment"


# ── _fix_alternation + _to_openai_messages integration tests ────────────


def count_role(openai_msgs: list[dict], role: str) -> int:
    """Count messages with a specific role in the OpenAI-format message list."""
    return sum(1 for m in openai_msgs if m.get("role") == role)


def test_fix_alternation_preserves_tool_result_when_merging_consecutive_users():
    """Consecutive user messages (tool_result + notification) must not lose tool blocks.

    Simulates the coordinator pattern where a tool_result user message is immediately
    followed by a <task-notification> user message.  _fix_alternation should insert a
    separator instead of merging, so that _to_openai_messages correctly converts the
    tool_result user into role:"tool" messages.
    """
    messages = [
        ma([tu("c1")]),                                      # assistant with tool_use
        mu([tr("c1")]),                                      # user with tool_result
        ut("<task-notification>worker done</task-notification>"),  # consecutive user!
    ]
    fixed = _fix_alternation(messages)
    # After fix, the two user messages should NOT be merged:
    # they should be separated by an assistant "Acknowledged."
    oai = _to_openai_messages(None, fixed)
    # There must be a role:"tool" message for tool_call_id "c1"
    tool_msgs = [m for m in oai if m.get("role") == "tool"]
    assert len(tool_msgs) >= 1, f"Expected >=1 tool messages, got {len(tool_msgs)}"
    assert any(m.get("tool_call_id") == "c1" for m in tool_msgs), (
        "tool_result for c1 was lost during _fix_alternation merge"
    )


def test_fix_alternation_does_not_merge_assistant_with_tool_use():
    """Consecutive assistant messages with tool_use should be separated, not merged."""
    messages = [
        ut("user input"),
        ma([tu("c2a")]),
        ma([tu("c2b")]),  # consecutive assistant with tool_use — coordinator multi-tool
    ]
    fixed = _fix_alternation(messages)
    oai = _to_openai_messages(None, fixed)
    # Both tool_calls should appear in the output
    tool_call_ids = []
    for m in oai:
        for tc in m.get("tool_calls", []):
            tool_call_ids.append(tc.get("id"))
    # The two assistant messages are separated by a user "Acknowledged."
    # Both should have their tool_calls preserved.
    assert "c2a" in tool_call_ids, f"tool_call c2a lost, got {tool_call_ids}"
    assert "c2b" in tool_call_ids, f"tool_call c2b lost, got {tool_call_ids}"


def test_fix_alternation_still_merges_plain_strings():
    """Plain string messages of same role should still be merged (no tool blocks)."""
    messages = [
        at("hello"),
        ut("first line"),
        ut("second line"),
    ]
    fixed = _fix_alternation(messages)
    # The two user messages should be merged into one
    users = [m for m in fixed if m.get("role") == "user"]
    assert len(users) == 1, f"Expected 1 user after merge, got {len(users)}"
    assert "first line" in users[0].get("content", "") and "second line" in users[0].get("content", "")


def test_fix_alternation_pure_tool_result_user_then_prompt():
    """tool_result-only user + prompt user: must not merge (data loss path).

    This is the exact scenario the separator at compact.py line 441-442 was
    added for.  _fix_alternation should independently prevent this merge.
    """
    messages = [
        ma([tu("c5")]),
        mu([tr("c5")]),
        ut("COMPACT_PROMPT: please summarize..."),
    ]
    fixed = _fix_alternation(messages)
    oai = _to_openai_messages(None, fixed)
    tool_msgs = [m for m in oai if m.get("role") == "tool"]
    assert any(m.get("tool_call_id") == "c5" for m in tool_msgs), (
        "tool_result for c5 lost — consecutive user merge corrupted blocks"
    )
