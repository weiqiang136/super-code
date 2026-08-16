"""Tests for tool-result pruning (剪枝) in the compact pipeline.

Covers:
- _prune_text: threshold boundary behavior, head/tail/marker preservation
- prune_tool_results: skips non-tool_result, preserves other blocks,
  does not mutate input messages, stats correctness
- CompactService.compact integration: prune-only skip path (no LLM call)
  vs prune-then-summary path (LLM called with pruned input)
- Pruning preserves tool_result → role:"tool" mapping for the OpenAI API
"""
import sys
sys.path.insert(0, "src")
from types import SimpleNamespace

import pytest

from features.compact import (
    PRUNE_MARKER,
    PRUNE_THRESHOLD_CHARS,
    CompactService,
    _prune_text,
    prune_tool_results,
)
from core.llm import _to_openai_messages


# ---------------------------------------------------------------------------
# _prune_text
# ---------------------------------------------------------------------------

def test_prune_text_below_threshold_returns_none():
    """长度不超过阈值 → 不剪（返回 None）。"""
    text = "x" * PRUNE_THRESHOLD_CHARS
    assert _prune_text(text) is None


def test_prune_text_above_threshold_keeps_head_tail_marker():
    """超过阈值 → 保留头尾 + 剪枝标记，总长 = head + marker + tail。"""
    text = "H" * 5000 + "M" * 10000 + "T" * 2000
    pruned = _prune_text(text)
    assert pruned is not None
    assert pruned.startswith("H" * 4096), "head 部分应保留 4096 字符"
    assert pruned.endswith("T" * 1024), "tail 部分应保留 1024 字符"
    assert PRUNE_MARKER in pruned, "中间段应被剪枝标记替换"
    assert "M" * 10000 not in pruned, "中间段应被裁掉"


# ---------------------------------------------------------------------------
# prune_tool_results
# ---------------------------------------------------------------------------

def _tool_result_user(tid, text, extra=None):
    block = {"type": "tool_result", "tool_use_id": tid, "content": text}
    if extra:
        block.update(extra)
    return {"role": "user", "content": [block]}


def _tool_use_assistant(tid):
    return {"role": "assistant", "content": [
        {"type": "tool_use", "id": tid, "name": "t", "input": {}}
    ]}


def test_prune_skips_messages_without_tool_result():
    """不含 tool_result 的消息（纯文本 / 纯 assistant）原样保留、同一对象引用。"""
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    out, stats = prune_tool_results(msgs)
    assert stats["pruned"] == 0
    assert out == msgs
    assert out[0] is msgs[0], "未剪的消息应保持原对象引用（零拷贝）"


def test_prune_trims_only_over_threshold_tool_result():
    """超阈值 tool_result 被剪，未超阈值的保留原样。"""
    big = "B" * (PRUNE_THRESHOLD_CHARS + 100)
    small = "s" * 100
    msgs = [
        _tool_use_assistant("c1"),
        _tool_result_user("c1", big),
        _tool_use_assistant("c2"),
        _tool_result_user("c2", small),
    ]
    out, stats = prune_tool_results(msgs)
    assert stats["pruned"] == 1
    assert PRUNE_MARKER in out[1]["content"][0]["content"]
    assert out[3] is msgs[3], "未超阈值的消息应保持原对象引用"


def test_prune_preserves_other_blocks_and_fields():
    """同一条消息里非 tool_result block 保留；tool_result 的附带字段不丢。"""
    big = "B" * (PRUNE_THRESHOLD_CHARS + 100)
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "note"},
            {"type": "tool_result", "tool_use_id": "c1", "content": big,
             "is_error": True, "metadata": {"k": "v"}},
        ],
    }]
    out, stats = prune_tool_results(msgs)
    assert stats["pruned"] == 1
    blocks = out[0]["content"]
    assert blocks[0] == {"type": "text", "text": "note"}, "非 tool_result block 原样保留"
    pruned_block = blocks[1]
    assert pruned_block["is_error"] is True, "is_error 字段应保留"
    assert pruned_block["metadata"] == {"k": "v"}, "metadata 字段应保留"
    assert pruned_block["tool_use_id"] == "c1"


def test_prune_does_not_mutate_input():
    """剪枝不污染调用方原始消息（重建新 dict，而非原地修改）。"""
    big = "B" * (PRUNE_THRESHOLD_CHARS + 100)
    msgs = [_tool_result_user("c1", big)]
    original_content = msgs[0]["content"][0]["content"]
    prune_tool_results(msgs)
    assert msgs[0]["content"][0]["content"] == original_content, "输入消息不应被原地修改"


def test_prune_stats_correct():
    """统计：剪了几条 + 省了多少字符。"""
    big1 = "A" * (PRUNE_THRESHOLD_CHARS + 500)
    big2 = "B" * (PRUNE_THRESHOLD_CHARS + 1000)
    msgs = [_tool_result_user("c1", big1), _tool_result_user("c2", big2)]
    out, stats = prune_tool_results(msgs)
    assert stats["pruned"] == 2
    expected = (len(big1) - len(out[0]["content"][0]["content"])) \
        + (len(big2) - len(out[1]["content"][0]["content"]))
    assert stats["chars_removed"] == expected


# ---------------------------------------------------------------------------
# CompactService.compact integration
# ---------------------------------------------------------------------------

class _DummyClient:
    """记录 create 调用次数的假 LLM 客户端；返回固定摘要。"""

    def __init__(self):
        self.calls = 0
        self.last_messages = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_messages = kwargs.get("messages")
        return SimpleNamespace(content=[
            {"type": "text", "text": "<analysis>draft</analysis><summary>summary text</summary>"}
        ])


def _big_tool_pair(tid, size):
    """构造 assistant(tool_use) + user(超大 tool_result) 配对。"""
    return [
        _tool_use_assistant(tid),
        _tool_result_user(tid, "R" * size),
    ]


def _pad_messages(count, size=20000):
    """生成 count 条纯文本 user 消息撑大上下文。"""
    return [{"role": "user", "content": "p" * size} for _ in range(count)]


def test_empty_summary_retries_then_succeeds():
    """摘要 LLM 第一次返回空 content（思考模型偶发），重试一次后成功。

    重试语义：不抛异常、正常返回压缩结果（调用 2 次）。
    """
    class _EmptyFirstClient(_DummyClient):
        def create(self, **kwargs):
            if self.calls == 0:
                self.calls += 1
                return SimpleNamespace(content=[])  # 第一次空 content
            return super().create(**kwargs)          # 第二次由超类计数并返回正常摘要

    client = _EmptyFirstClient()
    svc = CompactService(client, model="unknown-model")
    msgs = (
        _pad_messages(5, size=20000)
        + [{"role": "assistant", "content": "recent_a"},
           {"role": "user", "content": "recent_u"}]
    )
    new_msgs, summary = svc.compact(msgs, "sys")
    assert client.calls == 2, f"空摘要应重试一次，实际调用 {client.calls} 次"
    assert "summary text" in summary, "重试后应拿到正常摘要"


def test_empty_summary_twice_raises_without_result():
    """摘要 LLM 连续返回空 content → 抛 EmptySummaryError，而不是拿空摘要落盘。

    这是 2026-08-16 事故的回归测试：以前空摘要会变成
    "(compact produced empty summary)" 边界行覆盖历史（信息全丢）；
    现在必须抛异常，由调用方中止压缩、历史保持原样。
    """
    class _AlwaysEmptyClient(_DummyClient):
        def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(content=[])  # 永远空

    client = _AlwaysEmptyClient()
    svc = CompactService(client, model="unknown-model")
    msgs = (
        _pad_messages(5, size=20000)
        + [{"role": "assistant", "content": "recent_a"},
           {"role": "user", "content": "recent_u"}]
    )
    from features.compact import EmptySummaryError
    with pytest.raises(EmptySummaryError):
        svc.compact(msgs, "sys")
    assert client.calls == 2, f"重试 1 次后应抛异常，实际调用 {client.calls} 次"


def test_compact_skips_summary_when_prune_under_threshold():
    """剪枝后低于自动压缩阈值 → 跳过 LLM 摘要（client 零调用）、无 boundary。"""
    client = _DummyClient()
    svc = CompactService(client, model="unknown-model")  # 默认窗口 128K → 阈值 ~102K
    msgs = (
        _pad_messages(5, size=20000)          # ~2.5万 token
        + _big_tool_pair("c1", 90000)         # 超大输出：剪前 ~2.2万 token，剪后 ~0.5万
        + [{"role": "assistant", "content": "recent_a"},
           {"role": "user", "content": "recent_u"}]
    )
    new_msgs, summary = svc.compact(msgs, "sys", skip_if_under_threshold=True)
    assert client.calls == 0, "剪枝后已低于阈值，不应调用摘要 LLM"
    assert "prune only" in summary
    assert not any(
        isinstance(m.get("content", ""), str) and "COMPACT_BOUNDARY" in m["content"]
        for m in new_msgs
    ), "跳过摘要路径不应插入压缩边界 marker"
    assert len(new_msgs) == len(msgs), "剪枝不改变消息条数"


def test_compact_prune_then_summary_when_still_over():
    """剪枝后仍超阈值 → 走 LLM 摘要；recent 里的超大 tool_result 已剪（瘦身生效）。"""
    client = _DummyClient()
    svc = CompactService(client, model="unknown-model")
    msgs = (
        _pad_messages(20, size=30000)         # ~15万 token，剪枝后仍超 102K 阈值
        + _big_tool_pair("c1", 90000)
        + [{"role": "assistant", "content": "recent_a"},
           {"role": "user", "content": "recent_u"}]
    )
    new_msgs, summary = svc.compact(msgs, "sys", skip_if_under_threshold=True)
    assert client.calls == 1, "剪枝后仍超阈值，应调用摘要 LLM"
    assert "summary text" in summary
    # 返回消息（frozen + boundary + ack + pruned_recent）应包含剪枝标记：
    # 超大 tool_result 在 recent 里（_split_recent 按 token 保留尾部），剪枝后
    # 模型后续看到的上下文已瘦身
    assert any(
        PRUNE_MARKER in _flatten_text(m.get("content", "")) for m in new_msgs
    ), "返回消息的 recent 部分应包含剪枝标记"
    # 返回结果应包含压缩边界 marker（走的是摘要路径）
    assert any(
        isinstance(m.get("content", ""), str) and "COMPACT_BOUNDARY" in m["content"]
        for m in new_msgs
    ), "摘要路径应插入压缩边界 marker"


def test_compact_skip_flag_false_always_summarizes():
    """skip_if_under_threshold=False（旧行为）：剪枝后即使低于阈值也走摘要。"""
    client = _DummyClient()
    svc = CompactService(client, model="unknown-model")
    msgs = (
        _pad_messages(5, size=20000)
        + _big_tool_pair("c1", 90000)
        + [{"role": "assistant", "content": "recent_a"},
           {"role": "user", "content": "recent_u"}]
    )
    new_msgs, summary = svc.compact(msgs, "sys", skip_if_under_threshold=False)
    assert client.calls == 1, "False 时应保持旧行为：永远走摘要"
    assert "summary text" in summary


def test_prune_preserves_openai_tool_mapping():
    """剪枝后 _to_openai_messages 仍把 tool_result 正确转成 role:"tool" 消息。"""
    big = "B" * (PRUNE_THRESHOLD_CHARS + 100)
    msgs = [_tool_use_assistant("c9"), _tool_result_user("c9", big)]
    pruned, _ = prune_tool_results(msgs)
    oai = _to_openai_messages(None, pruned)
    tool_msgs = [m for m in oai if m.get("role") == "tool"]
    assert len(tool_msgs) == 1, f"剪枝后应仍有 1 条 role:tool 消息，got {len(tool_msgs)}"
    assert tool_msgs[0]["tool_call_id"] == "c9"
    assert PRUNE_MARKER in tool_msgs[0]["content"], "tool 消息内容应为剪枝后的文本"


def test_recent_uses_higher_threshold():
    """recent 用高阈值（32768）：10K 字符输出在 history 被剪、在 recent 保留完整。

    这是"recent 可能正被当前任务依赖"的防护：中等输出（>8192 但 <32768）
    在 recent 中不剪，只有真正的巨型输出（>32768）才剪。
    """
    from features.compact import PRUNE_RECENT_THRESHOLD_CHARS
    client = _DummyClient()
    svc = CompactService(client, model="unknown-model")
    # 布局：tool 对1 进 history（前面大量 padding 后紧跟），tool 对2 进 recent
    msgs = (
        _pad_messages(5, size=20000)
        + _big_tool_pair("c_hist", 10000)      # history 里 10K → 默认阈值 8192 → 剪
        + _pad_messages(5, size=20000)
        + _big_tool_pair("c_recent", 10000)    # recent 里 10K → 高阈值 32768 → 不剪
        + [{"role": "assistant", "content": "recent_a"},
           {"role": "user", "content": "recent_u"}]
    )
    new_msgs, summary = svc.compact(msgs, "sys", skip_if_under_threshold=True)
    assert client.calls == 0, "history 剪枝后应已低于阈值，跳过摘要"
    assert 10000 < PRUNE_RECENT_THRESHOLD_CHARS, "测试前提：10K 应低于 recent 高阈值"
    hist_pruned = [m for m in new_msgs if PRUNE_MARKER in _flatten_text(m.get("content", ""))]
    assert len(hist_pruned) == 1, f"history 里的 10K 输出应被剪（1 条），got {len(hist_pruned)}"
    # recent 里的 10K 输出应保留完整：找 tool_use_id == c_recent 的 tool_result
    recent_block = None
    for m in new_msgs:
        content = m.get("content", "")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result" \
                        and b.get("tool_use_id") == "c_recent":
                    recent_block = b
    assert recent_block is not None, "recent 的 tool_result 应存在"
    assert PRUNE_MARKER not in recent_block["content"], "recent 的 10K 输出不应被剪"
    assert recent_block["content"] == "R" * 10000, "recent 的 10K 输出应保留完整原文"


def _flatten_text(content):
    """提取消息 content 的纯文本（兼容 str 与 block list）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text", ""))
                c = b.get("content", "")
                if isinstance(c, str):
                    parts.append(c)
        return "".join(parts)
    return ""
