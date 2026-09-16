"""压缩判定接入精确 input_tokens 的测试。

背景（2026-09-14 esigning 事故）：minified JS 会话估算 669K / 实际 ~800K，
底栏显示 80% 但 should_compact 只看估算 → 不触发。修复 = max(估算, 精确值)。
"""
import sys
sys.path.insert(0, "src")

from types import SimpleNamespace

from features.compact import (
    CompactService,
    effective_context_tokens,
    estimate_tokens,
    should_compact,
)
from features.cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# effective_context_tokens
# ---------------------------------------------------------------------------

def test_effective_no_precise_returns_estimate():
    """没调过 API（precise=None/0）→ 纯估算。"""
    assert effective_context_tokens(500_000, None, None) == 500_000
    assert effective_context_tokens(500_000, 0, None) == 500_000


def test_effective_precise_higher_wins():
    """精确 800K > 估算 669K → 取精确（esigning 事故场景）。"""
    assert effective_context_tokens(668_664, 800_000, None) == 800_000


def test_effective_estimate_higher_wins():
    """估算含本轮新输出、精确值滞后 → 取估算。"""
    assert effective_context_tokens(900_000, 800_000, None) == 900_000


def test_effective_inturn_precise_plus_growth():
    """轮内：精确基线 500K + 本轮估算增量 300K = 800K > 纯估算 550K。"""
    assert effective_context_tokens(550_000, 500_000, 250_000) == 800_000


def test_effective_inturn_growth_clamped_at_zero():
    """估算倒退（异常场景）→ 增量按 0 处理，不低于精确基线。"""
    assert effective_context_tokens(200_000, 500_000, 250_000) == 500_000


# ---------------------------------------------------------------------------
# should_compact
# ---------------------------------------------------------------------------

def test_should_compact_precise_above_threshold_triggers():
    """回归：估算远低于阈值，但精确值超 800K → 必须触发。"""
    messages = [{"role": "user", "content": "hi"}]
    assert should_compact(messages, model="glm-5.3-flash",
                          last_input_tokens=801_000) is True


def test_should_compact_both_below_no_trigger():
    messages = [{"role": "user", "content": "hi"}]
    assert should_compact(messages, model="glm-5.3-flash",
                          last_input_tokens=100_000) is False


def test_should_compact_estimate_only_still_works():
    """不传精确值 → 行为与旧版一致（估算 > 阈值即触发）。"""
    big = [{"role": "user", "content": "a" * 3_000_000}]  # 非 CJK 3M 字符 ÷ 3 = 1M 估算
    assert should_compact(big, model="glm-5.3-flash") is True


# ---------------------------------------------------------------------------
# 压缩后底栏覆盖（usage 缺失也不许跳过）
# ---------------------------------------------------------------------------

class _NoUsageClient:
    """摘要响应不带 usage 的假客户端（2026-09-15 esigning 实测场景）。"""

    def create(self, **kwargs):
        return SimpleNamespace(
            content=[{"type": "text",
                      "text": "<analysis>draft</analysis><summary>summary text</summary>"}],
            usage=None,
        )


def test_compact_overrides_bar_even_without_usage():
    """回归：摘要响应 usage=None → 底栏仍必须被覆盖为新历史估算。

    旧实现 set_last_input_tokens 被 `if cost_tracker and response.usage` 守卫
    连带跳过 → 压缩后底栏残留 92%（摘要请求的旧历史精确值），挂到下个成功
    请求才纠正。修复 = 覆盖挪出 usage 守卫（覆盖本身不依赖 usage）。
    """
    svc = CompactService(client=_NoUsageClient(), model="unknown-model",
                         cost_tracker=CostTracker())
    msgs = [{"role": "user", "content": "p" * 5000} for _ in range(20)]
    new_msgs, summary = svc.compact(msgs, "sys", skip_if_under_threshold=False)

    assert summary
    expected = estimate_tokens(new_msgs)
    assert expected > 0
    # 关键断言：底栏被覆盖为压缩后新历史的估算值，而不是残留旧值/0
    assert svc._cost_tracker.last_input_tokens == expected
    # usage 为空 → 不误记账
    assert svc._cost_tracker.totals()["input_tokens"] == 0
