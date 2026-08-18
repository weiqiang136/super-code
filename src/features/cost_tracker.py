"""Token usage and cost tracking."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Pricing per million tokens ($/MTok)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PricingTier:
    input: float
    output: float
    cache_write: float = 0.0
    cache_read: float = 0.0


# OpenAI 模型定价（按 $/MTok）
_TIER_GPT4O = _PricingTier(input=2.5, output=10.0)
_TIER_GPT4O_MINI = _PricingTier(input=0.15, output=0.60)
_TIER_GPT4_TURBO = _PricingTier(input=10.0, output=30.0)

# Claude 模型定价
_TIER_3_15 = _PricingTier(input=3.0, output=15.0, cache_write=3.75, cache_read=0.30)
_TIER_5_25 = _PricingTier(input=5.0, output=25.0, cache_write=6.25, cache_read=0.50)
_TIER_HAIKU_45 = _PricingTier(input=1.0, output=5.0, cache_write=1.25, cache_read=0.10)

# 前缀匹配，先匹配先赢
_MODEL_PRICING: list[tuple[str, _PricingTier]] = [
    ("gpt-4o-mini", _TIER_GPT4O_MINI),
    ("gpt-4o", _TIER_GPT4O),
    ("gpt-4-turbo", _TIER_GPT4_TURBO),
    ("claude-haiku-4-5", _TIER_HAIKU_45),
    ("claude-opus-4", _TIER_5_25),
    ("claude-sonnet", _TIER_3_15),
    ("claude-3-5-sonnet", _TIER_3_15),
    ("claude-3-7-sonnet", _TIER_3_15),
]

_DEFAULT_TIER = _TIER_3_15


def _tier_for_model(model: str) -> _PricingTier | None:
    model_lower = model.lower()
    for prefix, tier in _MODEL_PRICING:
        if prefix in model_lower:
            return tier
    # 未知 OpenAI 模型不计费
    if model_lower.startswith(("gpt-", "o1", "o3", "o4")):
        return None
    return _DEFAULT_TIER


# ---------------------------------------------------------------------------
# Usage data
# ---------------------------------------------------------------------------

@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    pricing_known: bool = True


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_tokens(n: int) -> str:
    """格式化 token 数，使用 k/m 后缀。"""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}m" if v != int(v) else f"{int(v)}m"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.1f}k" if v != int(v) else f"{int(v)}k"
    return str(n)


def _fmt_duration(seconds: float) -> str:
    """格式化秒数为 'Xh Ym Zs' 格式。"""
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class CostTracker:
    """跨 API 调用累计 token 用量和费用。"""

    def __init__(self) -> None:
        self._total_cost_usd: float = 0.0
        self._model_usage: dict[str, ModelUsage] = {}
        self._wall_start: float = time.monotonic()
        self._last_input_tokens: int = 0

    @property
    def last_input_tokens(self) -> int:
        """最近一次 API 调用的 input_tokens（反映当前上下文大小）。"""
        return self._last_input_tokens

    def set_last_input_tokens(self, n: int) -> None:
        """手动覆盖 last_input_tokens（压缩成功后设为压缩后新历史的估算值）。

        压缩摘要请求的 input 是压缩前的旧历史，add_usage 会把它记成旧值，
        导致底部栏 ctx 占用率虚高；压缩后实际上下文是"摘要 + 尾部"，
        先估算覆盖，下一轮对话 API 调用后自动纠正为精确值。
        """
        self._last_input_tokens = n

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @staticmethod
    def calculate_cost(model: str, usage: dict) -> float:
        """计算单次 API 调用的费用（USD）。"""
        tier = _tier_for_model(model)
        if tier is None:
            return 0.0
        cost = (
            usage.get("input_tokens", 0) * tier.input
            + usage.get("output_tokens", 0) * tier.output
            + usage.get("cache_read_input_tokens", 0) * tier.cache_read
            + usage.get("cache_creation_input_tokens", 0) * tier.cache_write
        ) / 1_000_000
        return cost

    def add_usage(self, model: str, usage: dict) -> float:
        """记录 token 用量，返回本次调用费用。"""
        cost = self.calculate_cost(model, usage)
        self._total_cost_usd += cost
        self._last_input_tokens = usage.get("input_tokens", 0)

        mu = self._model_usage.setdefault(model, ModelUsage())
        mu.input_tokens += usage.get("input_tokens", 0)
        mu.output_tokens += usage.get("output_tokens", 0)
        mu.cache_read_input_tokens += usage.get("cache_read_input_tokens", 0)
        mu.cache_creation_input_tokens += usage.get("cache_creation_input_tokens", 0)
        mu.cost_usd += cost
        if _tier_for_model(model) is None:
            mu.pricing_known = False
        return cost

    def format_cost(self) -> str:
        """生成人类可读的费用摘要。"""
        if not self._model_usage:
            return "No API usage recorded."

        wall_s = time.monotonic() - self._wall_start
        unknown = any(not mu.pricing_known for mu in self._model_usage.values())
        lines: list[str] = [f"Total cost:            ${self._total_cost_usd:.4f}"]
        if unknown:
            lines.append("Pricing note:          Costs may be inaccurate (unknown model pricing)")
        lines.append(f"Total duration (wall): {_fmt_duration(wall_s)}")
        lines.append("Usage by model:")

        max_name = max(len(m) for m in self._model_usage)
        for model, mu in sorted(self._model_usage.items()):
            parts = [f"{_fmt_tokens(mu.input_tokens)} input",
                     f"{_fmt_tokens(mu.output_tokens)} output"]
            if mu.cache_read_input_tokens:
                parts.append(f"{_fmt_tokens(mu.cache_read_input_tokens)} cache read")
            if mu.cache_creation_input_tokens:
                parts.append(f"{_fmt_tokens(mu.cache_creation_input_tokens)} cache write")
            detail = ", ".join(parts)
            if not mu.pricing_known:
                detail += ", pricing unavailable"
            lines.append(f"  {model.rjust(max_name)}:  {detail} (${mu.cost_usd:.4f})")

        return "\n".join(lines)
