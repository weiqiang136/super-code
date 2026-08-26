"""Token usage tracking.

只统计 token 用量（来自 API 返回的 usage），不计算费用——
模型定价表难以覆盖各种服务商，估算费用会造成误导（如 DeepSeek 被按
Claude 定价高估 ~10 倍，见 V3.3.10+ 之前的历史问题）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Usage data
# ---------------------------------------------------------------------------

@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


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
    """跨 API 调用累计 token 用量。"""

    def __init__(self) -> None:
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
    def has_usage(self) -> bool:
        """是否记录到过任何 API 用量。"""
        return bool(self._model_usage)

    def add_usage(self, model: str, usage: dict) -> None:
        """记录一次 API 调用的 token 用量。"""
        self._last_input_tokens = usage.get("input_tokens", 0)

        mu = self._model_usage.setdefault(model, ModelUsage())
        mu.input_tokens += usage.get("input_tokens", 0)
        mu.output_tokens += usage.get("output_tokens", 0)
        mu.cache_read_input_tokens += usage.get("cache_read_input_tokens", 0)
        mu.cache_creation_input_tokens += usage.get("cache_creation_input_tokens", 0)

    def format_cost(self) -> str:
        """生成可读的 token 用量摘要。"""
        if not self._model_usage:
            return "No API usage recorded."

        wall_s = time.monotonic() - self._wall_start
        lines: list[str] = [f"Total duration (wall): {_fmt_duration(wall_s)}"]
        lines.append("Usage by model:")

        max_name = max(len(m) for m in self._model_usage)
        for model, mu in sorted(self._model_usage.items()):
            parts = [f"{_fmt_tokens(mu.input_tokens)} input",
                     f"{_fmt_tokens(mu.output_tokens)} output"]
            if mu.cache_read_input_tokens:
                parts.append(f"{_fmt_tokens(mu.cache_read_input_tokens)} cache read")
            if mu.cache_creation_input_tokens:
                parts.append(f"{_fmt_tokens(mu.cache_creation_input_tokens)} cache write")
            lines.append(f"  {model.rjust(max_name)}:  {', '.join(parts)}")

        return "\n".join(lines)
