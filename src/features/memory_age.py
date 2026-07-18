"""记忆陈旧度（freshness）警告生成。

三个纯函数：

- memory_age_days(mtime_ms): 距今天数（floor，负值钳到 0）
- memory_freshness_text(mtime_ms): 普通文本警告，≤1 天返回空串
- memory_freshness_note(mtime_ms): 同上但包好 <system-reminder> 标签

设计动机：
    模型对 ISO 时间戳不敏感（"2026-04-15 写的"无法直接触发"可能过时"的推理），
    但对"X days ago"敏感。把陈旧度计算下沉到这里、并在注入点用自然语言描述，
    可以让模型自发去验证而不是把老记忆当作 live state。

可注入 now_ms：
    所有函数支持 `now_ms` 关键字参数；默认走 time.time() * 1000，
    测试可显式传入以避免时区 / 单测时钟漂移。
"""
from __future__ import annotations

import time

# 1 天的毫秒数。提到模块级常量便于阅读 / 测试断言。
_MS_PER_DAY = 86_400_000


def _now_ms() -> float:
    """统一获取当前毫秒时间戳。封装一层是为了让单测可以 monkeypatch 此函数。"""
    return time.time() * 1000.0


def memory_age_days(mtime_ms: float, now_ms: float | None = None) -> int:
    """返回 mtime 距今的整数天数（floor）。

    - 今天 → 0；昨天 → 1；47 天前 → 47
    - 未来时间 / 时钟漂移导致负数 → 钳到 0（绝不输出负值，避免下游格式串崩坏）
    """
    now = _now_ms() if now_ms is None else now_ms
    delta_days = int((now - mtime_ms) // _MS_PER_DAY)
    return max(0, delta_days)


def memory_freshness_text(mtime_ms: float, now_ms: float | None = None) -> str:
    """≤1 天（今天 / 昨天）→ 空字符串（避免给新鲜记忆贴噪音 warning）。
    ≥2 天 → 一段 plain-text 警告，**不带** <system-reminder> 包裹，供已自带包裹层的调用方使用。
    """
    d = memory_age_days(mtime_ms, now_ms=now_ms)
    if d <= 1:
        return ""
    return (
        f"This memory is {d} days old. "
        "Memories are point-in-time observations, not live state — "
        "claims about code behavior or file:line citations may be outdated. "
        "Verify against current code before asserting as fact."
    )


def memory_freshness_note(mtime_ms: float, now_ms: float | None = None) -> str:
    """≤1 天 → 空字符串；≥2 天 → 已包好 <system-reminder>...</system-reminder> 的单行片段，
    末尾带换行便于直接拼到 prompt 字符串里。

    Step 6 findRelevantMemories 注入路径会直接调用本函数。
    """
    text = memory_freshness_text(mtime_ms, now_ms=now_ms)
    if not text:
        return ""
    return f"<system-reminder>{text}</system-reminder>\n"
