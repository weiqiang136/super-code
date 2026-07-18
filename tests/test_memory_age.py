"""
test_memory_age.py — Step 4 记忆陈旧度警告。

覆盖：
- memory_age_days:
    * 今天 → 0
    * 昨天（24h 前）→ 1
    * 47 天前 → 47
    * 未来时间（now < mtime） → 钳到 0
    * 边界：刚好 24h - 1ms → 0；刚好 48h - 1ms → 1
- memory_freshness_text:
    * 今天 / 昨天 → ""（避免噪音）
    * 2 天前 → 含 "2 days old"
    * 47 天前 → 含 "47 days old" + "Verify against current code"
- memory_freshness_note:
    * ≤1 天 → ""
    * ≥2 天 → 被 <system-reminder>...</system-reminder> 标签包裹、带末尾换行
"""
from features.memory_age import (
    memory_age_days,
    memory_freshness_note,
    memory_freshness_text,
)

# 固定一个 now（毫秒）方便所有测试断言。10 亿 ms ≈ 1970-01-12，不与真实时钟交互。
_NOW_MS = 10_000_000_000.0
_DAY_MS = 86_400_000


# ---------------------------------------------------------------------------
# memory_age_days
# ---------------------------------------------------------------------------

def test_age_today_is_zero():
    assert memory_age_days(_NOW_MS, now_ms=_NOW_MS) == 0


def test_age_yesterday_is_one():
    assert memory_age_days(_NOW_MS - _DAY_MS, now_ms=_NOW_MS) == 1


def test_age_47_days():
    assert memory_age_days(_NOW_MS - 47 * _DAY_MS, now_ms=_NOW_MS) == 47


def test_age_future_clamped_to_zero():
    """未来 mtime（如时钟回拨 / NTP 漂移）必须钳到 0，不能输出负数。"""
    assert memory_age_days(_NOW_MS + 5 * _DAY_MS, now_ms=_NOW_MS) == 0


def test_age_just_under_24h_is_zero():
    # 还差 1 ms 才满 24 小时
    assert memory_age_days(_NOW_MS - (_DAY_MS - 1), now_ms=_NOW_MS) == 0


def test_age_just_under_48h_is_one():
    assert memory_age_days(_NOW_MS - (2 * _DAY_MS - 1), now_ms=_NOW_MS) == 1


def test_age_default_now_uses_clock(monkeypatch):
    """不传 now_ms → 走 time.time()*1000；这里 monkeypatch _now_ms 验证默认路径连通。"""
    monkeypatch.setattr("features.memory_age._now_ms", lambda: _NOW_MS)
    assert memory_age_days(_NOW_MS - 3 * _DAY_MS) == 3


# ---------------------------------------------------------------------------
# memory_freshness_text
# ---------------------------------------------------------------------------

def test_freshness_text_today_empty():
    assert memory_freshness_text(_NOW_MS, now_ms=_NOW_MS) == ""


def test_freshness_text_yesterday_empty():
    """1 天也算"新"，避免给昨天写的记忆贴噪音 warning。"""
    assert memory_freshness_text(_NOW_MS - _DAY_MS, now_ms=_NOW_MS) == ""


def test_freshness_text_two_days_warns():
    text = memory_freshness_text(_NOW_MS - 2 * _DAY_MS, now_ms=_NOW_MS)
    assert "2 days old" in text


def test_freshness_text_47_days_full_warning():
    text = memory_freshness_text(_NOW_MS - 47 * _DAY_MS, now_ms=_NOW_MS)
    assert "47 days old" in text
    assert "Verify against current code" in text
    # 不应自带 <system-reminder> 包裹（那是 note 的职责）
    assert "<system-reminder>" not in text


# ---------------------------------------------------------------------------
# memory_freshness_note
# ---------------------------------------------------------------------------

def test_freshness_note_today_empty():
    assert memory_freshness_note(_NOW_MS, now_ms=_NOW_MS) == ""


def test_freshness_note_old_has_system_reminder_wrapper():
    note = memory_freshness_note(_NOW_MS - 10 * _DAY_MS, now_ms=_NOW_MS)
    assert note.startswith("<system-reminder>")
    # 末尾形如 "...</system-reminder>\n"
    assert note.endswith("</system-reminder>\n")
    assert "10 days old" in note
