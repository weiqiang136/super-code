"""记忆类型枚举与 frontmatter `type` 字段解析。

将四种合法类型 (`user / feedback / project / reference`) 收敛在一处，
给后续的扫描器 (memory_scan)、相关性精选 (find_relevant_memories)、
抽取 (extract_memories) 共用，避免到处散落 magic string。

设计要点：
- 未知 / 缺失 `type` 字段 → 返回 None，而不是抛异常。盘上的旧记忆
  几乎都没有 `type` 字段，强校验会让它们整体从扫描结果中消失。
- 区分大小写匹配。
"""
from __future__ import annotations

from typing import Literal

# 类型别名：受限于 Literal 的四种合法值
MemoryType = Literal["user", "feedback", "project", "reference"]

# 元组形式便于运行时遍历（Literal 自身不可迭代）
MEMORY_TYPES: tuple[MemoryType, ...] = ("user", "feedback", "project", "reference")


def parse_memory_type(raw: object) -> MemoryType | None:
    """把任意 frontmatter 字段值解析为 MemoryType，未知 / 非法返回 None。

    传入 object 而非 str，是因为调用方拿到的通常是 dict.get() 的结果，
    类型未知；在这里集中做类型守卫比每个调用点 isinstance 干净。
    """
    if not isinstance(raw, str):
        return None
    if raw in MEMORY_TYPES:
        # mypy 无法从 `in MEMORY_TYPES` 收窄到 Literal，需 cast
        return raw  # type: ignore[return-value]
    return None
