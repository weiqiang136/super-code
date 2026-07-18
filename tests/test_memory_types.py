"""
test_memory_types.py — 验证 parse_memory_type 在合法 / 非法 / 异常输入下的行为。

设计要点：
- 合法类型必须原样返回（区分大小写）；
- 任意非法输入（未知字符串、None、非字符串）均返回 None，绝不抛异常。
"""
from features.memory_types import MEMORY_TYPES, parse_memory_type


def test_parse_all_known_types_roundtrip():
    """四种合法类型必须能原样解析出来。"""
    for t in MEMORY_TYPES:
        assert parse_memory_type(t) == t


def test_parse_unknown_string_returns_none():
    assert parse_memory_type("bogus") is None
    assert parse_memory_type("") is None
    # 大小写敏感匹配
    assert parse_memory_type("User") is None


def test_parse_non_string_returns_none():
    assert parse_memory_type(None) is None
    assert parse_memory_type(123) is None
    assert parse_memory_type(["user"]) is None
    assert parse_memory_type({"type": "user"}) is None
