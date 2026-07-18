"""
test_memory_prompt.py — Step 4 系统提示重构后的 snapshot 测试。

只断言**关键句子**（不做整字符串比对），以保护未来段落顺序 / 措辞微调
不会引发脆弱失败，同时确保两段新增段落（WHEN_TO_ACCESS / TRUSTING_RECALL）
真的进入了系统提示。
"""
from pathlib import Path

from features.memory import (
    TRUSTING_RECALL_SECTION,
    TYPES_SECTION,
    WHAT_NOT_TO_SAVE_SECTION,
    WHEN_TO_ACCESS_SECTION,
    build_memory_system_section,
)


def test_section_contains_all_modular_blocks(tmp_path: Path):
    """所有静态段落必须出现在最终 prompt 中。"""
    out = build_memory_system_section(tmp_path)

    assert TYPES_SECTION in out
    assert WHAT_NOT_TO_SAVE_SECTION in out
    assert WHEN_TO_ACCESS_SECTION in out
    assert TRUSTING_RECALL_SECTION in out


def test_section_contains_trusting_recall_keywords(tmp_path: Path):
    """新增段落必须包含 eval-validated 的关键触发短语。"""
    out = build_memory_system_section(tmp_path)

    assert "## Before recommending from memory" in out
    assert "verify first" in out
    # drift caveat 关键句
    assert "stale over time" in out


def test_section_contains_memory_dir_path(tmp_path: Path):
    """memory_dir 路径必须被正确插值到序章和 Option B 两处。"""
    out = build_memory_system_section(tmp_path)
    assert str(tmp_path) in out


def test_section_empty_memory_fallback(tmp_path: Path):
    """目录下无 MEMORY.md → 出现 'No memories consolidated yet'。"""
    out = build_memory_system_section(tmp_path)
    assert "No memories consolidated yet." in out
    assert "## Current Memory Index" not in out


def test_section_with_memory_index(tmp_path: Path):
    """存在 MEMORY.md → 内容被作为索引段落附加。"""
    (tmp_path / "MEMORY.md").write_text(
        "- [Hooks](hooks.md) — never mock the DB in tests\n", encoding="utf-8"
    )
    out = build_memory_system_section(tmp_path)
    assert "## Current Memory Index (MEMORY.md)" in out
    assert "never mock the DB in tests" in out


def test_section_preserves_legacy_blocks(tmp_path: Path):
    """旧版关键句仍然保留，确保改造没有删掉用户已经习惯的指引。"""
    out = build_memory_system_section(tmp_path)
    assert "# Auto Memory" in out
    assert "## Types of memory" in out
    assert "## What NOT to save" in out
    assert "## How to save memories" in out
    assert "## Slash commands" in out
