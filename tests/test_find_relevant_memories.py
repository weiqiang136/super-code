"""
test_find_relevant_memories.py — Step 6 相关性精选注入。

策略：用一个**最简化的 FakeLLM 桩**替代真实 OpenAI 调用，集中验证：
- side-query 的 JSON 解析与白名单过滤
- 失败时静默返回空串 / 空列表
- 记忆数过少时跳过 side-query
- prefix 文本格式（摘要注入 + freshness 警告拼接）
- 会话级节流（重复查询窗口内跳过 LLM 调用）

每个测试都构造一个独立 tmp_path 目录，避免相互污染。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from features.find_relevant_memories import (
    MAX_SELECTED,
    MIN_MEMORIES_FOR_SIDE_QUERY,
    build_relevant_memories_prefix,
    find_relevant_memories,
    reset_relevant_memories_cache,
)
import features.find_relevant_memories as frm


# 所有用例之间互相隔离缓存与节流状态。性能优化加缓存后，若不 reset，前一个测试
# 的 prefix 会被相似 query 命中，造成偶发"用了上一个测试的 fake LLM 输出"的幽灵问题。
@pytest.fixture(autouse=True)
def _reset_cache():
    reset_relevant_memories_cache()
    yield
    reset_relevant_memories_cache()


def _clear_throttle(monkeypatch):
    """模拟节流窗口过期：把上次 side-query 时刻置 None（仅清节流，不动缓存）。"""
    monkeypatch.setattr(frm, "_last_side_query_at", None)


# 旧测试统一用这个 query —— 长度 >= _SKIP_LOOKUP_MIN_CHARS(10) 才能进入正常路径
_LONG_Q = "tell me about the relevant memories please"



# ---------------------------------------------------------------------------
# FakeLLM 桩
# ---------------------------------------------------------------------------

@dataclass
class _FakeReply:
    """与 StreamMessage 鸭子兼容：只看 .content list-of-text-blocks。"""
    content: list[dict]


class FakeLLM:
    """记录调用 + 按 fixture 返回预设答复。

    支持三种 mode：
        - "names": 返回 {"selected_memories": names_list}
        - "raw":   返回 raw_text 原样（用于测试解析鲁棒性）
        - "raise": 调用即抛异常（用于测试静默降级）
    """

    def __init__(self, mode: str = "names", names: list[str] | None = None,
                 raw_text: str = ""):
        self.mode = mode
        self.names = names or []
        self.raw_text = raw_text
        self.calls: list[dict] = []

    def create(self, *, model: str, max_tokens: int, messages: list[dict],
               system: str | None = None, **_: Any) -> _FakeReply:
        self.calls.append({
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "system": system,
        })
        if self.mode == "raise":
            raise RuntimeError("simulated API failure")
        if self.mode == "raw":
            return _FakeReply(content=[{"type": "text", "text": self.raw_text}])
        # default: 返回正经 JSON
        import json as _json
        text = _json.dumps({"selected_memories": self.names})
        return _FakeReply(content=[{"type": "text", "text": text}])


def _write_memory(path: Path, name: str, description: str, type_: str = "feedback",
                  body: str = "body", mtime: float | None = None) -> Path:
    """快捷写一个带 frontmatter 的记忆文件。返回路径。"""
    fp = path / f"{name}.md"
    fp.write_text(
        f"---\nname: {name}\ndescription: {description}\ntype: {type_}\n---\n{body}\n",
        encoding="utf-8",
    )
    if mtime is not None:
        os.utime(fp, (mtime, mtime))
    return fp


# ---------------------------------------------------------------------------
# find_relevant_memories
# ---------------------------------------------------------------------------

def test_empty_query_returns_nothing(tmp_path: Path):
    _write_memory(tmp_path, "a", "desc")
    assert find_relevant_memories("   ", tmp_path, FakeLLM(), "x") == []


def test_missing_dir_returns_nothing(tmp_path: Path):
    assert find_relevant_memories("hi", tmp_path / "no", FakeLLM(), "x") == []


def test_single_memory_skips_side_query(tmp_path: Path):
    """记忆数 < MIN_MEMORIES_FOR_SIDE_QUERY → 全部返回，不调用 LLM。"""
    assert MIN_MEMORIES_FOR_SIDE_QUERY >= 2  # 不变量
    _write_memory(tmp_path, "only", "the only memory")

    fake = FakeLLM(mode="raise")  # 若被调到立即抛
    result = find_relevant_memories("anything", tmp_path, fake, "x")

    assert len(result) == 1
    assert result[0].filename == "only.md"
    assert fake.calls == []


def test_selects_by_filename_whitelist(tmp_path: Path):
    _write_memory(tmp_path, "real_a", "a desc")
    _write_memory(tmp_path, "real_b", "b desc")
    # 模型返回一个不存在的 filename → 必须被白名单过滤掉
    fake = FakeLLM(names=["real_a.md", "hallucinated.md"])
    out = find_relevant_memories("q", tmp_path, fake, "x")
    assert [m.filename for m in out] == ["real_a.md"]


def test_truncates_to_max_selected(tmp_path: Path):
    """side-query 返回 > MAX_SELECTED 个 → 截断。"""
    for i in range(MAX_SELECTED + 3):
        _write_memory(tmp_path, f"m{i}", f"desc{i}")
    names = [f"m{i}.md" for i in range(MAX_SELECTED + 3)]
    fake = FakeLLM(names=names)
    out = find_relevant_memories("q", tmp_path, fake, "x")
    assert len(out) == MAX_SELECTED


def test_dedup_in_selection(tmp_path: Path):
    _write_memory(tmp_path, "a", "x")
    _write_memory(tmp_path, "b", "y")
    fake = FakeLLM(names=["a.md", "a.md", "b.md"])
    out = find_relevant_memories("q", tmp_path, fake, "x")
    assert [m.filename for m in out] == ["a.md", "b.md"]


def test_llm_exception_returns_empty(tmp_path: Path):
    _write_memory(tmp_path, "a", "x")
    _write_memory(tmp_path, "b", "y")
    fake = FakeLLM(mode="raise")
    assert find_relevant_memories("q", tmp_path, fake, "x") == []


def test_invalid_json_returns_empty(tmp_path: Path):
    _write_memory(tmp_path, "a", "x")
    _write_memory(tmp_path, "b", "y")
    fake = FakeLLM(mode="raw", raw_text="not json at all")
    assert find_relevant_memories("q", tmp_path, fake, "x") == []


def test_handles_markdown_fence_wrapping(tmp_path: Path):
    """有些模型偏要包 ```json...``` —— 应能正常解析。"""
    _write_memory(tmp_path, "a", "x")
    _write_memory(tmp_path, "b", "y")
    fake = FakeLLM(
        mode="raw",
        raw_text='```json\n{"selected_memories": ["a.md"]}\n```',
    )
    out = find_relevant_memories("q", tmp_path, fake, "x")
    assert [m.filename for m in out] == ["a.md"]


def test_unexpected_json_shape_returns_empty(tmp_path: Path):
    _write_memory(tmp_path, "a", "x")
    _write_memory(tmp_path, "b", "y")
    # selected_memories 是个 dict 而不是 list → 跳过
    fake = FakeLLM(mode="raw", raw_text='{"selected_memories": {"k": "v"}}')
    assert find_relevant_memories("q", tmp_path, fake, "x") == []


# ---------------------------------------------------------------------------
# build_relevant_memories_prefix
# ---------------------------------------------------------------------------

def test_prefix_empty_when_no_match(tmp_path: Path):
    fake = FakeLLM(mode="raise")
    assert build_relevant_memories_prefix("q", tmp_path, fake, "x") == ""


def test_prefix_format_with_fresh_memory(tmp_path: Path):
    """新鲜记忆不带 freshness 警告；注入的是摘要（description）而非正文。"""
    _write_memory(tmp_path, "a", "desc-a", body="this is body A")
    _write_memory(tmp_path, "b", "desc-b", body="this is body B")
    fake = FakeLLM(names=["a.md"])

    prefix = build_relevant_memories_prefix(_LONG_Q, tmp_path, fake, "x")

    assert prefix.startswith("<system-reminder>")
    assert prefix.endswith("\n\n")
    assert "Relevant memories selected for this turn (1)" in prefix
    assert "## a.md" in prefix
    # 注入摘要（frontmatter description），不注入正文
    assert "desc-a" in prefix
    assert "this is body A" not in prefix
    # 给出记忆目录路径，便于模型按需 Read
    assert f"Memory directory: {tmp_path}" in prefix
    # 新鲜 → 不应带 freshness 警告子块
    assert "days old" not in prefix
    # 未被选中的不出现
    assert "## b.md" not in prefix


def test_prefix_includes_freshness_for_old_memory(tmp_path: Path):
    """≥2 天的记忆必须带 <system-reminder>...days old...</system-reminder>。"""
    _write_memory(tmp_path, "old", "desc")
    _write_memory(tmp_path, "padding", "desc2")
    # 把 old.md 的 mtime 设到 30 天前
    import time
    old_mtime = time.time() - 30 * 86400
    os.utime(tmp_path / "old.md", (old_mtime, old_mtime))

    fake = FakeLLM(names=["old.md"])
    prefix = build_relevant_memories_prefix(_LONG_Q, tmp_path, fake, "x")

    assert "days old" in prefix
    assert "Verify against current code" in prefix


def test_prefix_skips_file_missing_from_scan(tmp_path: Path):
    """已被 scan 排除的文件（如被删除）不应让整个 prefix 崩，只该跳过这一条。"""
    _write_memory(tmp_path, "good", "g")
    _write_memory(tmp_path, "bad", "b")
    # 把 bad.md 删掉 → scan 阶段就排除，selected 里只剩 good.md
    (tmp_path / "bad.md").unlink()

    fake = FakeLLM(names=["good.md", "bad.md"])
    prefix = build_relevant_memories_prefix(_LONG_Q, tmp_path, fake, "x")
    assert "## good.md" in prefix
    assert "## bad.md" not in prefix


def test_prefix_excludes_memory_md_even_if_selected(tmp_path: Path):
    """scan 已经排除 MEMORY.md；即使有人手动用 selected 注入也要保险拦截。

    构造方式：写两个普通 memory 让 scan 通过，模型 hallucinate 一个 MEMORY.md。
    白名单过滤已经会过滤掉它，这里再次断言。
    """
    _write_memory(tmp_path, "a", "x")
    _write_memory(tmp_path, "b", "y")
    (tmp_path / "MEMORY.md").write_text("- index\n", encoding="utf-8")
    fake = FakeLLM(names=["MEMORY.md", "a.md"])
    prefix = build_relevant_memories_prefix(_LONG_Q, tmp_path, fake, "x")
    assert "## MEMORY.md" not in prefix
    assert "## a.md" in prefix


def test_prefix_uses_placeholder_when_no_description(tmp_path: Path):
    """frontmatter 缺 description 时降级占位，不崩。"""
    (tmp_path / "a.md").write_text("---\ntype: feedback\n---\nbody\n", encoding="utf-8")
    _write_memory(tmp_path, "b", "y")
    fake = FakeLLM(names=["a.md"])
    prefix = build_relevant_memories_prefix(_LONG_Q, tmp_path, fake, "x")
    assert "## a.md" in prefix
    assert "read the file for details" in prefix


# ---------------------------------------------------------------------------
# side-query call shape sanity
# ---------------------------------------------------------------------------

def test_side_query_passes_query_and_manifest(tmp_path: Path):
    _write_memory(tmp_path, "a", "desc-a")
    _write_memory(tmp_path, "b", "desc-b")
    fake = FakeLLM(names=[])
    find_relevant_memories("how do I run the tests?", tmp_path, fake, "test-model")

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["model"] == "test-model"
    # 给 system 提示了选择规则
    assert "selecting memory files" in call["system"]
    # user msg 同时包含 query 与 manifest
    user_text = call["messages"][0]["content"]
    assert "how do I run the tests" in user_text
    assert "a.md" in user_text and "b.md" in user_text


# ---------------------------------------------------------------------------
# 性能优化：方案 3（跳过短输入）
# ---------------------------------------------------------------------------

class _SpyLLM(FakeLLM):
    """记录 create 调用次数，用于断言"被跳过 → 0 次调用"。"""
    pass


def test_skip_empty_query(tmp_path: Path):
    _write_memory(tmp_path, "a", "x")
    _write_memory(tmp_path, "b", "y")
    spy = _SpyLLM(names=["a.md"])
    assert build_relevant_memories_prefix("", tmp_path, spy, "m") == ""
    assert build_relevant_memories_prefix("   ", tmp_path, spy, "m") == ""
    assert spy.calls == []


def test_skip_short_query(tmp_path: Path):
    """< 10 字符的输入直接跳过，零 LLM 调用。"""
    _write_memory(tmp_path, "a", "x")
    _write_memory(tmp_path, "b", "y")
    spy = _SpyLLM(names=["a.md"])
    assert build_relevant_memories_prefix("继续", tmp_path, spy, "m") == ""
    assert build_relevant_memories_prefix("fix bug", tmp_path, spy, "m") == ""
    assert spy.calls == []


def test_skip_confirm_words_even_when_long_enough(tmp_path: Path):
    """全词命中确认词列表也跳过（即便长度过了 _SKIP_LOOKUP_MIN_CHARS）。"""
    _write_memory(tmp_path, "a", "x")
    _write_memory(tmp_path, "b", "y")
    spy = _SpyLLM(names=["a.md"])
    # "continue" 8 字符已小于 10，先被长度跳过；用一个更长且仍是单词的确认
    # 验证：长度 < 10 + 是确认词，照样跳过
    assert build_relevant_memories_prefix("continue", tmp_path, spy, "m") == ""
    assert spy.calls == []


def test_normal_long_query_not_skipped(tmp_path: Path):
    """长输入不应被跳过，应正常走 side-query。"""
    _write_memory(tmp_path, "a", "desc")
    _write_memory(tmp_path, "b", "desc2")
    spy = _SpyLLM(names=["a.md"])
    out = build_relevant_memories_prefix(_LONG_Q, tmp_path, spy, "m")
    assert out != ""
    assert len(spy.calls) == 1


# ---------------------------------------------------------------------------
# 性能优化：方案 1（连续提问缓存）
# ---------------------------------------------------------------------------

def test_cache_hit_for_similar_query(tmp_path: Path):
    """两次几乎相同的 query → 第二次命中缓存，不再调 LLM。"""
    _write_memory(tmp_path, "a", "desc")
    _write_memory(tmp_path, "b", "desc2")
    spy = _SpyLLM(names=["a.md"])

    out1 = build_relevant_memories_prefix(
        "How do I run the integration tests?", tmp_path, spy, "m")
    out2 = build_relevant_memories_prefix(
        "How do I run the integration tests now?", tmp_path, spy, "m")

    assert out1 != ""
    # 命中缓存：返回值复用、调用次数只 +1
    assert out1 == out2
    assert len(spy.calls) == 1


def test_cache_miss_for_different_topic(tmp_path: Path, monkeypatch):
    """两次完全不同话题的 query → 第二次必须重新 side-query（节流过期后）。"""
    _write_memory(tmp_path, "a", "desc")
    _write_memory(tmp_path, "b", "desc2")
    spy = _SpyLLM(names=["a.md"])

    build_relevant_memories_prefix(
        "How do I run the integration tests?", tmp_path, spy, "m")
    # 模拟节流窗口过期：否则不同话题的第二次调用会被节流拦下
    _clear_throttle(monkeypatch)
    build_relevant_memories_prefix(
        "What does the authentication middleware do exactly?", tmp_path, spy, "m")
    assert len(spy.calls) == 2


def test_cache_invalidated_by_different_memory_dir(tmp_path: Path, monkeypatch):
    """memory_dir 切换 → 缓存必须失效（避免跨项目拿错记忆）。"""
    dir_a = tmp_path / "proj_a"
    dir_b = tmp_path / "proj_b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_memory(dir_a, "a", "desc")
    _write_memory(dir_a, "b", "desc2")
    _write_memory(dir_b, "c", "desc")
    _write_memory(dir_b, "d", "desc2")

    spy = _SpyLLM(names=["a.md"])
    q = "How do I run the integration tests?"
    build_relevant_memories_prefix(q, dir_a, spy, "m")
    # 切到 dir_b，即使 query 完全一致也要重跑（先清节流避免被拦）
    _clear_throttle(monkeypatch)
    spy.names = ["c.md"]
    build_relevant_memories_prefix(q, dir_b, spy, "m")
    assert len(spy.calls) == 2


def test_cache_caches_empty_prefix(tmp_path: Path):
    """selector 返回 0 条 → 空 prefix 也要进缓存，连续提问就不重复调 LLM。"""
    _write_memory(tmp_path, "a", "x")
    _write_memory(tmp_path, "b", "y")
    spy = _SpyLLM(names=[])  # 明确不选任何记忆

    out1 = build_relevant_memories_prefix(_LONG_Q, tmp_path, spy, "m")
    out2 = build_relevant_memories_prefix(_LONG_Q + ".", tmp_path, spy, "m")
    assert out1 == "" and out2 == ""
    # 第二次相似 query 命中缓存，不重复调
    assert len(spy.calls) == 1


def test_reset_relevant_memories_cache(tmp_path: Path):
    """reset 后再发同样 query 应重新走 side-query。"""
    _write_memory(tmp_path, "a", "desc")
    _write_memory(tmp_path, "b", "desc2")
    spy = _SpyLLM(names=["a.md"])

    build_relevant_memories_prefix(_LONG_Q, tmp_path, spy, "m")
    assert len(spy.calls) == 1

    reset_relevant_memories_cache()
    build_relevant_memories_prefix(_LONG_Q, tmp_path, spy, "m")
    assert len(spy.calls) == 2


# ---------------------------------------------------------------------------
# 性能优化：方案 4（会话级节流）
# ---------------------------------------------------------------------------

def test_throttle_blocks_repeat_side_query(tmp_path: Path):
    """缓存未命中 + 节流窗口内 → 跳过 side-query（零 LLM 调用、不写缓存）。"""
    _write_memory(tmp_path, "a", "desc")
    _write_memory(tmp_path, "b", "desc2")
    spy = _SpyLLM(names=["a.md"])

    out1 = build_relevant_memories_prefix(_LONG_Q, tmp_path, spy, "m")
    assert out1 != ""
    assert len(spy.calls) == 1

    # 不同话题 → 缓存未命中；但节流窗口内 → 直接空返回
    out2 = build_relevant_memories_prefix(
        "What does the authentication middleware do exactly?", tmp_path, spy, "m")
    assert out2 == ""
    assert len(spy.calls) == 1


def test_throttle_skipped_does_not_poison_cache(tmp_path: Path):
    """节流导致的空结果不写缓存：窗口过期后同话题提问仍能重新 side-query。"""
    _write_memory(tmp_path, "a", "desc")
    _write_memory(tmp_path, "b", "desc2")
    spy = _SpyLLM(names=["a.md"])

    build_relevant_memories_prefix(_LONG_Q, tmp_path, spy, "m")
    # 不同话题被节流拦下 → 空结果，且不写缓存
    build_relevant_memories_prefix(
        "What does the authentication middleware do exactly?", tmp_path, spy, "m")
    assert len(spy.calls) == 1

    # 清节流后，被拦过的话题再问 → 重新 side-query（缓存没被空结果污染）
    frm._last_side_query_at = None
    out = build_relevant_memories_prefix(
        "What does the authentication middleware do exactly?", tmp_path, spy, "m")
    assert out != ""
    assert len(spy.calls) == 2


def test_will_need_side_query_respects_throttle(tmp_path: Path):
    """spinner 判断（will_need_side_query）与 build 行为一致：节流窗口内返回 False。"""
    _write_memory(tmp_path, "a", "desc")
    _write_memory(tmp_path, "b", "desc2")
    spy = _SpyLLM(names=["a.md"])

    build_relevant_memories_prefix(_LONG_Q, tmp_path, spy, "m")
    assert frm.will_need_side_query(
        "What does the authentication middleware do exactly?", tmp_path) is False
    # 同话题相似提问命中缓存 → 也不弹 spinner
    assert frm.will_need_side_query(_LONG_Q, tmp_path) is False
