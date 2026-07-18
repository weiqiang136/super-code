"""Step 6 — 按相关性精选记忆并注入到当轮 user message。

    用户每次提问 → 扫记忆目录 → side-query 让小模型从 manifest 里挑最多 5 条 →
    把这些记忆的**全文**（每条前面拼 Step 4 freshness warning）打包成一个
    <system-reminder> 块，附在 user message 前面。

为什么走 side-query 而不是简单的关键字匹配：
    关键字（如 "auth" 命中 feedback-handler.md）会过度触发；模型挑选有上下文判断，
    准确率高得多。max_tokens=256 + JSON schema 让 side-query 成本可忽略。

最小改动原则：
    - 不修改 engine / context / run_query / permissions
    - 仅暴露 build_relevant_memories_prefix(...) 一个供 tui/app.py 调用的入口
    - 任何失败（IO / API / JSON 解析）静默降级返回空字符串
"""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from features.memory_age import memory_freshness_text
from features.memory_scan import (
    ENTRYPOINT_NAME,
    MemoryHeader,
    format_memory_manifest,
    scan_memory_files,
)

# side-query 至多挑选这么多记忆。
MAX_SELECTED = 5

# 单次 side-query 的 token 上限。返回 JSON 很短（几个 filename），256 已足够冗余。
SELECT_MAX_TOKENS = 256

# 单个被选中记忆的正文截断。防止某个超大 topic 文件把当轮 prompt 撑爆。
MAX_MEMORY_BODY_CHARS = 8_000

# 触发 side-query 的最低记忆数：少于这个数量直接全注入（不调用 LLM 反而更便宜）。
MIN_MEMORIES_FOR_SIDE_QUERY = 2

# 性能优化：跳过明显不需要记忆的输入（方案 3）
# - 太短：上下文不足，selector 也挑不准
# - 单 token 输入：通常是 "ok / yes / 继续 / stop" 这种确认信号
# 阈值松一点（10 字符）覆盖大多数确认词，不会误伤"修一下 bug"这种短指令
_SKIP_LOOKUP_MIN_CHARS = 10
_CONFIRM_WORDS = {
    "yes", "no", "ok", "okay", "y", "n",
    "continue", "stop", "go", "next", "done",
    "继续", "停", "好", "好的", "嗯", "对",
}

# 性能优化：连续提问缓存（方案 1）
# 同一会话内连续提问大概率围绕同一话题（"看 auth.py" → "它有 bug 吗" → "修一下"），
# 上一次精选出的记忆完全可以复用。用 SequenceMatcher 算文本相似度，> 阈值即命中。
_CACHE_SIMILARITY_THRESHOLD = 0.6


# 模块级缓存。封进 dict 是为了让 reset 一次性清空、也便于将来切到 LRU。
# memory_dir 不同 → 不复用（避免切项目时拿错记忆）。
_lookup_cache: dict[str, Any] = {
    "memory_dir": None,   # str | None：上次查询时的 memory_dir 绝对路径
    "query": None,        # str | None：上次查询的 user_input
    "prefix": "",         # str：上次产出的 prefix 文本
}



_SELECT_SYSTEM_PROMPT = (
    "You are selecting memory files that will be useful to a coding assistant "
    "processing the user's query. You will be given the user's query and a list of "
    "available memory files with their filenames and one-line descriptions.\n\n"
    f"Return a JSON object with key 'selected_memories' whose value is a list of "
    f"filenames (at most {MAX_SELECTED}). Include only memories that are clearly "
    "useful based on their name and description. If unsure, do NOT include — "
    "be selective. If no memory is clearly useful, return an empty list."
)


# ---------------------------------------------------------------------------
# 公共数据
# ---------------------------------------------------------------------------

def _select_with_side_query(query: str, memories: list[MemoryHeader],
                            llm_client: Any, model: str) -> list[str]:
    """向 LLM 发起一次轻量 side-query，让它从 manifest 中挑选最多 MAX_SELECTED 个 filename。

    设计要点：
        - 直接复用主对话同款 LLMClient.create（非流式），不引入新依赖
        - prompt 里强制要求"JSON 对象 + key=selected_memories"，本端用 json.loads 解析
        - 任何异常（API / 解析 / 类型不符）返回空列表，调用方据此降级
        - 返回值会被白名单过滤：只保留 manifest 中真实存在的 filename，避免幻觉
    """
    manifest = format_memory_manifest(memories)
    user_msg = (
        f"User query:\n{query}\n\n"
        f"Available memories:\n{manifest}\n\n"
        "Respond ONLY with a JSON object like: "
        "{\"selected_memories\": [\"file1.md\", \"file2.md\"]}\n"
        "Do not wrap in markdown fences. Do not add any commentary."
    )

    try:
        result = llm_client.create(
            model=model,
            max_tokens=SELECT_MAX_TOKENS,
            messages=[{"role": "user", "content": user_msg}],
            system=_SELECT_SYSTEM_PROMPT,
        )
    except Exception:
        return []

    # 提取 text 块
    text = ""
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += str(block.get("text", ""))

    if not text.strip():
        return []

    # 一些模型仍会偶尔加 ```json 包装，宽容处理
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # 去掉第一行 ```... 与最后一行 ```
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            cleaned = "\n".join(lines[1:-1])

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("selected_memories")
    if not isinstance(raw, list):
        return []

    # 白名单过滤 + 去重 + 截断到 MAX_SELECTED
    valid_filenames = {m.filename for m in memories}
    selected: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        if item in valid_filenames and item not in seen:
            seen.add(item)
            selected.append(item)
            if len(selected) >= MAX_SELECTED:
                break
    return selected


def find_relevant_memories(query: str, memory_dir: Path,
                           llm_client: Any, model: str) -> list[MemoryHeader]:
    """返回与 query 最相关的 MemoryHeader 列表（最多 MAX_SELECTED 条）。

    流程：
        1. scan_memory_files 拿全部 header（自动排除 MEMORY.md / logs/）
        2. 记忆数 < MIN_MEMORIES_FOR_SIDE_QUERY → 全部返回，省一次 LLM 调用
        3. 否则走 _select_with_side_query
    任何失败静默返回空列表。
    """
    if not query.strip():
        return []
    try:
        memories = scan_memory_files(memory_dir)
    except Exception:
        return []
    if not memories:
        return []
    if len(memories) < MIN_MEMORIES_FOR_SIDE_QUERY:
        return memories

    selected_names = _select_with_side_query(query, memories, llm_client, model)
    if not selected_names:
        return []
    by_name = {m.filename: m for m in memories}
    return [by_name[name] for name in selected_names if name in by_name]


# ---------------------------------------------------------------------------
# 注入文本构造
# ---------------------------------------------------------------------------

def _read_memory_body(header: MemoryHeader) -> str:
    """读取记忆全文。失败返回空串（调用方据此把整条 skip 掉）。"""
    try:
        text = header.file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > MAX_MEMORY_BODY_CHARS:
        text = text[:MAX_MEMORY_BODY_CHARS] + "\n…[truncated]"
    return text


def build_relevant_memories_prefix(query: str, memory_dir: Path,
                                   llm_client: Any, model: str) -> str:
    """供 tui/app.py 调用的总入口。

    返回值规范：
        - 拿不到相关记忆 / 失败 / 短输入 → 空字符串（调用方直接 `prefix + user_input`，
          空串自然降级为 user_input）
        - 有相关记忆 → 一段 <system-reminder>...</system-reminder> 包裹的文本，
          末尾自带 "\n\n" 分隔符，便于直接拼到 user_input 前面

    性能优化（不影响功能正确性）：
        1. _should_skip_lookup：短输入 / 确认词直接跳过，零 LLM 调用
        2. _cache_hit：连续提问同话题命中缓存，零 LLM 调用
        3. 命中缓存时 freshness 也是缓存的旧值——可接受，新鲜度对几秒/几分钟内的
           话题切换没有实际差异
    """
    if _should_skip_lookup(query):
        return ""

    memory_dir_key = str(memory_dir)
    cached = _cache_hit(query, memory_dir_key)
    if cached is not None:
        return cached

    prefix = _build_prefix_uncached(query, memory_dir, llm_client, model)
    _cache_store(query, memory_dir_key, prefix)
    return prefix


def _build_prefix_uncached(query: str, memory_dir: Path,
                           llm_client: Any, model: str) -> str:
    """实际执行精选 + 拼装的内核。从 build_relevant_memories_prefix 拆出来，
    便于缓存包装；签名 / 行为与原函数完全一致。"""
    selected = find_relevant_memories(query, memory_dir, llm_client, model)
    if not selected:
        return ""

    parts: list[str] = []
    for h in selected:
        # MEMORY.md 不应出现（scan_memory_files 已过滤），这里再保险一道
        if h.filename == ENTRYPOINT_NAME:
            continue
        body = _read_memory_body(h)
        if not body:
            continue
        # Step 4：≥2 天的记忆带 freshness 警告；新鲜的不带（freshness_text 返回 "")
        freshness = memory_freshness_text(h.mtime_ms)
        freshness_block = f"<system-reminder>{freshness}</system-reminder>\n" if freshness else ""
        parts.append(f"## {h.filename}\n{freshness_block}{body.strip()}")

    if not parts:
        return ""

    body = "\n\n".join(parts)
    return (
        f"<system-reminder>\n"
        f"Relevant memories selected for this turn ({len(parts)}):\n\n"
        f"{body}\n"
        f"</system-reminder>\n\n"
    )


# ---------------------------------------------------------------------------
# 性能优化辅助函数（方案 1 + 方案 3）
# ---------------------------------------------------------------------------

def _should_skip_lookup(query: str) -> bool:
    """判断是否跳过 side-query。方案 3：避免对确认词 / 超短输入也起 LLM 往返。

    规则（任一命中即跳过）：
        - strip 后为空
        - 总字符数 < _SKIP_LOOKUP_MIN_CHARS
        - 全部是确认词 / 终止词（大小写不敏感）
    """
    s = query.strip()
    if not s:
        return True
    if len(s) < _SKIP_LOOKUP_MIN_CHARS:
        # 短输入：再检查一次是不是确认词，是的话明确跳过；不是的话也跳过
        # （10 字符以下不足以触发有意义的 selector 判断）
        return True
    # 长度够但全是确认词（如 "yes please" / "OK 继续" 这种）也跳过
    if s.lower() in _CONFIRM_WORDS:
        return True
    return False


def _cache_hit(query: str, memory_dir_key: str) -> str | None:
    """命中返回缓存的 prefix（可能是空字符串也算命中），未命中返回 None。

    命中条件：memory_dir 完全相同 + 与上次 query 文本相似度 ≥ 阈值。
    用 str | None 而不是 (bool, str)：None 明确表示"未命中"，与"命中但 prefix=空"区分。
    """
    if _lookup_cache["memory_dir"] != memory_dir_key:
        return None
    last_query = _lookup_cache["query"]
    if not last_query:
        return None
    # SequenceMatcher 在短文本上计算很快（数十微秒级），不引入新依赖
    similarity = SequenceMatcher(None, last_query, query).ratio()
    if similarity >= _CACHE_SIMILARITY_THRESHOLD:
        return _lookup_cache["prefix"]
    return None


def _cache_store(query: str, memory_dir_key: str, prefix: str) -> None:
    """覆盖式写入；不做 LRU——TUI 是单用户单进程，一对游标足够。"""
    _lookup_cache["memory_dir"] = memory_dir_key
    _lookup_cache["query"] = query
    _lookup_cache["prefix"] = prefix


def will_need_side_query(query: str, memory_dir: Path) -> bool:
    """纯本地判断：`build_relevant_memories_prefix()` 是否会触发 LLM side-query。

    供 TUI 层在调用前缀构建函数前决定是否显示 "Searching memories…" spinner。
    不复用内部函数的判断结果——刻意独立实现，避免未来任一端的逻辑变更造成
    调用层与实现层的隐性耦合。性能：仅做 scan_memory_files（内存文件数通常
    ≤100，~ms 级），无网络 IO。
    """
    if _should_skip_lookup(query):
        return False
    memory_dir_key = str(memory_dir)
    if _cache_hit(query, memory_dir_key) is not None:
        return False
    try:
        memories = scan_memory_files(memory_dir)
    except Exception:
        return False
    return len(memories) >= MIN_MEMORIES_FOR_SIDE_QUERY


def reset_relevant_memories_cache() -> None:
    """供测试 / `/clear` 命令调用，清空缓存。生产场景换记忆目录会自动失效，
    一般不需要手动 reset。"""
    _lookup_cache["memory_dir"] = None
    _lookup_cache["query"] = None
    _lookup_cache["prefix"] = ""

