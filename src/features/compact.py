"""Context compression — summarise old messages to free token budget."""
from __future__ import annotations

import re
from typing import Any
from core.llm import LLMClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4                     # 非 CJK 字符的粗估比例（约 4 字符 ≈ 1 token）
MIN_RECENT_MESSAGES = 6                 # 至少保留最近 N 条消息不压缩
MIN_RECENT_TOKENS = 10_000              # 至少保留最近 N 个 token 不压缩
COMPACT_MAX_OUTPUT_TOKENS = 16_384        # 摘要最大输出 token 数

# ── 按模型计算自动压缩阈值（Step 6） ───────────────────────────────────────
# 简化版：触发阈值 = 模型 context window × COMPACT_TRIGGER_RATIO。
# 估算函数 estimate_tokens 已 CJK-aware（中文 1 字 ≈ 1 token），中文重的会话也能
# 被正确触发。
COMPACT_TRIGGER_RATIO = 0.8
DEFAULT_CONTEXT_WINDOW = 128_000        # 未识别模型的兜底窗口
# 已知模型 → context window 映射。匹配规则：先精确，再按 key 长度倒序做前缀匹配，
# 让 "gpt-4-32k-0613" 先命中 "gpt-4-32k" 而不是 "gpt-4"。
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # DeepSeek (V4 系列) - 全系标配 1M 上下文
    "deepseek-v4": 1_000_000,
    "deepseek-v4-pro": 1_000_000,

    # GLM (智谱) - 保留 GLM-4-Plus，新增 GLM-4.7 系列，移除较小的 GLM-4 基础/轻量版
    "glm-4-plus": 128_000,  # 高端系列，窗口保持 128K
    "glm-4.7": 200_000,  # 最新旗舰模型
    "glm-4.7-air": 200_000,  # 轻量版但窗口相同
    "glm-4.7-flash": 200_000,  # 极速版但窗口相同

    # Claude (Anthropic) - 4.6 及以上版本支持 1M (beta)
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-4-6": 200_000,
}
# ── PTL（Prompt Too Long）兜底重试 ────────────────────────────────────────
# 当 summarizer 调用本身因输入过长被 API 拒绝时，从头部丢一段消息再重试，
# 最多 PTL_RETRY_MAX 次。避免用户被永久卡死（每次 autocompact 都拿同样过大的输入再撞）。
PTL_RETRY_MAX = 3
# 重试时在新输入头部插入的合成 marker：让 summarizer 知道前面有内容被截掉了，
# 同时保证首条消息是 user 角色（满足 OpenAI 协议要求）。
PTL_MARKER = "[earlier conversation truncated for compaction retry]"

# 压缩边界 marker：嵌入在 summary user 消息 content 开头，标识"此处是上一次压缩点"。
# 用纯文本嵌入而非 dict 字段：天然兼容会话持久化（content 是必持久化字段），且作为
# 普通文本发给 LLM API 也无副作用。下一次压缩时扫到 marker → 只总结 marker 之后的
# 增量对话，避免旧 summary 被反复套娃总结导致的信息逐次劣化。
COMPACT_BOUNDARY_MARKER = "<!-- COMPACT_BOUNDARY -->"

# 兼容老 session：在引入 marker 之前生成的 summary 消息以这个固定前缀开头，
# 也识别为边界，避免老 session 第一次重新压缩时仍然套娃。
_LEGACY_SUMMARY_PREFIX = "[This is a summary of the conversation so far"

# 三明治结构 prompt：
#   - 首尾各嵌一遍"禁止调工具"硬指令（NO_TOOLS_PREAMBLE/TRAILER）：summarizer 偶尔
#     不听话会试图调工具；本路径调用没启用 tools，模型若返回 tool_use 块会让我们
#     拿不到任何文本输出，整个压缩失败
#   - 强制 <analysis> 草稿区 + <summary> 正式区：让模型先把思考过程写下来再写
#     正式总结，明显抬高 summary 质量；analysis 块不会进入下一轮 context（被
#     _format_compact_summary 整段剥除）
#   - "All user messages" 强制全列：防止模型主观漏掉早期但关键的用户约束
#   - "Next Step verbatim" 强制引用原文：防止"下一步"被脑补成用户没说过的需求
COMPACT_PROMPT = """\
IMPORTANT: Do NOT call any tools or functions in your response. Output ONLY plain text.

Please provide a detailed summary of our conversation so far. This summary \
will replace the earlier messages to free up context space, so it must \
preserve every detail needed to continue the work seamlessly.

Before writing the final summary, work through your analysis inside an \
<analysis> block. The <analysis> block will be discarded — it exists only to \
help you produce a higher-quality <summary>. Then write the final summary \
inside a <summary> block using EXACTLY the section headers below.

<analysis>
Step through the conversation chronologically. Note every user message, every \
file touched, every error encountered, every decision made. Identify what is \
load-bearing for continuing the work. Be exhaustive — this draft is for you, \
not for the user.
</analysis>

<summary>
## Primary Request and Intent
What the user is trying to accomplish overall.

## Key Technical Concepts
Important technical details, patterns, frameworks, or constraints established.

## Files and Code Sections
Key files discussed or modified, with brief notes on what was done to each. \
Quote short relevant code snippets where helpful.

## Errors and Fixes
Any errors encountered and how they were resolved. Pay special attention to \
user feedback / corrections.

## Problem Solving
Problems already solved + any ongoing troubleshooting.

## All User Messages
List EVERY non-tool-result user message verbatim or near-verbatim, in order. \
Do NOT skip messages you consider unimportant — user intent is the most \
critical signal in this summary.

## Pending Tasks
Outstanding work items the user explicitly asked for but that are not yet done.

## Current Work
What was being worked on most recently and its current status.

## Optional Next Step
The next concrete action. This MUST be a verbatim quote of the user's most \
recent request or instruction; if the user did not specify one, write \
"(no explicit next step)".
</summary>

REMINDER: Output ONLY plain text. Do NOT call any tools. Do NOT emit \
tool_use blocks. The entire response must be a single text message containing \
the <analysis> and <summary> blocks.\
"""

COMPACT_SYSTEM = "You are a conversation summarizer. Produce a structured, detailed summary following the user's requested format."


# ---------------------------------------------------------------------------
# Summary 格式化：剥 <analysis>，提 <summary>
# ---------------------------------------------------------------------------

# 注意 re.DOTALL：让 . 跨行匹配；非贪婪 *? 防止把多个 block 误并成一个。
_ANALYSIS_RE = re.compile(r"<analysis>.*?</analysis>", re.DOTALL | re.IGNORECASE)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)


def _format_compact_summary(raw: str) -> str:
    """把 summarizer 的原始输出处理成最终摘要文本。

    处理顺序（每一步都带 fallback，确保任何输入形态都能产出非空字符串）：
      1) 若存在 <summary>...</summary>：取闭包内文本（这是模型按格式输出的正常路径）
      2) 否则若存在 <analysis>...</analysis>：把 analysis 块整段删掉，剩下的当作摘要
         （某些模型会忽略 <summary> 包装直接写正文）
      3) 否则原文返回（最坏情况：模型完全没听格式指令）
    最后压一下连续空行，让最终注入 context 的内容尽量干净。
    """
    text = raw or ""
    m = _SUMMARY_RE.search(text)
    if m:
        body = m.group(1)
    else:
        # 没找到 <summary>：退而求其次，删 <analysis> 块（如果有）保留剩余
        body = _ANALYSIS_RE.sub("", text)
    # 多余空行压成最多一个空行
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


# ---------------------------------------------------------------------------
# PTL helpers
# ---------------------------------------------------------------------------

# 常见 prompt-too-long / context length 的判定关键词（小写匹配）。
# OpenAI 的 BadRequestError 消息里通常含 "context_length_exceeded" 或 "maximum context length"；
# 兼容供应商（DeepSeek/Moonshot 等）措辞略有差异，这里用关键词模糊命中。
_PTL_KEYWORDS = (
    "context_length_exceeded",
    "maximum context length",
    "prompt is too long",
    "too many tokens",
    "context length",
    "request too large",
    "exceeds max length",
)


def _is_ptl_error(exc: BaseException) -> bool:
    """识别一次 LLM 调用异常是否为 PTL（输入过长）。

    判定策略保守：只看异常字符串是否包含已知关键词。其它 4xx/5xx 不当 PTL，
    避免把鉴权失败 / 限流 / 网络错误也当 PTL 反复丢消息重试。
    """
    msg = str(exc).lower()
    return any(k in msg for k in _PTL_KEYWORDS)


def _drop_head_for_ptl(messages: list[dict]) -> list[dict] | None:
    """从消息列表头部丢一段，重试更小的输入。返回截断后的新列表；无法再截则返回 None。

    策略（最小可用版本）：
      - 一次丢约 20% 的消息，至少丢 1 条
      - 不允许把列表丢空：至少保留 1 条业务消息（再加上压缩 prompt 那条 user）
      - 头部插入 PTL_MARKER 合成 user，保证首条是 user 角色
    注意：messages 末尾那条是压缩 prompt（user），永不丢；只动业务部分。
    """
    if len(messages) <= 2:
        # 只剩 1 条业务 + 1 条 prompt（或更少），再丢就没意义
        return None
    # 末尾的 prompt 不动；业务部分 = messages[:-1]
    body = messages[:-1]
    drop = max(1, len(body) // 5)
    if drop >= len(body):
        # 至少保留一条业务消息
        drop = len(body) - 1
    if drop <= 0:
        return None
    truncated_body = body[drop:]
    # 头部插合成 user marker：(1) 满足"首条必须是 user"；(2) 让 summarizer 知道前面被截
    new_messages = [{"role": "user", "content": PTL_MARKER}] + truncated_body + [messages[-1]]
    # 再过一遍交替修正，避免合成 marker 之后正好接一条 user 造成连续同角色
    return _fix_alternation(new_messages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_of(content: Any) -> str:
    """从消息 content 中提取纯文本（兼容 str、list of blocks 等格式）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
                c = block.get("content", "")
                if isinstance(c, str):
                    parts.append(c)
                parts.append(str(block.get("input", "")))
            elif hasattr(block, "text"):
                parts.append(getattr(block, "text", ""))
        return " ".join(parts)
    return str(content) if content else ""


# CJK 字符（中日韩）按 1 token 计；其它字符走 chars/4。修复 chars/4 对中文 4 倍
# 低估的 bug —— 中文重的会话不会因为估算偏小永远到不了阈值。
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯豈-﫿]")


def estimate_tokens(messages: list[dict]) -> int:
    """粗略估算 token 数：CJK 字符按 1:1，其它字符按 1:CHARS_PER_TOKEN。"""
    total = 0
    for m in messages:
        text = _text_of(m.get("content", ""))
        if not text:
            continue
        cjk = len(_CJK_RE.findall(text))
        total += cjk + (len(text) - cjk) // CHARS_PER_TOKEN
    return total


def get_context_window(model: str | None) -> int:
    """返回模型的 context window；未识别模型走 DEFAULT_CONTEXT_WINDOW。

    匹配顺序：精确 → 最长前缀。最长前缀避免 "gpt-4-32k-0613" 被 "gpt-4" 误命中。
    """
    if not model:
        return DEFAULT_CONTEXT_WINDOW
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]
    # 按 key 长度倒序做前缀匹配（先具体后宽松）
    for key in sorted(MODEL_CONTEXT_WINDOWS.keys(), key=len, reverse=True):
        if model.startswith(key):
            return MODEL_CONTEXT_WINDOWS[key]
    return DEFAULT_CONTEXT_WINDOW


def should_compact(messages: list[dict], model: str | None = None,
                   last_input_tokens: int | None = None) -> bool:
    """判断是否需要自动压缩对话。阈值按模型 context window 的 COMPACT_TRIGGER_RATIO 计算。"""
    # 阈值下限是 MIN_RECENT_TOKENS：避免极小窗口模型（如 gpt-4 8K）算出的阈值太低
    # 导致每轮都触发压缩 → 死循环。
    threshold = max(
        int(get_context_window(model) * COMPACT_TRIGGER_RATIO),
        MIN_RECENT_TOKENS,
    )
    return estimate_tokens(messages) > threshold


# ---------------------------------------------------------------------------
# Compact boundary 识别
# ---------------------------------------------------------------------------

def _is_compact_boundary(msg: dict) -> bool:
    """判断一条消息是否为压缩边界（上一次压缩生成的 summary user 消息）。

    检测规则：
      - 必须是 user 消息
      - content 提取的纯文本去掉前导空白后，以 COMPACT_BOUNDARY_MARKER 开头
        或者以老格式 summary 前缀开头（兼容引入 marker 之前生成的 session）
    """
    if msg.get("role") != "user":
        return False
    text = _text_of(msg.get("content", "")).lstrip()
    return text.startswith(COMPACT_BOUNDARY_MARKER) or text.startswith(_LEGACY_SUMMARY_PREFIX)


def _find_last_boundary_index(messages: list[dict]) -> int:
    """返回最后一个 boundary 消息的下标；不存在返回 -1。倒序扫描，命中即返回。"""
    for i in range(len(messages) - 1, -1, -1):
        if _is_compact_boundary(messages[i]):
            return i
    return -1


def get_messages_after_compact_boundary(messages: list[dict]) -> list[dict]:
    """返回最后一次压缩边界之后的"增量"消息（不含 boundary user 消息本身及其紧随的 ack）。

    用途：调用方需要单独看"自上次压缩以来新增了什么"时使用。
    若不存在 boundary，返回完整列表的浅拷贝。

    边界结构约定：[..., boundary_user(=summary), ack_assistant, 增量对话 ...]
    因此跳过 2 条（boundary + ack）。若 boundary 后没有 ack（极端情况），跳 1 条。
    """
    idx = _find_last_boundary_index(messages)
    if idx < 0:
        return list(messages)
    # boundary 紧随其后通常是我们写死的 ack assistant；跳过 boundary + ack
    skip = 2
    if idx + 1 >= len(messages) or messages[idx + 1].get("role") != "assistant":
        skip = 1
    return list(messages[idx + skip:])


# ---------------------------------------------------------------------------
# Message splitting
# ---------------------------------------------------------------------------

def _split_recent(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """将消息切分为 (待压缩的历史部分, 需要保留的最近消息)。"""
    if len(messages) <= MIN_RECENT_MESSAGES:
        return [], list(messages)

    keep_start = len(messages)
    kept_tokens = 0
    kept_msgs = 0

    for i in range(len(messages) - 1, -1, -1):
        # 与 estimate_tokens 同套 CJK-aware 规则，避免中文会话保留过少消息
        kept_tokens += estimate_tokens([messages[i]])
        kept_msgs += 1
        keep_start = i
        if kept_msgs >= MIN_RECENT_MESSAGES and kept_tokens >= MIN_RECENT_TOKENS:
            break

    # 不拆分 tool_use / tool_result 对：如果 keep_start 前一条 assistant 消息含
    # tool_use，则把它也纳入保留范围。改为检查前一条 assistant 而非当前 user 的
    # block 类型：原 all() 检查在 user 含混合 block（_fix_alternation 合并后）或
    # string content 时会漏检，导致 assistant(tool_use) 留在 history 里、对应的
    # tool_result 在 recent 里 → API 400 "insufficient tool messages"。
    # while 循环处理连续嵌套的 tool_use/tool_result 对（如 A1(tool_use), U1(result),
    # A2(tool_use) 都在 split 边界时）。
    while keep_start > 0:
        prev_msg = messages[keep_start - 1]
        if prev_msg.get("role") != "assistant":
            break
        prev_content = prev_msg.get("content", "")
        if not (isinstance(prev_content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in prev_content
        )):
            break
        keep_start -= 1

    return messages[:keep_start], messages[keep_start:]


# ---------------------------------------------------------------------------
# CompactService
# ---------------------------------------------------------------------------

class CompactService:
    """通过 API 摘要压缩对话上下文。"""

    def __init__(self, client: LLMClient, model: str, effort: str | None = None):
        self._client = client
        self._model = model
        self._effort = effort

    def compact(
        self,
        messages: list[dict],
        system_prompt: str,
        custom_instructions: str = "",
        attachments: list[dict] | None = None,
    ) -> tuple[list[dict], str]:
        """压缩 messages，返回 (new_messages, summary_text)。

        返回的消息列表结构：
            [frozen_prefix（含历次旧边界及其 ack）, user: 新边界+摘要, assistant: 确认,
             最近消息 …, 重注入 attachments …]

        关键设计：本次压缩只针对"上一次边界之后"的增量对话做总结。历次旧 summary 及其
        ack 作为 frozen_prefix 原封不动地保留下来。这样可避免多次压缩时旧 summary 被
        反复再总结导致的信息逐次劣化。

        attachments：调用方按需构造的"压缩后状态恢复"消息（如 plan reminder、
        worker 状态等），追加在 recent 之后。每条必须是合法 message dict
        （含 role/content）。下次压缩时这些消息会被 `_find_last_boundary_index`
        视为 boundary 之后的普通消息，但因紧贴上次 boundary、且 _split_recent
        优先保留尾部，正常情况下会进入 recent 而非被重新总结。
        """
        # 1) 切出 frozen_prefix（旧边界及之前） 和 active（边界之后的增量）
        boundary_idx = _find_last_boundary_index(messages)
        if boundary_idx >= 0:
            # boundary 之后我们写死了一条 assistant ack；frozen_prefix 含 ack 一起冻结
            ack_present = (
                boundary_idx + 1 < len(messages)
                and messages[boundary_idx + 1].get("role") == "assistant"
            )
            frozen_end = boundary_idx + 2 if ack_present else boundary_idx + 1
            frozen_prefix = list(messages[:frozen_end])
            active = list(messages[frozen_end:])
        else:
            frozen_prefix = []
            active = list(messages)

        # 2) 在 active 内部切出 (history, recent)
        history, recent = _split_recent(active)

        if not history:
            # 增量太少，没东西可总结：保持 messages 不变
            return list(messages), "(nothing to compact)"

        prompt = COMPACT_PROMPT
        if custom_instructions:
            prompt += f"\n\nAdditional instructions: {custom_instructions}"

        # 去除图片/文档以节省 token
        cleaned = _strip_media(history)
        # 防止 history 末尾若是 user(含 tool_result) 与下面追加的 prompt user 被
        # _fix_alternation 合并：合并后这条 user.content 会变成 [tool_result_block,
        # text_block] 混合结构，下游 _to_openai_messages 走多模态分支静默丢弃
        # tool_result，前一条 assistant(tool_use) 找不到响应 → API 返回 400
        # "insufficient tool messages following tool_calls message"。
        # 插入一条短 ack assistant 隔开即可，对 summary 质量无可见影响。
        if cleaned and cleaned[-1].get("role") == "user":
            cleaned.append({"role": "assistant", "content": "Acknowledged."})
        cleaned.append({"role": "user", "content": prompt}) # 然后把压缩的prompt作为user消息提供给ai

        # 确保首条消息是 user 角色，否则会报错
        if cleaned and cleaned[0].get("role") != "user":
            cleaned.insert(0, {"role": "user", "content": "(conversation start)"})

        cleaned = _fix_alternation(cleaned)

        # PTL 兜底：summarizer 调用自身可能因输入过长被 API 拒（典型异常含
        # "context_length_exceeded"）。捕获后从头部丢一段重试，最多 PTL_RETRY_MAX 次。
        # 不在此处吞其它异常（鉴权、网络、限流等），避免把无关错误误判成 PTL 反复丢消息。
        # 失败终态：把最后一次异常抛出去，由调用方（tui/app.py 的熔断器或 _cmd_compact）处理。
        attempt = 0
        while True:
            try:
                response = self._client.create(
                    model=self._model,
                    max_tokens=COMPACT_MAX_OUTPUT_TOKENS,
                    system=COMPACT_SYSTEM,
                    messages=cleaned,
                    effort=self._effort,
                )
                break
            except Exception as e:
                if not _is_ptl_error(e) or attempt >= PTL_RETRY_MAX:
                    raise
                truncated = _drop_head_for_ptl(cleaned)
                if truncated is None:
                    # 已经无法再丢（业务消息只剩一条），抛出让上层处理
                    raise
                cleaned = truncated
                attempt += 1

        # 提取摘要文本
        summary_text = ""
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                summary_text += block.get("text", "")
            elif hasattr(block, "text"):
                summary_text += block.text

        # Step 4：剥 <analysis> 草稿块、提取 <summary> 正式块。
        # _format_compact_summary 内部带多级 fallback，对任何输入形态都不会返回 None。
        summary_text = _format_compact_summary(summary_text)

        if not summary_text.strip():
            summary_text = "(compact produced empty summary)"

        # 3) 拼装：frozen_prefix + [新边界 summary, ack] + recent
        # COMPACT_BOUNDARY_MARKER 必须放在 content 最开头（_is_compact_boundary 用 lstrip 后
        # startswith 判定），否则下次压缩识别不到边界。
        new_summary_content = (
            f"{COMPACT_BOUNDARY_MARKER}\n\n"
            "[This is a summary of the conversation so far — "
            "the original messages have been compacted to save context space.]\n\n"
            + summary_text
        )
        new_messages: list[dict] = list(frozen_prefix)
        new_messages.append({
            "role": "user",
            "content": new_summary_content,
        })
        new_messages.append({
            "role": "assistant",
            "content": (
                "Understood. I've reviewed the conversation summary and I'm "
                "ready to continue from where we left off."
            ),
        })
        new_messages.extend(recent)
        # Step 7-A：把"压缩后状态恢复"附件追加到 recent 之后。
        # 仅追加合法 dict（含 role/content），其它静默跳过，避免上层构造错误连锁影响压缩主流程。
        # 角色交替守护：附件全是 user 角色，若直接追加会和 recent 末尾的 user 消息
        # （典型场景：tool_result 收尾的轮次、resume 的脏 session）连成两条 user，
        # 触发部分 API 的 400 报错。每次追加前检查上一条 role，撞了就先插一条短 ack
        # assistant 隔开。不调用 _fix_alternation：它会合并 content 改变类型，对含
        # tool_use/tool_result 的复杂消息有副作用。
        if attachments:
            for att in attachments:
                if not (isinstance(att, dict) and "role" in att and "content" in att):
                    continue
                if new_messages and new_messages[-1].get("role") == att.get("role"):
                    new_messages.append({
                        "role": "assistant" if att.get("role") == "user" else "user",
                        "content": "Acknowledged.",
                    })
                new_messages.append(att)
        return new_messages, summary_text


# ---------------------------------------------------------------------------
# Media stripping + alternation fix
# ---------------------------------------------------------------------------

def _strip_media(messages: list[dict]) -> list[dict]:
    """在发送给 LLM 进行总结前，移除图片/文档等多媒体内容以节省 Token。"""
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            new_blocks: list[Any] = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype in ("image", "document"):
                        new_blocks.append({"type": "text", "text": f"[{btype}]"})   # 图片被替换为占位符
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)
            out.append({"role": msg["role"], "content": new_blocks})
        else:
            out.append(dict(msg))
    return out


def _has_tool_blocks(content: Any) -> bool:
    """检查消息 content 中是否包含 tool_use 或 tool_result block。

    这些 block 在合并时被污染会导致下游 _to_openai_messages 静默丢弃数据：
    - user 含 tool_result + 任意其它 block → _to_openai_messages 视作混合内容、
      走 _user_content_blocks_to_openai 分支，tool_result 被丢弃 → 前一条
      assistant(tool_use) 找不到响应 → API 400。
    - assistant 含 tool_use + 任意其它 block → 合并本身安全（_to_openai_messages 同时
      处理 text 和 tool_calls），但仍以分隔符代替合并以保持一致性。
    """
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
        for b in content
    )


def _fix_alternation(messages: list[dict]) -> list[dict]:
    """修正消息列表，确保 user/assistant 角色严格交替，符合 API 要求。

    相邻两条消息 role 相同时分两种情况：
    - 若任一消息含 tool_use 或 tool_result block → 插入分隔符（不合并），
      避免合并后混合 block 导致 _to_openai_messages 静默丢弃 tool 相关数据。
    - 否则合并到前一条消息（两个纯字符串拼成换行分隔的单个字符串；
      至少一个 list 时转成 list 后拼接）。
    """
    if not messages:
        return messages
    fixed: list[dict] = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == fixed[-1]["role"]:
            # 涉及 tool block 的合并会静默丢数据 → 插入分隔符
            if _has_tool_blocks(fixed[-1].get("content", "")) or _has_tool_blocks(msg.get("content", "")):
                sep_role = "assistant" if msg["role"] == "user" else "user"
                fixed.append({"role": sep_role, "content": "Acknowledged."})
                fixed.append(msg)
                continue
            # 安全合并
            prev = fixed[-1].get("content", "")
            cur = msg.get("content", "")
            if isinstance(prev, str) and isinstance(cur, str):
                fixed[-1]["content"] = prev + "\n" + cur
            else:
                def _as_list(c: Any) -> list:
                    return list(c) if isinstance(c, list) else [{"type": "text", "text": str(c)}]
                fixed[-1]["content"] = _as_list(prev) + _as_list(cur)
        else:
            fixed.append(msg)
    return fixed
