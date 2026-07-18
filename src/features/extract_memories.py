"""Step 5 — 后台 extract_memories（独立子 agent）。

设计目标：
    每个交互轮结束后，后台启动一个**独立**的小型 LLM agent，让它"看最近 N 条消息
    然后写记忆"。主智能体可能在主流程里忘了主动写记忆——后台抽取作为兜底，
    把"用户偏好 / 项目事实 / 反馈"持久化到 memory_dir。

最小改动原则：
    1. 不修改 core/engine.py / core/permissions.py。后台 agent 用**自己专属的**
       Engine + PermissionChecker 实例（与主对话完全隔离，避免 dream_mode 全局
       状态污染主流程）。
    2. 复用 permissions.enter_dream_mode 作为沙箱（语义已经匹配 extract 需求：
       Read/Glob/Grep 全开；Edit/Write 限于 memory_dir 内）。
    3. 不复用主对话 prompt cache。当前 LLM 层没有 cache 抽象，
       第一版接受 cache miss、把"最近 N 条消息片段"以纯文本方式拼进 prompt。
    4. fire-and-forget：后台 daemon 线程跑，主流程不阻塞。

游标与互斥（闭包内状态）：
    - last_processed_count: 已抽取过的消息总数；新一轮只看尾巴增量
    - in_progress: 防止重叠运行
    - 主智能体本轮已自己写了 memory_dir 文件 → 跳过（mutual exclusion）
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from core.engine import Engine
from core.permissions import PermissionChecker
from features.memory_scan import format_memory_manifest, scan_memory_files
from tools.file_edit import FileEditTool
from tools.file_read import FileReadTool
from tools.file_write import FileWriteTool
from tools.glob_tool import GlobTool
from tools.grep_tool import GrepTool

# extract agent 每次最多跑这么多 turn。turn 1 全部并行 Read，turn 2 全部并行 Write，
# 5 是给"先列目录再细读"的边界场景留余地。
MAX_EXTRACT_TURNS = 5

# 抽取 prompt 里"最近 N 条对话"的上限。控制 prompt 体积，避免一次抽取就把上下文撑满。
RECENT_MESSAGES_FOR_EXTRACT = 20

# 节流：少于这么多新增 model-visible 消息时跳过（避免 user 仅按回车也触发）。
MIN_NEW_MESSAGES = 1


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

def _format_recent_excerpt(messages: list[dict], limit: int = RECENT_MESSAGES_FOR_EXTRACT) -> str:
    """把最近 N 条 user/assistant 消息序列化为纯文本片段。

    跳过 tool_result（噪音多、价值低）与 reasoning_content（思考模型私有字段）。
    每条消息只取**文本块**，丢弃 tool_use 的结构化字段——抽取 agent 关注的是"用户
    说了什么 / 模型回了什么文字"，工具调用细节通常不需要进入抽取上下文。
    """
    out: list[str] = []
    # 只看 user/assistant，按时间正序保留尾部 limit 条
    visible = [m for m in messages if m.get("role") in ("user", "assistant")]
    for msg in visible[-limit:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and block.get("text"):
                        parts.append(str(block["text"]))
                    # tool_result 形态：{"type": "tool_result", ...} —— 跳过
                elif isinstance(block, str):
                    parts.append(block)
            text = "\n".join(parts).strip()
        else:
            text = str(content)
        if not text:
            continue
        # 单条消息超长截断，避免单条 assistant 长 markdown 输出撑爆 excerpt
        if len(text) > 4000:
            text = text[:4000] + "…[truncated]"
        out.append(f"[{role}]\n{text}")
    return "\n\n".join(out)


def build_extract_prompt(message_excerpt: str, existing_memories_manifest: str,
                         memory_dir: Path) -> str:
    """构造后台抽取 agent 的 user prompt。

    设计要点（对齐 buildExtractAutoOnlyPrompt）：
        - 显式声明可用工具 + 写入只允许 memory_dir，让模型不要去试别的工具
        - 预先注入"已有记忆清单"避免重复创建同名文件
        - 强调"只看以下对话片段，不要去 grep/git 验证"——抽取阶段不应分心做研究
        - 强约束 frontmatter 格式，对齐 Step 1/4 的类型枚举
    """
    manifest_block = (
        f"\n\n## Existing memory files\n\n{existing_memories_manifest}\n\n"
        "Check this list before writing — update an existing file rather than creating a duplicate."
        if existing_memories_manifest
        else ""
    )

    return f"""You are now acting as the memory extraction subagent. Analyze the conversation excerpt below and update the persistent memory system.

Memory directory: `{memory_dir}`

Available tools: Read, Grep, Glob, and Edit/Write **only for paths inside the memory directory**. Any attempt to edit code outside the memory directory will be denied. You have at most {MAX_EXTRACT_TURNS} turns.

Strategy:
  - turn 1: issue all Read calls in parallel for files you might update
  - turn 2: issue all Write/Edit calls in parallel
  - Do NOT investigate or verify content beyond the excerpt (no grepping source code, no git commands)

If nothing is worth saving, do nothing and end your turn — extraction is best-effort.

## Memory types

Use exactly one of: `user`, `feedback`, `project`, `reference`.
- user      — facts about the user's role, preferences, knowledge
- feedback  — corrections or confirmed approaches; body must include **Why:** and **How to apply:** lines
- project   — ongoing work / decisions / incidents not derivable from code or git; convert relative dates to absolute
- reference — pointers to external systems (dashboards, ticket trackers)

## What NOT to save
- Code patterns / architecture / file paths — derivable by reading the project
- Git history / who-changed-what
- Debugging fix recipes — they live in the commit message
- Ephemeral task details from this conversation

## File format

Each memory is its own `.md` file with frontmatter:

```markdown
---
name: <kebab-case slug>
description: <one line; used by future relevance selection>
type: <user|feedback|project|reference>
---

<body — Why/How to apply for feedback & project>
```

After writing a topic file, append (or update) a one-line pointer in `{memory_dir}/MEMORY.md`:
`- [Title](file.md) — one-line hook`

Keep `MEMORY.md` under 200 lines.{manifest_block}

## Conversation excerpt

{message_excerpt}
"""


# ---------------------------------------------------------------------------
# 沙箱：与 _run_dream 同模式 —— 后台 agent 用自己专属的 PermissionChecker，
# 避免 dream_mode 全局状态污染主对话
# ---------------------------------------------------------------------------

def _build_extract_engine(app_config: Any) -> tuple[Engine, PermissionChecker]:
    """创建一个专门给 extract 后台 agent 用的临时 Engine。

    工具列表故意不含 Bash / Agent / AskUserQuestion / MCP 等：
        - Bash 在 extract 阶段没有合理用途，关掉就不存在"判定只读"的麻烦
        - Agent 等会让 extract 越权 spawn subagent
    PermissionChecker.dream_mode 会进一步把 Edit/Write 锁死在 memory_dir 内。
    """
    perms = PermissionChecker(auto_approve=True)
    engine = Engine(
        tools=[FileReadTool(), GlobTool(), GrepTool(), FileEditTool(), FileWriteTool()],
        system_prompt="",
        permission_checker=perms,
        provider=app_config.provider,
        api_key=app_config.api_key,
        base_url=app_config.base_url,
        model=app_config.model,
        max_tokens=app_config.max_tokens,
    )
    return engine, perms


def _run_extract_agent(prompt: str, memory_dir: Path, app_config: Any) -> list[str]:
    """同步执行一次抽取。返回写入的文件绝对路径列表（去重，排除 MEMORY.md 单独统计）。

    异常一律吞掉，记日志—— extract 是 best-effort，绝不影响主流程。
    """
    engine, perms = _build_extract_engine(app_config)
    perms.enter_dream_mode(str(memory_dir))
    written: list[str] = []
    seen: set[str] = set()
    turns_seen = 0
    try:
        for event in engine.submit(prompt):
            kind = event[0]
            if kind == "tool_call":
                # event: ("tool_call", tool_name, tool_input, activity, tool_use_id)
                _, tool_name, tool_input, _act, _tid = event
                if tool_name in ("Edit", "Write"):
                    fp = tool_input.get("file_path") if isinstance(tool_input, dict) else None
                    if isinstance(fp, str) and fp and fp not in seen:
                        seen.add(fp)
                        written.append(fp)
            elif kind == "waiting":
                # waiting = 模型已发完一段 text、准备 emit 工具调用；用它粗略数 turn
                turns_seen += 1
                if turns_seen >= MAX_EXTRACT_TURNS:
                    engine.abort()
    except Exception:
        # 静默失败：网络断、API key 失效、模型拒绝……不能让 extract 把 TUI 拖崩
        pass
    finally:
        perms.exit_dream_mode()
    return written


# ---------------------------------------------------------------------------
# 公共入口：闭包式状态 + fire-and-forget 线程
# ---------------------------------------------------------------------------


def _last_assistant_wrote_memory(messages: list[dict], since_index: int,
                                 memory_dir: Path) -> bool:
    """判断从上次后台抽取游标之后的新消息里，主智能体有没有动过 memory 目录下的文件。避免重复。"""
    memory_dir_abs = str(memory_dir.resolve())
    for msg in messages[since_index:]:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if block.get("name") not in ("Edit", "Write"):
                continue
            inp = block.get("input") or {}
            fp = inp.get("file_path") if isinstance(inp, dict) else None
            if not isinstance(fp, str):
                continue
            try:
                resolved = str(Path(fp).resolve())
            except OSError:
                continue
            if resolved.startswith(memory_dir_abs):
                return True
    return False


def init_extract_memories() -> tuple[
        Callable[[list[dict], Any, Path], bool],
    Callable[[], None],
]:
    """创建一个闭包，封装游标 / in_progress / 锁等可变状态。

    返回 (execute, reset)：
        execute(messages, app_config, memory_dir) -> bool
            尝试启动一次后台抽取；返回 True 表示真的起了线程，False 表示被跳过（节流 / 互斥 / 主智能体已写）。
        reset() —— 仅供测试在 setup 时清零状态。
    """
    state = {
        "last_processed_count": 0,
        "in_progress": False,
    }
    lock = threading.Lock()

    def execute(messages: list[dict], app_config: Any, memory_dir: Path) -> bool:
        with lock:
            if state["in_progress"]:
                return False
            new_count = len(messages) - state["last_processed_count"]
            if new_count < MIN_NEW_MESSAGES:
                return False
            # 主智能体已写记忆 → 推进游标 + 跳过抽取
            if _last_assistant_wrote_memory(messages, state["last_processed_count"], memory_dir):
                state["last_processed_count"] = len(messages)
                return False
            state["in_progress"] = True
            snapshot = list(messages)  # 防止 worker 跑期间外部修改 messages

        def _worker():
            try:
                excerpt = _format_recent_excerpt(snapshot)
                if not excerpt.strip():
                    return
                manifest = format_memory_manifest(scan_memory_files(memory_dir))
                prompt = build_extract_prompt(excerpt, manifest, memory_dir)
                _run_extract_agent(prompt, memory_dir, app_config)
            finally:
                with lock:
                    state["last_processed_count"] = len(snapshot)
                    state["in_progress"] = False

        threading.Thread(target=_worker, daemon=True).start()
        return True

    def reset() -> None:
        with lock:
            state["last_processed_count"] = 0
            state["in_progress"] = False

    return execute, reset


# 模块级单例：与 features.memory._last_session_scan_at 同模式
_extractor: Callable[[list[dict], Any, Path], bool] | None = None
_reset: Callable[[], None] | None = None


def execute_extract_memories(messages: list[dict], app_config: Any, memory_dir: Path) -> bool:
    """入口函数。第一次调用时懒初始化闭包。"""
    global _extractor, _reset
    if _extractor is None:
        _extractor, _reset = init_extract_memories()
    return _extractor(messages, app_config, memory_dir)


def reset_extract_memories() -> None:
    """仅供测试 setUp/tearDown 使用，清零游标与 in_progress。"""
    global _extractor, _reset
    if _reset is not None:
        _reset()
