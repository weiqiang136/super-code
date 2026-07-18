"""KAIROS memory system — append-only daily logs, dream consolidation, session persistence."""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

# 配置根目录：用于派生全局 fallback 目录 + 项目级目录的 projects/ 容器。
# 单独抽出来便于测试 monkeypatch。
BASE_CONFIG_DIR = Path.home() / ".config" / "super-code"

# 全局 fallback 记忆目录：当无法识别为 git 仓库时使用（保留旧行为，避免破坏既有用户的盘上数据）。
GLOBAL_MEMORY_DIR = BASE_CONFIG_DIR / "memory"

# 向后兼容：旧代码 `from features.memory import MEMORY_DIR` 仍可工作；
# 新代码应改用 `get_memory_dir(cwd)` 以获得项目级隔离。
MEMORY_DIR = GLOBAL_MEMORY_DIR

# sanitize_path 单段最大长度：超出后截断 + 拼接 hash 后缀，避免触发文件系统 255 字节上限。
MAX_SANITIZED_LENGTH = 200

MAX_ENTRYPOINT_LINES = 200
# MEMORY.md 注入系统提示前的字节上限，与行上限配合使用：行少但单行很长（例如索引条目超过 150 字符）
# 也可能撑爆 prompt，所以需要"行 + 字节"双重 cap。
MAX_ENTRYPOINT_BYTES = 25_000
ENTRYPOINT_NAME = "MEMORY.md"
LOCK_FILE_NAME = ".consolidate-lock"
HOLDER_STALE_S = 3600
SESSION_SCAN_INTERVAL_S = 600

_last_session_scan_at: float = 0.0


# ---------------------------------------------------------------------------
# 项目级记忆目录解析（Step 2）
# ---------------------------------------------------------------------------

# 不可见非字母数字字符替换正则。
_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9]")


def _sanitize_path(name: str) -> str:
    """把任意路径字符串变成可作为目录名的 safe slug。

    '/Users/foo/my-project' → '-Users-foo-my-project'
    'C:\\\\Users\\\\foo'    → 'C--Users-foo'

    超过 MAX_SANITIZED_LENGTH（200 字符）时截断 + 拼接稳定 hash 后缀，
    既避免触发文件系统单段 255 字节上限，又保留唯一性。

    使用 hashlib.sha1 取前 8 个 hex 作为 hash（确定性、跨平台稳定）。
    """
    sanitized = _SANITIZE_RE.sub("-", name)
    if len(sanitized) <= MAX_SANITIZED_LENGTH:
        return sanitized
    # 截断 + hash 后缀。hash 输入用原始 name（不是 sanitized），保留完整信息熵。
    import hashlib
    digest = hashlib.sha1(name.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{sanitized[:MAX_SANITIZED_LENGTH]}-{digest}"


def _find_git_root(start: Path) -> Path | None:
    """从 start 目录向上查找 git 仓库根目录。

    优先调用 `git rev-parse --show-toplevel`（处理 worktree、submodule、bare 仓库等
    所有边界情况，最权威）；找不到 git 可执行文件或非 git 仓库 → 返回 None。

    异常一律降级为 None，绝不让记忆目录解析破坏主流程启动。
    """
    try:
        # 故意不传 text=True：Windows 中文环境的系统默认 ANSI 代码页（GBK 等）
        # 解码 git 的 UTF-8 输出会失败，导致 stdout=None 后续 .strip() AttributeError。
        # 用 bytes 模式 + 显式 utf-8/replace 解码，规避平台编码差异。
        result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            capture_output=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout or b""
    top = raw.decode("utf-8", errors="replace").strip()
    if not top:
        return None
    # git rev-parse 在 Windows 下返回 forward slash 路径，Path 能正确处理
    return Path(top)


def get_memory_dir(cwd: Path | str | None = None) -> Path:
    """返回当前工作目录对应的记忆目录。

    解析顺序：
        1. 若 cwd 位于 git 仓库内 → <BASE_CONFIG_DIR>/projects/<sanitized-git-root>/memory/
           同一仓库（含 worktree）共享同一目录。
        2. 否则 → GLOBAL_MEMORY_DIR（保留旧行为，避免破坏非 git 工作流）。

    参数：
        cwd: 任何 PathLike 或 None；None → Path.cwd()。
    """
    base = Path(cwd) if cwd is not None else Path.cwd()
    git_root = _find_git_root(base)
    if git_root is None:
        return GLOBAL_MEMORY_DIR
    # 用 absolute 字符串作为 sanitize 输入。
    # 不解析 symlink：保持与 cwd 显示一致，避免不同入口指向同一 inode 但 sanitize 结果不同。
    return BASE_CONFIG_DIR / "projects" / _sanitize_path(str(git_root)) / "memory"


# ---------------------------------------------------------------------------
# 目录工具
# ---------------------------------------------------------------------------

def ensure_memory_dir(memory_dir: Path) -> None:
    """创建记忆目录及 logs 子目录。"""
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "logs").mkdir(parents=True, exist_ok=True)


def daily_log_path(memory_dir: Path, today: date | None = None) -> Path:
    """返回当天日志路径 memory_dir/logs/YYYY/MM/YYYY-MM-DD.md，自动创建父目录。"""
    today = today or date.today()
    path = memory_dir / "logs" / str(today.year) / f"{today.month:02d}" / f"{today.isoformat()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_to_daily_log(memory_dir: Path, entry: str) -> None:
    """向当天日志追加一条带时间戳的记录。"""
    path = daily_log_path(memory_dir)
    timestamp = datetime.now().strftime("%H:%M")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- [{timestamp}] {entry}\n")


# ---------------------------------------------------------------------------
# 记忆索引
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntrypointTruncation:
    """truncate_entrypoint_content 的结构化返回值。

    - content: 截断后（含 WARNING 尾巴）的最终文本，可直接拼进 system prompt
    - line_count / byte_count: **原始**行数 / 字节数（不是截断后），用于上层埋点 / 调试
    - was_line_truncated / was_byte_truncated: 哪种 cap 触发了截断
    """
    content: str
    line_count: int
    byte_count: int
    was_line_truncated: bool
    was_byte_truncated: bool


def truncate_entrypoint_content(raw: str) -> EntrypointTruncation:
    """把 MEMORY.md 原文按"行 + 字节"双重 cap 截断，并附 WARNING 提示模型被截断。

        1. 先按行截（MAX_ENTRYPOINT_LINES = 200）—— 行是自然边界，先按它切
        2. 若仍超字节上限（MAX_ENTRYPOINT_BYTES = 25KB），在 ≤ cap 内的最后一个换行处下刀，
           避免切到半行让模型看到一段破碎的索引条目
        3. 在末尾拼一条 WARNING（带具体触发原因），让模型主动意识到自己只看到部分索引

    传入空字符串 / 未超 cap 时不附 WARNING、不改动内容（除 strip 前后空白）。
    """
    trimmed = raw.strip()
    content_lines = trimmed.split("\n") if trimmed else []
    line_count = len(content_lines)
    # 按 UTF-8 字节数衡量；str.encode 一次即可（25KB 体量可忽略开销）
    byte_count = len(trimmed.encode("utf-8"))

    was_line_truncated = line_count > MAX_ENTRYPOINT_LINES
    # 用**原始**字节数判定字节超限：行截之后体积会缩小，但 WARNING 的语义是
    # "你的索引文件本身就过大"，要让上层埋点拿到真实数据
    was_byte_truncated = byte_count > MAX_ENTRYPOINT_BYTES

    if not was_line_truncated and not was_byte_truncated:
        return EntrypointTruncation(
            content=trimmed,
            line_count=line_count,
            byte_count=byte_count,
            was_line_truncated=False,
            was_byte_truncated=False,
        )

    truncated = (
        "\n".join(content_lines[:MAX_ENTRYPOINT_LINES])
        if was_line_truncated
        else trimmed
    )

    # 字节再切：在 MAX_ENTRYPOINT_BYTES 范围内找最后一个换行；找不到（单行超大）退化为硬切
    if len(truncated.encode("utf-8")) > MAX_ENTRYPOINT_BYTES:
        # 按字节定位切点：先编码再回切，避免多字节字符被一刀两断
        encoded = truncated.encode("utf-8")[:MAX_ENTRYPOINT_BYTES]
        # 在 bytes 上找最后一个 '\n'（0x0A 是单字节，安全）
        cut_at = encoded.rfind(b"\n")
        if cut_at > 0:
            encoded = encoded[:cut_at]
        # errors="ignore" 兜底：rfind 找不到换行硬切时可能切到多字节边界
        truncated = encoded.decode("utf-8", errors="ignore")

    # WARNING 文案：根据触发原因变化
    if was_byte_truncated and not was_line_truncated:
        reason = f"{byte_count} bytes (limit: {MAX_ENTRYPOINT_BYTES}) — index entries are too long"
    elif was_line_truncated and not was_byte_truncated:
        reason = f"{line_count} lines (limit: {MAX_ENTRYPOINT_LINES})"
    else:
        reason = f"{line_count} lines and {byte_count} bytes"

    warning = (
        f"\n\n> WARNING: {ENTRYPOINT_NAME} is {reason}. "
        f"Only part of it was loaded. Keep index entries to one line under ~200 chars; "
        f"move detail into topic files."
    )

    return EntrypointTruncation(
        content=truncated + warning,
        line_count=line_count,
        byte_count=byte_count,
        was_line_truncated=was_line_truncated,
        was_byte_truncated=was_byte_truncated,
    )


def load_memory_index(memory_dir: Path) -> str:
    """读取 MEMORY.md，按"行 + 字节"双重 cap 截断后返回（含 WARNING 尾巴）。

    不存在 / 读取异常 → 返回空字符串（保留旧行为，让上层 build_memory_system_section
    走"no memories yet"分支）。返回值类型仍是 str，对外签名不变。
    """
    path = memory_dir / ENTRYPOINT_NAME
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return truncate_entrypoint_content(text).content


# ---------------------------------------------------------------------------
# 整合锁（防止多进程并发 dream）
# ---------------------------------------------------------------------------

def _lock_path(memory_dir: Path) -> Path:
    return memory_dir / LOCK_FILE_NAME


def read_last_consolidated_at(memory_dir: Path) -> float:
    """返回上次整合的 epoch 秒数，从未整合则返回 0。"""
    lp = _lock_path(memory_dir)
    try:
        return lp.stat().st_mtime
    except OSError:
        return 0.0


def try_acquire_lock(memory_dir: Path) -> bool:
    """尝试获取整合锁，成功返回 True。

    锁文件持久存在，身兼两职：
    - 互斥锁：存 PID，防止多进程并发 dream
    - 时间戳：mtime 记录上次整合完成时间（供 should_auto_dream 读取）
    """
    lp = _lock_path(memory_dir)
    my_pid = os.getpid()
    try:
        stat = lp.stat()
        age = datetime.now().timestamp() - stat.st_mtime
        holder_pid = int(lp.read_text().strip())
        # 同进程重入：上次 dream 正常完成后锁文件仍存在（mtime 已更新），
        # 但 holder_pid 是自己，不可能有"另一个" dream 在跑 → 直接允许
        if holder_pid == my_pid:
            lp.write_text(str(my_pid))
            return True
        if age < HOLDER_STALE_S:
            try:
                os.kill(holder_pid, 0)  # 检查进程是否存活
                return False
            except OSError:
                pass
    except (OSError, ValueError):
        pass
    lp.write_text(str(my_pid))
    return True


def release_lock(memory_dir: Path) -> None:
    """更新锁文件 mtime 为当前时间（标记整合完成时间）。"""
    lp = _lock_path(memory_dir)
    try:
        now = datetime.now().timestamp()
        os.utime(lp, (now, now))
    except OSError:
        pass


def record_consolidation(memory_dir: Path) -> None:
    """记录一次整合完成（手动 /dream 也调用此函数）。"""
    lp = _lock_path(memory_dir)
    lp.write_text(str(os.getpid()))
    now = datetime.now().timestamp()
    os.utime(lp, (now, now))    # 更新锁文件的访问时间和修改时间为当前时间


def should_auto_dream(memory_dir: Path, min_hours: float, min_sessions: int,
                      current_session_id: str,
                      sessions_dir: Path | None = None) -> bool:
    """检查是否满足自动 dream 条件：距上次整合超过 min_hours 且新会话数 >= min_sessions。"""
    global _last_session_scan_at

    last = read_last_consolidated_at(memory_dir)
    now = datetime.now().timestamp()
    hours_since = (now - last) / 3600 if last > 0 else float("inf")

    if hours_since < min_hours:
        return False

    if now - _last_session_scan_at < SESSION_SCAN_INTERVAL_S:
        return False
    _last_session_scan_at = now

    scan_dir = sessions_dir or (Path.home() / ".config" / "super-code" / "sessions")
    count = 0
    if scan_dir.exists():
        for f in scan_dir.iterdir():
            if f.suffix == ".jsonl" and current_session_id not in f.name and f.stat().st_mtime > last:
                count += 1
    return count >= min_sessions


# ---------------------------------------------------------------------------
# <system_reminder> 标签提取
# ---------------------------------------------------------------------------

def extract_memory_tags(text: str) -> list[str]:
    """从 assistant 输出中提取 <system_reminder>...</system_reminder> 内容。"""
    return [m.strip() for m in re.findall(r"<system_reminder>(.*?)</system_reminder>", text, re.DOTALL)]


def list_sessions_since(since_ts: float, sessions_dir: Path | None = None,
                        current_session_id: str = "") -> list[str]:
    """返回 since_ts 之后修改过的会话 ID 列表（排除当前会话）。"""
    scan_dir = sessions_dir or (Path.home() / ".config" / "super-code" / "sessions")
    result: list[str] = []
    if not scan_dir.exists():
        return result
    for f in scan_dir.iterdir():
        if (f.suffix == ".jsonl"
                and current_session_id not in f.name
                and f.stat().st_mtime > since_ts):
            result.append(f.stem)
    return result


# ---------------------------------------------------------------------------
# 系统提示词段落 — 精简版
#
# 与旧版（~6800 tokens）的差异：
#   - 保留 MEMORY.md 索引注入（常驻环境上下文，不受 query 语义限制）
#   - 删除 TYPES_SECTION / WHAT_NOT_TO_SAVE / WHEN_TO_ACCESS / TRUSTING_RECALL
#     四段长文本 (~2000 tokens)，细节由 /memory 命令和 find_relevant_memories 提供
#   - 保留一句 TRUSTING_RECALL 护栏（推荐前先确认文件存在）
#
# 分工：
#   MEMORY.md 索引 = 常驻环境上下文（身份、项目事实）
#   find_relevant_memories = 按 query 精选 5 条全文注入（技术细节、历史决策）
# ---------------------------------------------------------------------------


def build_memory_system_section(memory_dir: Path) -> str:
    """生成记忆系统说明 + MEMORY.md 索引，拼接到系统提示词。

    保留 MEMORY.md 内容注入（常驻环境上下文），
    删除冗长的类型说明/保存格式/访问规则（省 ~2000 tokens）。
    find_relevant_memories 仍按需注入精选记忆全文，两者互补。
    """
    preamble = (
        f"你有持久化记忆系统，位于 `{memory_dir}/`。\n"
        "读 MEMORY.md 了解索引，参照已有 .md 文件的 frontmatter 格式保存新记忆。\n"
        "用户要求记住某事时立即保存；要求忘记某事时找到并删除对应文件。\n"
        "基于记忆做推荐前，先用 Read/Grep 确认相关文件/函数仍然存在。\n"
        "可用命令：/dream（整合记忆）、/memory（查看索引）、/remember <内容>（手动追加日志）。\n"
    )

    index = load_memory_index(memory_dir)
    if index:
        return preamble + f"\n{index}\n"
    return preamble + "\n尚无已整合的记忆。\n"


# ---------------------------------------------------------------------------
# Dream 整合提示词
# ---------------------------------------------------------------------------

def build_dream_prompt(memory_dir: Path, transcript_dir: str = "", # 会话记录目录
                       session_ids: list[str] | None = None) -> str:
    """构建 dream 整合的四阶段提示词。"""
    extra_parts: list[str] = []
    extra_parts.append(
        "**Tool constraints for this run:** Bash is not available. "
        "Edit and Write are restricted to files within the memory directory. "
        "Read, Grep, and Glob are unrestricted."
    )
    if session_ids:
        extra_parts.append(
            f"Sessions since last consolidation ({len(session_ids)}):\n"
            + "\n".join(f"- {sid}" for sid in session_ids)
        )
    extra = "\n\n".join(extra_parts)
    extra_section = f"\n\n## Additional context\n\n{extra}" if extra else ""

    transcript_line = ""
    if transcript_dir:
        transcript_line = (
            f"\nSession transcripts: `{transcript_dir}` "
            "(large JSONL files — grep narrowly, don't read whole files)\n"
        )

    return f"""\
# Dream: Memory Consolidation

You are performing a dream — a reflective pass over your memory files. \
Synthesize what you've learned recently into durable, well-organized memories \
so that future sessions can orient quickly.

Memory directory: `{memory_dir}`
This directory already exists — write to it directly with the Write tool \
(do not run mkdir or check for its existence).
{transcript_line}
---

**⚠️ CRITICAL — 两种文件格式完全不同：**

| | Topic 文件（Phase 3 产出） | MEMORY.md（Phase 4 产出） |
|---|---|---|
| 格式 | YAML frontmatter + Markdown body | 纯 Markdown 链接列表 |
| 示例 | `---\\nname: foo\\n---\\n\\n内容` | `- [标题](foo.md) — 描述` |
| 有标题？ | ✅ body 内可以有 | ❌ 绝对禁止 # / ## |

**MEMORY.md 每行固定格式（必须严格遵守）：**
```
- [从 topic 文件 frontmatter name 摘的标题](filename.md) — 一行简短描述
```
- 必须以 `- ` 开头，然后是 `[标题]`，然后是 `(filename.md)`，然后是 ` — `（em-dash），最后是描述
- 不是 `filename.md - 描述` ❌
- 不是 `**filename.md** 描述` ❌
- 不是 `# Memory Index` / `## 用户记忆` ❌

---

## Phase 1 — Orient

- Use Glob to list all files in `{memory_dir}/` to see what already exists
- Read `{ENTRYPOINT_NAME}` to understand the current index
- Skim existing topic files so you improve them rather than creating duplicates

## Phase 2 — Gather recent signal

Look for new information worth persisting:
1. **Daily logs** (`logs/YYYY/MM/YYYY-MM-DD.md`) if present
2. **Existing memories that drifted** — facts that contradict something you see now

## Phase 3 — Consolidate

For each thing worth remembering, write or update a memory file at the top \
level of the memory directory.

**File format** (frontmatter required):
```
---
name: kebab-case-slug
description: one-line summary
type: user | feedback | project | reference
---

body content
```

**Types:**
- `user` — 用户角色、偏好、知识背景（永远相关）
- `feedback` — 用户纠正过的行为规则，含 **Why:** 和 **How to apply:** 行
- `project` — 代码/git 推导不出的项目事实、决策、deadline；相对日期转绝对日期
- `reference` — 外部系统的资源指针

**不要保存：**
- 代码结构、架构、文件路径 — 读代码就能推导
- git 历史、谁改了什么 — git log/blame 是权威
- 调试方案/修复配方 — fix 在代码里，commit message 有上下文
- AGENTS.md 已记录的内容
- 当前会话的临时任务状态

## Phase 4 — Prune and index

Update `{ENTRYPOINT_NAME}` so it stays under {MAX_ENTRYPOINT_LINES} lines.

**严格格式要求：MEMORY.md 是一个纯列表文件 —— 第一行直接开始第一个条目，没有任何标题。**

- 每行一条：`- [标题](filename.md) — 一行描述（≤150字符）`
- 禁止任何标题：不能有 `#`、`##`，不能有"Memory Index"、"用户记忆"等分类标签
- 禁止用粗体文件名（`**filename**`）代替链接
- 禁止创造 user/feedback/project/reference 以外的分类
- 整个文件是纯平铺列表，无前言、无标题、无分段

❌ 错误（会被拒绝）:
```
# Memory Index
## 用户
- [用户背景](user.md) — ...
```
❌ 错误（缺少 `- ` 前缀和 `[]()` 链接）:
```
user.md - 中文开发者，偏好先分析不动手
bugs-and-fixes.md - 21 bugs: 12 fixed, 6 open
```
✅ 正确:
```
- [用户背景](user.md) — 中文开发者，Java后端，偏好先分析不动手
- [Bug清单](bugs-and-fixes.md) — 21 bugs：12 fixed, 6 open, 2 partial
- [只改被要求的代码](feedback/only-change-when-told.md) — 硬边界：未经授权不Edit/Write
- [Token优化策略](project/token-optimization.md) — 5层方案，省~10,500 tokens/轮
```

---

Return a brief summary of what you consolidated, updated, or pruned.{extra_section}"""
