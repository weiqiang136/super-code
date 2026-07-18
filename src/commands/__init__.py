"""Slash command system — parsing and dispatch."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape as _escape
from rich.table import Table

if TYPE_CHECKING:
    from core.engine import Engine
    from core.permissions import PermissionChecker
    from core.session import SessionStore
    from features.compact import CompactService
    from features.cost_tracker import CostTracker
    from features.plan import PlanModeManager


@dataclass
class CommandContext:
    engine: Engine
    session_store: "SessionStore | None"
    console: Console
    cwd: str
    model: str
    permissions: "PermissionChecker | None" = None
    new_session_store: object = None                    # 创建新会话的工厂函数，当执行完/clear的时候，会清空当前会话，创建新会话
    compact_service: "CompactService | None" = None
    plan_manager: "PlanModeManager | None" = None
    worker_manager: object = None           # WorkerManager | None；Step 7 压缩后注入 in-flight worker 状态用
    cost_tracker: "CostTracker | None" = None
    pending_query: str | None = None        # 命令执行后需要触发的后续 LLM 查询（如 /init 让模型扫描项目并写 AGENTS.md）
    memory_dir: object = None               # Path | None，记忆系统目录（Phase 6）


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_command(text: str) -> tuple[str, str] | None:
    """If text starts with '/', return (command_name, args)."""
    text = text.strip()
    if not text.startswith("/"):
        return None
    parts = text.split(None, 1)
    name = parts[0][1:].lower()
    args = parts[1] if len(parts) > 1 else ""
    return name, args


def is_known_command(name: str) -> bool:
    """判断 name 是否是已注册的命令（内置命令或 skill）。

    用于让调用方区分"用户敲了未知斜杠命令"和"用户输入恰好以 / 开头但不是命令"
    （如 `/query 接口的路径是什么` —— 这里 `/query` 是用户想问 LLM 的内容，
    不是命令名）。这种情况下应当把整段文本作为普通输入传给 LLM，而不是报
    "Unknown command"。

    判定顺序与 handle_command 保持一致：先查内置 _HANDLERS，再查 skill 注册表。
    """
    if name in _HANDLERS:
        return True
    from features.skills import get_skill
    return get_skill(name) is not None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

# /init 用：让模型扫描项目并写 AGENTS.md。AGENTS.md 会被 src/core/context.py
# 自动注入到系统提示词，所以一次写入持续生效。
_INIT_PROMPT_TEMPLATE = """请分析当前代码库并创建（或改进）`AGENTS.md` 文件。

`AGENTS.md` 会在每次 super-code 会话启动时被自动注入到系统提示词（参见
src/core/context.py 的 _get_agents_md_section），所以它必须**精炼**——只写从代码里
看不出来、但 super-code 每次都需要知道的事。

工作目录：{cwd}{existing_block}

## 要写什么

1. **常用命令**：build / lint / test / 运行单个测试。重点是非标准的命令——
   能从 pyproject.toml / package.json / Makefile 直接看到的标准命令（如 `pytest`、
   `npm test`）不必写。
2. **高层架构**：需要读多个文件才能理解的"big picture"。例如模块间的协作关系、
   核心数据流、关键扩展点。

## 怎么扫描

- 读 `pyproject.toml` / `package.json` / `Cargo.toml` / `go.mod` 等清单文件
- 读 `README*`（如果存在），把重要内容**提取**而不是复制
- 用 `Glob` / `Bash ls` 摸清顶层结构与关键入口
- 用 `Bash git log --oneline -20` 看 commit 信息风格（团队约定的 commit 格式
  通常需要写进 AGENTS.md）
- 检查这些 AI 配置文件，把重要部分纳入 AGENTS.md：
  `.cursor/rules/`、`.cursorrules`、`.github/copilot-instructions.md`、
  `.windsurfrules`、`.clinerules`、`.claude/CLAUDE.md`、`.super-code/`
- 用 `Grep` 找 lint / 格式化配置（ruff、eslint、prettier 等）

## 避免什么

- 不要重复废话，比如"为新工具写单测"、"提供有用的错误信息"、"不要把 API key
  写进代码"——这些 super-code 已经知道
- 不要逐个列组件 / 文件结构——super-code 会用 `Glob` / `Read` 自己发现
- 不要写通用编程实践（"写好代码"、"处理边界条件"）
- 不要瞎编"Common Development Tasks"、"Tips for Development"、"Support" 这种
  段——只写从你**实际读到的文件**里得到的信息
- 不要重复 README 已有的内容——简短引用即可

## 风格要求

- 使用**中文**编写
- 总长度 ≤ 150 行；超过 200 行就是在写文档而不是给 agent 提示
- 关键约定带"为什么"，例如 "Commit 格式 `fix：V3.0.XX：xxx`（项目历史风格）"
- 文件开头加上：

```
# AGENTS.md

本文件给 super-code（以及其它 AI 编码助手）在本仓库中工作时提供必要上下文。
```

## 已存在 AGENTS.md 的处理

如果上面提供了"当前 AGENTS.md 内容"块：
- 读懂现有内容，**保留用户手写、个性化、非显而易见的段落**
- 用具体 diff 的方式提改进建议（哪些段过时了、哪些命令路径变了、哪里可以补充）
- 经用户确认后再 Write 覆盖；不要静默覆盖

完成后用一两句话告诉用户改了什么。{extra_hint}
"""


def _cmd_help(ctx: CommandContext, args: str) -> None:
    table = Table(title="Available Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="green")
    table.add_column("Description")
    for name, desc, _ in _COMMAND_TABLE:
        table.add_row(f"/{name}", desc)
    ctx.console.print(table)


def _cmd_clear(ctx: CommandContext, args: str) -> None:     # 清空当前引擎的对话历史并创建一个新的会话存储
    ctx.engine.set_messages([])
    # Phase 3: 清空旧会话的 snippet 状态
    from core.file_state import clear_session_state
    old_sid = getattr(ctx.session_store, "session_id", "") if ctx.session_store else ""
    if old_sid:
        clear_session_state(old_sid)
    if callable(ctx.new_session_store):
        new_store = ctx.new_session_store()
        ctx.engine.set_session_store(new_store)
        ctx.session_store = new_store
    ctx.console.print("[green]✓[/green] Conversation cleared. New session started.")


def _cmd_history(ctx: CommandContext, args: str) -> None:       # 列出当前工作目录下所有已保存的会话目录，包括ID、标题、时间等元数据
    from core.session import SessionStore

    sessions = SessionStore.list_sessions(ctx.cwd)
    if not sessions:
        ctx.console.print("[dim]No saved sessions for this directory.[/dim]")
        return

    table = Table(title="Session History", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("ID", style="dim", width=10)
    table.add_column("Title")
    table.add_column("Messages", justify="right", width=8)
    table.add_column("Updated", width=20)

    for i, meta in enumerate(sessions, 1):
        from core.session import format_local_time
        table.add_row(
            str(i),
            meta.session_id[:8],
            meta.title[:50],
            str(meta.message_count),
            format_local_time(meta.updated_at, "%Y-%m-%d %H:%M:%S"),
        )
    ctx.console.print(table)


def _cmd_resume(ctx: CommandContext, args: str) -> None: # 根据提供的序列号或者会话ID查找并加载历史会话，恢复对话记录
    from core.session import SessionStore

    sessions = SessionStore.list_sessions(ctx.cwd)
    if not sessions:
        ctx.console.print("[dim]No saved sessions to resume.[/dim]")
        return

    if not args:
        # 无参数：启动交互式选择器，用方向键选择历史会话
        from tui.prompt import pick_session
        target_meta = pick_session(sessions)
        if target_meta is None:
            return  # 用户取消
    else:
        target_meta = None
        try:
            idx = int(args.strip()) - 1
            if 0 <= idx < len(sessions):
                target_meta = sessions[idx]
        except ValueError:
            pass

        if target_meta is None:
            needle = args.strip().lower()
            for meta in sessions:
                if meta.session_id.lower().startswith(needle):
                    target_meta = meta
                    break

        if target_meta is None:
            ctx.console.print(f"[red]Session not found: {args}[/red]")
            return

    if ctx.session_store and target_meta.session_id == ctx.session_store.session_id:
        ctx.console.print("[dim]Already in this session.[/dim]")
        return

    meta, messages = SessionStore.load_session(target_meta.session_id, ctx.cwd)
    if not messages:
        ctx.console.print("[red]Session has no messages.[/red]")
        return

    # Re-open the existing session store (no new file created)
    from core.session import SessionStore as SS
    resumed_store = SS(cwd=ctx.cwd, model=ctx.model,
                       session_id=target_meta.session_id)
    resumed_store._message_count = target_meta.message_count
    resumed_store._title = target_meta.title

    ctx.engine.set_messages(messages)
    ctx.engine.set_session_store(resumed_store)
    ctx.session_store = resumed_store

    # Phase 3: 从历史 tool_result metadata 重建 snippet 注册表
    rebuilt = ctx.engine.rebuild_snippets_from_messages()
    if rebuilt > 0:
        ctx.console.print(f"[dim]Restored {rebuilt} file snippet(s) from session history.[/dim]")

    ctx.console.print(
        f"[green]✓[/green] Resumed session [bold]{target_meta.session_id[:8]}[/bold]: "
        f"{_escape(target_meta.title[:50])}  ({len(messages)} messages)"
    )

    # 展示会话中的可见消息（跳过 tool_result 列表）
    from rich.markdown import Markdown
    # 过滤出可展示的消息：user 纯文本 + assistant 有文本内容的
    visible = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            if isinstance(content, list):
                continue  # tool_result 列表，跳过
            visible.append(msg)
        elif role == "assistant":
            text = (
                " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
                if isinstance(content, list) else str(content or "")
            )
            if text:
                visible.append({**msg, "_text": text})

    for msg in visible:
        role = msg.get("role", "")
        if role == "user":
            ctx.console.print(f"\n[bold cyan]You:[/bold cyan] {_escape(msg.get('content', ''))}")
        elif role == "assistant":
            ctx.console.print("\n[bold green]Assistant:[/bold green]")
            ctx.console.print(Markdown(msg["_text"]))


def _build_post_compact_attachments(ctx: CommandContext) -> list[dict]:
    """Step 7-A/B/C：构造"压缩后状态恢复"附件列表。

    目前覆盖：
      - Plan 状态：若仍在 plan 模式，注入 reminder + plan 文件当前内容
      - Worker 状态：若有 in-flight worker，注入运行状态描述
      - Skill 内容（Phase B）：若本会话调用过 skill，注入其 body 截断版供模型回看
      - 最近文件（Phase C）：取最近 Read 过的 top-N 文件，重新读取最新内容并注入
    每一类都用 try/except 单独包裹，单项失败不影响其它附件，也不影响主压缩流程。
    """
    out: list[dict] = []

    # Plan 重注入：plan 模式下让模型压缩后仍知道自己处于 plan 模式 + plan 文件内容
    plan_manager = ctx.plan_manager
    if plan_manager is not None:
        try:
            if plan_manager.is_active:
                plan_path = plan_manager.plan_file_path or "(unknown)"
                plan_content = plan_manager.get_plan_content() or "(empty)"
                out.append({
                    "role": "user",
                    "content": (
                        f"[plan-mode-reminder] You are still in plan mode. "
                        f"Plan file: {plan_path}\n\n"
                        f"[plan-file-content]\n{plan_content}"
                    ),
                })
        except Exception:
            # plan 状态读取异常不应阻塞压缩
            pass

    # Worker 重注入：把 in-flight worker 序列化成简短状态描述
    worker_manager = ctx.worker_manager
    if worker_manager is not None:
        try:
            running = worker_manager.get_running_status()  # list[dict]
            if running:
                lines = ["[worker-status] In-flight async workers:"]
                for w in running:
                    lines.append(
                        f"  - task_id={w.get('task_id')} "
                        f"description={w.get('description')!r} "
                        f"tool_uses={w.get('tool_uses')} "
                        f"activity={w.get('activity')!r}"
                    )
                out.append({"role": "user", "content": "\n".join(lines)})
        except Exception:
            pass

    # Skill 重注入（Phase B）：恢复"调用过哪些 skill"的内容，让压缩后模型仍能
    # 按 skill 指令工作。注意是"为上下文回顾"，不是"再次执行"——附件内容里
    # 显式提示模型不要 re-execute。
    try:
        from features.skills import get_invoked_skills, get_skill
        # 单 skill body 字符上限 ≈ 5K tokens（保头截尾）
        SINGLE_MAX = 20_000
        # 总字符预算 ≈ 25K tokens；超出后剩余 skill 仅占位不展开
        TOTAL_BUDGET = 100_000
        TRUNCATE_MARKER = (
            "\n\n[... skill content truncated for compaction; "
            "the full body remains earlier in the conversation if needed.]"
        )
        used = 0
        for name in get_invoked_skills():
            skill = get_skill(name)
            if skill is None:
                # skill 已被卸载（例如目录变更）→ 跳过，不抛错
                continue
            try:
                body = skill.get_prompt("") or ""
            except Exception:
                continue
            if not body.strip():
                continue
            # 单条上限：保头截尾，附 marker 提示模型可回看完整版
            if len(body) > SINGLE_MAX:
                body = body[:SINGLE_MAX] + TRUNCATE_MARKER
            # 总预算检查：超了则注入占位不展开
            if used + len(body) > TOTAL_BUDGET:
                out.append({
                    "role": "user",
                    "content": (
                        f"[skill-attachment:{name}] (omitted: total skill content "
                        f"budget exceeded; refer to earlier conversation if needed)"
                    ),
                })
                continue
            used += len(body)
            out.append({
                "role": "user",
                "content": (
                    f"[skill-attachment:{name}] You previously invoked this skill. "
                    f"Below is its content for reference only — do NOT re-execute "
                    f"these instructions; they are already accounted for in the "
                    f"conversation summary.\n\n{body}"
                ),
            })
    except Exception:
        # skill 模块异常不应阻塞压缩
        pass

    # 最近文件重注入（Phase C）：取最近 Read 过的 top-N 文件，重新读取最新内容
    # 注入 messages，让压缩后模型仍能就这些文件的具体内容继续工作。
    # 为什么"重新读"而不"复用历史 tool_result"：文件可能在压缩之间被改过 / 模型
    # 自己 Edit 过；当前内容才是模型继续工作需要的。
    try:
        from tools.file_read import FileReadTool

        # 二进制扩展名黑名单：注入这些只会得到乱码占位，浪费 token。
        # 后缀对比小写。点号包含。常见可读文本扩展名一律放行。
        _BINARY_EXTS = {
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
            ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
            ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".lib",
            ".class", ".pyc", ".pyo", ".jar", ".war", ".ear",
            ".mp3", ".mp4", ".wav", ".avi", ".mkv", ".mov", ".webm", ".flac",
            ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".db", ".sqlite", ".sqlite3", ".dat",
            ".woff", ".woff2", ".ttf", ".otf", ".eot",
        }
        TOP_N = 5
        SINGLE_MAX = 20_000          # 单文件字符上限（保头截尾）
        TOTAL_BUDGET = 100_000       # 总字符预算
        TRUNCATE_MARKER = (
            "\n\n[... file content truncated for compaction; "
            "use the Read tool with this exact path to fetch the full content if needed.]"
        )

        from pathlib import Path
        recents = FileReadTool.get_recent_reads()  # [(path, ts), ...] 按 ts desc
        used = 0
        emitted = 0
        for fpath, _ts in recents:
            if emitted >= TOP_N:
                break
            # 二进制扩展名黑名单
            try:
                ext = Path(fpath).suffix.lower()
            except Exception:
                ext = ""
            if ext in _BINARY_EXTS:
                continue

            # 重新读取最新内容（文件可能已被改/删）
            try:
                p = Path(fpath)
                if not p.exists() or not p.is_file():
                    out.append({
                        "role": "user",
                        "content": (
                            f"[file-attachment:{fpath}] (file no longer exists; "
                            f"refer to earlier conversation for prior content if needed)"
                        ),
                    })
                    emitted += 1
                    continue
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                # 读取失败（权限/编码/磁盘异常等）→ 跳过，不注入占位也不抛
                continue

            if not content:
                continue

            # 单文件上限：保头截尾 + marker
            if len(content) > SINGLE_MAX:
                content = content[:SINGLE_MAX] + TRUNCATE_MARKER

            # 总预算：超了则给占位（让模型仍知道这个文件最近被读过）
            if used + len(content) > TOTAL_BUDGET:
                out.append({
                    "role": "user",
                    "content": (
                        f"[file-attachment:{fpath}] (omitted: total file content "
                        f"budget exceeded; use Read tool to fetch if needed)"
                    ),
                })
                emitted += 1
                continue

            used += len(content)
            emitted += 1
            out.append({
                "role": "user",
                "content": (
                    f"[file-attachment:{fpath}] You recently read this file. "
                    f"Below is its current content (re-read at compaction time) "
                    f"for your reference:\n\n{content}"
                ),
            })
    except Exception:
        # 文件重注入整体异常不应阻塞压缩
        pass

    return out


def _cmd_compact(ctx: CommandContext, args: str) -> None:
    """压缩对话上下文，保留最近消息，用摘要替换历史消息。文件里的消息采用直接覆盖的方式"""
    from features.compact import estimate_tokens

    if ctx.compact_service is None:
        ctx.console.print("[dim]Compact service not available.[/dim]")
        return

    messages = ctx.engine.get_messages()
    if len(messages) < 4:
        ctx.console.print("[dim]Too few messages to compact.[/dim]")
        return

    pre_tokens = estimate_tokens(messages)
    ctx.console.print(f"[dim]Compacting {len(messages)} messages (~{pre_tokens:,} tokens)…[/dim]")

    # Step 7-A：构造压缩后重注入附件（plan / worker），与 compact 主流程解耦
    attachments = _build_post_compact_attachments(ctx)

    new_msgs, _ = ctx.compact_service.compact(
        messages, ctx.engine.system_prompt,
        custom_instructions=args,
        attachments=attachments,
    )
    ctx.engine.set_messages(new_msgs)

    # Phase 4: 压缩后 snippet 已过期（历史 read/edit 记录被摘要替代），
    # 失效所有旧 snippet 强制模型重新 read 文件。
    if ctx.session_store is not None:
        sid = getattr(ctx.session_store, "session_id", "")
        if sid:
            from core.file_state import invalidate_all_snippets
            invalidate_all_snippets(sid)

    # 将压缩后的消息持久化到当前 session
    if ctx.session_store is not None:
        import json
        from core.session import _serialize_message, _now_iso
        with open(ctx.session_store._jsonl_path, "w", encoding="utf-8") as fh:  # 压缩后的消息直接覆盖掉文件里的旧的历史消息
            for msg in new_msgs:
                safe = _serialize_message(msg)
                safe["_ts"] = _now_iso()
                fh.write(json.dumps(safe, ensure_ascii=False) + "\n")
        ctx.session_store._message_count = len(new_msgs)
        ctx.session_store._save_meta()

    from features.compact import estimate_tokens as et
    post_tokens = et(new_msgs)
    ctx.console.print(
        f"[green]✓[/green] Compacted: {pre_tokens:,} → {post_tokens:,} tokens "
        f"({len(messages)} → {len(new_msgs)} messages)"
    )



def _cmd_skills(ctx: CommandContext, args: str) -> None:
    """列出所有可用的 skill。"""
    from features.skills import list_skills

    skills = list_skills(user_invocable_only=True)
    if not skills:
        ctx.console.print("[dim]No skills available.[/dim]")
        return

    table = Table(title="Available Skills", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="green")
    table.add_column("Source", style="dim", width=8)
    table.add_column("Description")
    for s in skills:
        hint = f" [{s.argument_hint}]" if s.argument_hint else ""
        table.add_row(f"/{s.name}{hint}", s.source, s.description)
    ctx.console.print(table)


def _cmd_cost(ctx: CommandContext, args: str) -> None:
    """显示本次会话的 token 用量和费用摘要。"""
    if ctx.cost_tracker is None:
        ctx.console.print("[dim]Cost tracking not available.[/dim]")
        return
    ctx.console.print(ctx.cost_tracker.format_cost())


def _cmd_remember(ctx: CommandContext, args: str) -> None:
    """手动向当天日志追加一条记录。给用户一个入口来记住一些重要的事"""
    from features.memory import append_to_daily_log, ensure_memory_dir
    from pathlib import Path

    if not args.strip():
        ctx.console.print("[dim]Usage: /remember <text>[/dim]")
        return
    memory_dir = Path(ctx.memory_dir) if ctx.memory_dir else None
    if memory_dir is None:
        ctx.console.print("[dim]Memory system not available.[/dim]")
        return
    ensure_memory_dir(memory_dir)
    append_to_daily_log(memory_dir, args.strip())
    ctx.console.print("[green]✓[/green] Remembered.")


def _cmd_memory(ctx: CommandContext, args: str) -> None:
    """查看 / 列出 / 编辑记忆文件（Step 10）。

    用法（Option A：保留旧行为为默认，新增子用法都是显式参数）：
        /memory               — 打印 MEMORY.md 索引（与旧行为完全一致）
        /memory list          — 列出 memory_dir 下所有 topic 文件（按 mtime 倒序）
        /memory <number>      — 用 $EDITOR 打开"list"中编号 N 的文件
        /memory <substring>   — 模糊匹配 filename / description，打开第一个命中

    设计原则：
        - 旧无参数行为不变，老用户 muscle memory 受保护
        - 不做交互式上下键选择器：保持函数纯命令式、易测、轻量
        - 任何错误情况只 console.print 警告，绝不抛异常出栈
    """
    # 局部 import：避免 commands 模块导入时拉起整个记忆/编辑器调用链
    from features.memory import load_memory_index
    from features.memory_scan import scan_memory_files, format_memory_manifest

    memory_dir = Path(ctx.memory_dir) if ctx.memory_dir else None
    if memory_dir is None:
        ctx.console.print("[dim]Memory system not available.[/dim]")
        return

    arg = args.strip()

    # —— 无参数：保持旧行为，打印 MEMORY.md ——
    if not arg:
        index = load_memory_index(memory_dir)
        if index:
            ctx.console.print(index)
        else:
            ctx.console.print("[dim]No memories consolidated yet.[/dim]")
        ctx.console.print(
            "[dim]Tip: `/memory list` to browse topic files, "
            "`/memory <number|substring>` to edit one.[/dim]"
        )
        return

    # 提前 scan，list / number / substring 三条路径都需要
    headers = scan_memory_files(memory_dir)

    # —— list 子命令：纯打印清单（与 manifest 同格式），不进编辑器 ——
    if arg.lower() == "list":
        if not headers:
            ctx.console.print("[dim]No topic memories yet.[/dim]")
            return
        # 在 manifest 每行前面补编号，便于用户接着敲 `/memory N`
        manifest = format_memory_manifest(headers).splitlines()
        ctx.console.print(f"[dim]Available memories ({len(headers)}):[/dim]")
        for i, line in enumerate(manifest, 1):
            ctx.console.print(f"  {i}. {line[2:]}" if line.startswith("- ") else f"  {i}. {line}")
        ctx.console.print(
            "[dim]Use `/memory <number>` to edit, "
            "or `/memory <substring>` to search by name.[/dim]"
        )
        return

    # —— 数字：按 1-based 编号定位 ——
    if arg.isdigit():
        idx = int(arg)
        if not headers:
            ctx.console.print("[dim]No topic memories to open.[/dim]")
            return
        if not (1 <= idx <= len(headers)):
            ctx.console.print(
                f"[red]Number out of range:[/red] expected 1..{len(headers)}, got {idx}."
            )
            return
        _open_in_editor(ctx, headers[idx - 1].file_path)
        return

    # —— 子串模糊匹配（大小写不敏感）：filename 与 description 都参与 ——
    needle = arg.lower()
    matches = [
        h for h in headers
        if needle in h.filename.lower()
        or (h.description and needle in h.description.lower())
    ]
    if not matches:
        ctx.console.print(f"[dim]No memory matches `{arg}`.[/dim]")
        return
    if len(matches) > 1:
        # 多匹配时提示选择，但仍然打开第一个（保持单参数语义）
        names = ", ".join(m.filename for m in matches[:5])
        more = "" if len(matches) <= 5 else f", +{len(matches) - 5} more"
        ctx.console.print(f"[dim]{len(matches)} matches: {names}{more}. Opening the first.[/dim]")
    _open_in_editor(ctx, matches[0].file_path)


def _open_in_editor(ctx: CommandContext, path: Path) -> None:
    """用 $EDITOR / notepad / vi 打开 path，阻塞直到用户保存退出。

    跨平台策略：
        1. $EDITOR / $VISUAL 环境变量优先（git 同款约定）
        2. Windows 兜底 notepad；其它平台兜底 vi
        3. 阻塞期间 spinner 不在跑（命令路径本就同步），直接接管终端即可
    任何失败只 console.print，不抛出。
    """
    import subprocess

    if not path.exists():
        ctx.console.print(f"[red]File not found:[/red] {path}")
        return

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        editor = "notepad" if os.name == "nt" else "vi"

    ctx.console.print(f"[dim]Opening {path.name} with {editor}…[/dim]")
    try:
        # shell=False：editor 取自环境变量或硬编码，不需要 shell 解析，避免注入风险
        # 失败码（含用户在 vi 里 :cq）不当作异常，只提示
        result = subprocess.run([editor, str(path)])
        if result.returncode != 0:
            ctx.console.print(
                f"[dim]Editor exited with code {result.returncode}.[/dim]"
            )
    except FileNotFoundError:
        ctx.console.print(
            f"[red]Editor not found:[/red] `{editor}`. "
            "Set $EDITOR to a valid command (e.g. `code -w`, `nano`)."
        )
    except Exception as exc:
        ctx.console.print(f"[red]Failed to launch editor:[/red] {exc}")


def _cmd_dream(ctx: CommandContext, args: str) -> None:
    """触发 dream 整合：复用 app._run_dream()，带权限隔离和锁保护。"""
    from features.memory import (
        ensure_memory_dir, try_acquire_lock, release_lock, record_consolidation,
    )
    from pathlib import Path
    from tui.app import _run_dream

    memory_dir = Path(ctx.memory_dir) if ctx.memory_dir else None
    if memory_dir is None:
        ctx.console.print("[dim]Memory system not available.[/dim]")
        return

    ensure_memory_dir(memory_dir)
    if not try_acquire_lock(memory_dir):
        ctx.console.print("[dim]Another dream consolidation is already running.[/dim]")
        return

    try:
        _run_dream(ctx.engine, memory_dir, ctx.permissions, quiet=False)
        record_consolidation(memory_dir)
        ctx.console.print("[green]✓[/green] Dream consolidation complete.")
    except Exception as exc:
        ctx.console.print(f"[red]Dream failed: {exc}[/red]")
    finally:
        release_lock(memory_dir)


def _cmd_rename(ctx: CommandContext, args: str) -> None:
    if ctx.session_store is None:
        ctx.console.print("[dim]No active session to rename.[/dim]")
        return
    if not args.strip():
        ctx.console.print(
            f"Current session [bold]{ctx.session_store.session_id[:8]}[/bold]: "
            f"{_escape(ctx.session_store._title or '(untitled)')}"
        )
        return
    new_title = args.strip()[:80]
    ctx.session_store._title = new_title
    ctx.session_store._save_meta()
    ctx.console.print(
        f"[green]✓[/green] Session [bold]{ctx.session_store.session_id[:8]}[/bold] renamed to: "
        f"{_escape(new_title)}"
    )


def _cmd_init(ctx: CommandContext, args: str) -> None:
    """扫描项目并生成（或更新）AGENTS.md。命令本身只组装 prompt，扫描和写文件由模型完成。"""
    cwd = Path(ctx.cwd)
    target = cwd / "AGENTS.md"

    existing_block = ""
    if target.exists():
        try:
            existing = target.read_text(encoding="utf-8", errors="replace")[:10_000]
            existing_block = (
                f"\n\n## 当前 AGENTS.md 内容（请提改进 diff，不要静默覆盖）\n\n"
                f"```markdown\n{existing}\n```\n"
            )
            ctx.console.print(
                f"[dim]发现已有 AGENTS.md（{len(existing)} 字符），将提改进建议。[/dim]"
            )
        except OSError:
            pass
    else:
        ctx.console.print("[dim]未发现 AGENTS.md，将生成新文件。[/dim]")

    extra_hint = f"\n\n额外用户指示：{args.strip()}" if args.strip() else ""

    ctx.pending_query = _INIT_PROMPT_TEMPLATE.format(
        cwd=str(cwd),
        existing_block=existing_block,
        extra_hint=extra_hint,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_COMMAND_TABLE: list[tuple[str, str, object]] = [
    ("help",     "Show available commands",                         _cmd_help),
    ("clear",    "Clear conversation, start new session",           _cmd_clear),
    ("history",  "List saved sessions for this directory",          _cmd_history),
    ("resume",   "Resume a past session [number|session-id]",       _cmd_resume),
    ("compact",  "Compress conversation context [instructions]",    _cmd_compact),
    ("skills",   "List all available skills",                       _cmd_skills),
    ("cost",     "Show token usage and cost summary",               _cmd_cost),
    ("remember", "Append a note to today's memory log [text]",      _cmd_remember),
    ("memory",   "View MEMORY.md, list/open topic memories [list|<n>|<text>]", _cmd_memory),
    ("dream",    "Consolidate daily logs into persistent memories",  _cmd_dream),
    ("rename",   "Rename current session [new-title]",                    _cmd_rename),
    ("init",     "Scan project and write/update AGENTS.md [extra hints]", _cmd_init),
]

_HANDLERS: dict[str, object] = {name: h for name, _, h in _COMMAND_TABLE}


def handle_command(name: str, args: str, ctx: CommandContext) -> bool:
    """Dispatch slash command. Returns True if handled.

    内置命令优先；未匹配时尝试作为 skill 名称执行。
    """
    handler = _HANDLERS.get(name)
    if handler is not None:
        handler(ctx, args)  # type: ignore[operator]
        return True

    # 未匹配内置命令 → 尝试作为 skill 调用
    from features.skills import get_skill
    skill = get_skill(name)
    if skill is not None:
        return _execute_skill(skill, args, ctx)

    ctx.console.print(f"[red]Unknown command: /{name}[/red]  (try /help or /skills)")
    return False


def _execute_skill(skill, args: str, ctx: CommandContext) -> bool:
    """执行 skill：inline 模式注入当前对话，fork 模式在独立会话中运行。"""
    from tui.query import run_query
    from features.skills import mark_skill_invoked

    prompt = skill.get_prompt(args)
    if not prompt:
        ctx.console.print(f"[dim]Skill /{skill.name} produced no prompt.[/dim]")
        return True

    ctx.console.print(f"[dim]Running skill: /{skill.name}…[/dim]")
    # Phase B：记录这次调用，压缩时重注入 skill body 用
    mark_skill_invoked(skill.name)

    if skill.context == "fork":
        # fork 模式：保存当前消息，独立运行，恢复原消息
        saved = list(ctx.engine.get_messages())
        ctx.engine.set_messages([])
        try:
            run_query(ctx.engine, prompt, print_mode=False, permissions=ctx.permissions)
        finally:
            ctx.engine.set_messages(saved)  # 无论成功还是失败，都将消息列表恢复到执行前的状态
    else:
        # inline 模式：注入到当前对话
        run_query(ctx.engine, prompt, print_mode=False, permissions=ctx.permissions)

    return True
