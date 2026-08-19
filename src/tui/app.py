"""super-code entry point — argparse, engine setup, and interactive REPL."""
from __future__ import annotations

import argparse
import atexit
import sys
import threading
import time
import uuid
from pathlib import Path

from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markup import escape as _escape

from commands import CommandContext, handle_command, is_known_command, parse_command
from core.config import load_app_config
from core.context import build_system_prompt
from core.engine import Engine
from core.permissions import PermissionChecker
from core.session import SessionStore
from features.compact import CompactService, get_context_window, should_compact
from features.coordinator import (
    get_coordinator_system_prompt, get_coordinator_user_context,
    get_worker_system_prompt, is_coordinator_mode, set_coordinator_mode,
)
from features.cost_tracker import CostTracker
from features.memory import (
    get_memory_dir, append_to_daily_log, ensure_memory_dir, extract_memory_tags,
    build_dream_prompt, list_sessions_since, read_last_consolidated_at,
    release_lock, should_auto_dream, try_acquire_lock,
)
from features.plan import PlanModeManager
from features.skills import discover_skills, build_skills_prompt_section
from features.worker_manager import WorkerManager
from mcp.loader import load_mcp_tools, shutdown_mcp
from tools import AskUserQuestionTool, BashTool, FileEditTool, FileReadTool, FileWriteTool, GlobTool, GrepTool, \
    WebFetchTool, WebSearchTool
from tools.agent import AgentTool, SendMessageTool, TaskStopTool
from tui.prompt import bordered_prompt, slash_completer
from tui.query import run_query
from tui.rendering import SpinnerManager, SPINNER_MEMORY

console = Console()

_DOUBLE_PRESS_TIMEOUT = 0.8

_LOGO_LINES = [
    r"███████╗██╗   ██╗██████╗ ███████╗██████╗",
    r"██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗",
    r"███████╗██║   ██║██████╔╝█████╗  ██████╔╝",
    r"╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗",
    r"███████║╚██████╔╝██║     ███████╗██║  ██║",
    r"╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝",
]


def _fmt_tokens(n: int) -> str:
    """token 数格式化：≥1M 显示 M（整数省略小数位），否则显示 K。"""
    if n >= 1_000_000:
        m = n / 1_000_000
        return f"{m:.0f}M" if m == int(m) else f"{m:.1f}M"
    return f"{round(n / 1024)}K"


def _print_banner(app_config, cwd: str, session_id: str) -> None:
    """打印启动横幅：Panel 包裹，左侧紫色 ASCII logo，右侧模型信息。

    Context（上下文窗口）按模型动态映射（features/compact.get_context_window），
    与自动压缩触发阈值同源，保证展示和实际行为一致。
    """
    from rich.table import Table
    from rich.text import Text
    from features.compact import get_context_window

    logo = Text("\n".join(_LOGO_LINES), style="italic bold bright_magenta")

    info = Text()
    info.append("Model       : ", style="cyan"); info.append(f"{app_config.model}\n", style="bright_cyan")
    info.append("Context     : ", style="cyan"); info.append(f"{_fmt_tokens(get_context_window(app_config.model))}\n", style="bright_green")
    info.append("Max Output  : ", style="cyan"); info.append(f"{_fmt_tokens(app_config.max_tokens)}\n", style="bright_yellow")
    info.append("Session     : ", style="cyan"); info.append(f"{session_id[:8]}\n", style="bright_blue")
    info.append("CWD         : ", style="cyan"); info.append(f"{cwd}\n", style="bright_white")
    info.append("Version     : ", style="cyan"); info.append("v3.2.0", style="bold bright_magenta")

    table = Table.grid(padding=(0, 3))
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True, vertical="middle")
    table.add_row(logo, info)

    console.print()
    console.print(table)


def _run_dream(engine, memory_dir, permissions, quiet: bool = True,
               transcript_dir: str = "", session_ids: list | None = None) -> None:
    """执行 dream 整合：用 LLM 将日志提炼为持久记忆文件并更新 MEMORY.md。

    quiet=True 时静默运行（自动触发），quiet=False 时显示输出（手动 /dream）。
    dream 模式下权限隔离：只允许 Read/Glob/Grep/Edit/Write（限 memory_dir 内）。
    """
    if not quiet:
        console.print("[dim]Starting dream consolidation…[/dim]")
    permissions.enter_dream_mode(str(memory_dir))
    try:
        prompt = build_dream_prompt(memory_dir, transcript_dir=transcript_dir,
                                    session_ids=session_ids)
        run_query(engine, prompt, print_mode=False, permissions=permissions, quiet=quiet)
    finally:
        permissions.exit_dream_mode()


def _trigger_auto_dream_bg(app_config, memory_dir: Path, session_store) -> bool:
    """检查是否满足自动 dream 条件，满足则在后台线程启动 dream，立即返回。

    Returns True if dream was triggered, False otherwise.
    """
    current_sid = session_store.session_id if session_store else ""
    sessions_path = getattr(session_store, "_dir", None)
    if not should_auto_dream(memory_dir,
                             min_hours=app_config.dream_interval_hours,
                             min_sessions=app_config.dream_min_sessions,
                             current_session_id=current_sid,
                             sessions_dir=sessions_path):
        return False

    prior_mtime = read_last_consolidated_at(memory_dir)
    if not try_acquire_lock(memory_dir):
        return False

    sids = list_sessions_since(prior_mtime, sessions_dir=sessions_path,
                               current_session_id=current_sid)
    transcript_dir = str(sessions_path) if sessions_path else ""
    dream_perms = PermissionChecker(auto_approve=True)
    dream_engine = Engine(
        tools=[FileReadTool(), GlobTool(), GrepTool(), FileEditTool(), FileWriteTool()],
        system_prompt="",
        permission_checker=dream_perms,
        provider=app_config.provider,
        api_key=app_config.api_key,
        base_url=app_config.base_url,
        model=app_config.model,
        max_tokens=app_config.max_tokens,
        timeout=app_config.timeout,
        model_profiles=app_config.model_profiles,
    )

    def _worker():
        try:
            _run_dream(dream_engine, memory_dir, dream_perms, quiet=True,
                       transcript_dir=transcript_dir, session_ids=sids)
            release_lock(memory_dir)
        except Exception:
            from features.memory import _lock_path
            import os as _os
            try:
                lp = _lock_path(memory_dir)
                if lp.exists():
                    _os.utime(lp, (prior_mtime, prior_mtime))
            except OSError:
                pass

    threading.Thread(target=_worker, daemon=True).start()
    return True


def main() -> None:
    print("\033]0;super-code\007", end="")  # 设置终端标题
    parser = argparse.ArgumentParser(prog="super-code", description="Minimal AI coding assistant")
    parser.add_argument("prompt", nargs="?", help="Prompt to send (optional)")
    parser.add_argument("-p", "--print", action="store_true",
                        help="Non-interactive: print response and exit")
    parser.add_argument("--auto-approve", action="store_true",
                        help="Auto-approve all tool permissions")
    parser.add_argument("--config", help="Path to a JSON config file")
    parser.add_argument("--provider", choices=("openai",), help="API provider")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--max-tokens", type=int, help="Maximum output tokens")
    parser.add_argument("--mode", choices=("default", "dream"), default="default",
                        help="Permission mode: default (prompt for writes) or dream (auto-approve all)")
    parser.add_argument("--resume", metavar="SESSION", help="Resume a past session by ID or number")
    parser.add_argument("--coordinator", action="store_true",
                        help="Enable coordinator mode (orchestrate workers via Agent tool)")
    parser.add_argument("--sandbox", action="store_true",
                        help="Enable sandbox mode (block dangerous commands via blacklist, not OS-level isolation)")
    parser.add_argument("--auto-dream", action="store_true",
                        help="Disable automatic dream consolidation")
    parser.add_argument("--dream-interval", type=float, metavar="HOURS",
                        help="Hours between auto-dream runs (default: 24)")
    parser.add_argument("--dream-min-sessions", type=int, metavar="N",
                        help="Minimum new sessions before auto-dream triggers (default: 5)")
    args = parser.parse_args()

    try:
        app_config = load_app_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.coordinator or app_config.coordinator:
        set_coordinator_mode(True)

    cwd = str(Path.cwd())

    # 初始化记忆系统目录（按 git 仓库根隔离，非 git 目录回退到全局目录）
    memory_dir = get_memory_dir(Path(cwd))
    ensure_memory_dir(memory_dir)

    # 发现并注册 skill，注入系统提示词
    discover_skills(cwd)
    skills_section = build_skills_prompt_section()
    system_prompt = build_system_prompt(cwd=cwd, model=app_config.model, memory_dir=memory_dir)
    if skills_section:
        system_prompt = system_prompt + "\n\n" + skills_section

    # coordinator 模式：追加 worker 工具上下文 + coordinator 系统提示词
    worker_tool_names = ["Bash", "Read", "Edit", "Write", "Glob", "Grep"]
    if is_coordinator_mode():
        extra = get_coordinator_user_context(worker_tool_names)
        worker_context = extra.get("workerToolsContext")
        if worker_context:
            system_prompt += "\n\n# Coordinator Context\n" + worker_context
        system_prompt += "\n\n" + get_coordinator_system_prompt()

    # 沙箱：--sandbox 参数或 super-code.json 中 sandbox.enabled = true 均启用
    # 必须在 PermissionChecker 之前初始化，因为 auto_approve_if_sandboxed 依赖它
    from core.sandbox import SandboxManager, SandboxConfig
    _sandbox_cfg: SandboxConfig | None = None
    if args.sandbox or (app_config.sandbox or {}).get("enabled"):
        _sandbox_cfg = SandboxConfig.from_dict(app_config.sandbox)
        if args.sandbox:  # CLI 参数优先
            _sandbox_cfg.enabled = True
    sandbox = SandboxManager(_sandbox_cfg) if _sandbox_cfg else None

    permissions = PermissionChecker(auto_approve=args.auto_approve or args.mode == "dream",
                                    sandbox_manager=sandbox)

    # Session setup
    session_store = SessionStore(cwd=cwd, model=app_config.model)

    # 费用追踪器
    cost_tracker = CostTracker()

    # WorkerManager：每个 worker 拥有独立的 engine 实例
    def _build_worker_engine() -> Engine:
        return Engine(
            tools=[FileReadTool(sandbox_manager=sandbox), GlobTool(), GrepTool(),
                   BashTool(sandbox_manager=sandbox),
                   FileEditTool(sandbox_manager=sandbox), FileWriteTool(sandbox_manager=sandbox)],
            system_prompt=get_worker_system_prompt(),
            permission_checker=PermissionChecker(auto_approve=True),
            provider=app_config.provider,
            api_key=app_config.api_key,
            base_url=app_config.base_url,
            model=app_config.model,
            max_tokens=app_config.max_tokens,
            # worker 不挂 session_store（避免污染主会话 JSONL），显式提供 git-ai 钩子所需的最小信息
            repo_dir=cwd,
            agent_session_id=f"worker-{uuid.uuid4().hex[:8]}",
            timeout=app_config.timeout,
            model_profiles=app_config.model_profiles,
        )

    worker_manager = WorkerManager(build_worker_engine=_build_worker_engine)

    # 主 engine 工具列表（含 AgentTool + Skill）
    mcp_tools = load_mcp_tools(cwd)   # 读取 .mcp.json，启动 MCP server，返回工具代理列表
    from tools.skill import SkillTool   # 局部 import：避免 worker engine 误带
    tools = [
        FileReadTool(sandbox_manager=sandbox), GlobTool(), GrepTool(), BashTool(sandbox_manager=sandbox),
        FileEditTool(sandbox_manager=sandbox), FileWriteTool(sandbox_manager=sandbox),
        AskUserQuestionTool(), WebFetchTool(), WebSearchTool(),
        AgentTool(worker_manager), SendMessageTool(worker_manager), TaskStopTool(worker_manager),
        SkillTool(),
        *mcp_tools,
    ]

    engine = Engine(
        tools=tools,
        system_prompt=system_prompt,
        permission_checker=permissions,
        provider=app_config.provider,
        api_key=app_config.api_key,
        base_url=app_config.base_url,
        model=app_config.model,
        max_tokens=app_config.max_tokens,
        session_store=session_store,
        cost_tracker=cost_tracker,
        timeout=app_config.timeout,
        model_profiles=app_config.model_profiles,
    )

    # Plan mode manager — 先创建，再绑定 engine（避免循环依赖）
    plan_manager = PlanModeManager()
    plan_manager.bind_engine(engine)
    plan_manager.set_permissions(permissions)
    permissions.set_plan_manager(plan_manager)

    # Compact service — 复用 engine 内部的 LLMClient
    # cost_tracker 注入：压缩成功后记账 + 覆盖 last_input_tokens（底部栏 ctx 占用率）
    compact_service = CompactService(client=engine._client, model=app_config.model,
                                     cost_tracker=cost_tracker)
    engine.set_compact_service(compact_service)  # 注入 engine 供轮内紧急压缩使用

    # 注入 worker 通知回调：engine 在每轮工具执行完成后 drain 通知队列，
    # 将已完成 worker 的结果注入 conversation，coordinator 在同一 turn 内自动感知。
    def _check_worker_notifications():
        notifications = worker_manager.drain_notifications()
        return "\n\n".join(notifications) if notifications else None

    engine.set_on_after_tools(_check_worker_notifications)

    def _new_session_store() -> SessionStore:
        store = SessionStore(cwd=cwd, model=app_config.model)
        engine.set_session_store(store)
        return store

    cmd_ctx = CommandContext(   # 把当前程序运行所需的核心对象和状态打包成一个上下文对象，方便后续统一访问
        engine=engine,
        session_store=session_store,
        console=console,
        cwd=cwd,
        model=app_config.model,
        permissions=permissions,
        new_session_store=_new_session_store,
        compact_service=compact_service,
        plan_manager=plan_manager,
        worker_manager=worker_manager,
        cost_tracker=cost_tracker,
        memory_dir=memory_dir,
    )

    # --resume flag: load a past session before starting
    if args.resume:
        handle_command("resume", args.resume, cmd_ctx)

    # Non-interactive mode
    if args.print or args.prompt:
        prompt_text = args.prompt or sys.stdin.read()
        run_query(engine, prompt_text, print_mode=args.print, permissions=permissions)
        return

    # Interactive REPL
    # 进程级 stdout/stderr patch：把所有 print / console.print 路由到 prompt_toolkit
    # 的 StdoutProxy，避免后台线程（extract_memories 等）在 bordered_prompt 运行期间
    # 直接写终端、把输出糊进输入框。
    # - 无 active prompt 时（流式输出、spinner、命令执行）走真实 stdout，行为不变
    # - 有 active prompt 时由 proxy 缓冲 + run_in_terminal 安全插入输入框上方
    # - raw=True 保留 ANSI 序列，Rich 颜色不丢
    # - Rich Console.file 是动态 property，自动跟随 sys.stdout，无需重建已有 console
    # 任意环境异常（如 PyInstaller stderr=None / 无 Win32 console）直接降级 no-op，
    # 不阻塞 REPL 启动。
    try:
        from prompt_toolkit.patch_stdout import StdoutProxy
        _stdout_proxy = StdoutProxy(raw=True)
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _stdout_proxy
        if sys.stderr is not None:
            sys.stderr = _stdout_proxy

        def _restore_stdio():
            sys.stdout = _orig_stdout
            if _orig_stderr is not None:
                sys.stderr = _orig_stderr
            try:
                _stdout_proxy.close()
            except Exception:
                pass
        atexit.register(_restore_stdio)
    except Exception:
        pass

    _print_banner(app_config, cwd, session_store.session_id)

    # 启动 git-ai daemon（如果已安装）：后台线程，不阻塞 UI 显示。
    # checkpoint 数据只存在 daemon 内存中，电脑重启后 daemon 不在运行，
    # 提前启动可避免 checkpoint 丢失导致 commit 归属失败。
    # 若后台启动尚未完成时发生首次编辑，before_edit() 内部会同步补启动（幂等）。
    from features.git_ai import ensure_daemon
    threading.Thread(target=ensure_daemon, daemon=True).start()

    # 历史记录文件，保存在 memory_dir 同级目录
    history_file = memory_dir.parent / "repl_history"   # 用户可以通过上下方向键回溯之前的input
    pt_history = FileHistory(str(history_file))
    mode_ref = [False]  # [False]=normal, [True]=plan，传给 bordered_prompt 共享状态

    def _toggle_plan_mode() -> None:
        """Shift+Tab 回调：真正切换 plan mode，同时同步 UI 状态。
        不打印任何消息——边框颜色变化已是足够的视觉反馈。"""
        if plan_manager.is_active:
            plan_manager.exit()
            mode_ref[0] = False
        else:
            plan_manager.enter()
            mode_ref[0] = True

    last_ctrlc_time = 0.0
    # 自动压缩熔断器：连续失败计数。达到阈值后本 session 不再尝试 autocompact，
    # 避免一次失败（PTL/网络/超时等）后每一轮用户输入都重新触发并失败，浪费 API 调用。
    # 仅对自动压缩生效；用户主动敲 /compact 不受限。成功压缩一次即清零。
    consecutive_compact_failures = 0
    MAX_CONSECUTIVE_COMPACT_FAILURES = 3

    def _process_pending_notifications() -> bool:
        """Drain worker 通知队列并喂给 coordinator（plan mode 下跳过）。

        Returns True if notifications were processed, False if queue was empty.
        """
        if plan_manager.is_active:
            return False
        notifications = worker_manager.drain_notifications()
        if not notifications:
            return False
        count = len(notifications)
        console.print(f"[dim]Worker completed ({count}).[/dim]" if count > 1
                      else "[dim]Worker completed.[/dim]")
        combined = "\n\n".join(notifications)
        try:
            run_query(engine, combined, print_mode=False, permissions=permissions)
        except KeyboardInterrupt:
            engine.cancel_turn()
            console.print("\n[dim yellow]⏹ Turn cancelled[/dim yellow]")
        except Exception as e:
            engine.cancel_turn()
            console.print(f"\n[red]Failed to process worker notifications:[/red] {_escape(str(e))}")
        return True

    while True:
        try:
            # 底部栏 ctx 占用率：用最近一次 API 返回的 input_tokens（精确计数），
            # 除以模型 context window。0 = 尚未调用 API，显示层会隐藏。
            ctx_usage = [cmd_ctx.cost_tracker.last_input_tokens if cmd_ctx.cost_tracker else 0,
                         get_context_window(cmd_ctx.model)]
            user_input = bordered_prompt(console, history=pt_history,
                                         completer=slash_completer, mode_ref=mode_ref,
                                         on_mode_toggle=_toggle_plan_mode,
                                         session_title=cmd_ctx.session_store._title,
                                         ctx_usage=ctx_usage,
                                         # worker 进度面板：仅协调者模式启用
                                         # （get_panel_status 是线程安全快照，含完成态；普通模式传 None 零影响）
                                         worker_status_cb=worker_manager.get_panel_status
                                         if is_coordinator_mode() else None)
            if user_input is None:
                user_input = ""
            user_input = user_input.strip()
        except KeyboardInterrupt:
            now = time.monotonic()
            if now - last_ctrlc_time <= _DOUBLE_PRESS_TIMEOUT:
                console.print("\n[dim]Goodbye.[/dim]")
                break
            last_ctrlc_time = now
            console.print("\n[dim yellow]Press Ctrl+C again to exit[/dim yellow]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye.[/dim]")
            break

        last_ctrlc_time = 0.0

        # 每次输入后先检查 worker 通知（空回车也能触发）。
        # 同 mode 的待处理命令一次性 drain 喂给 LLM，N 条 <task-notification> 合并成 1 轮 run_query：
        #   - 减少 N 倍的 LLM 调用开销
        #   - 避免协调者对每条通知都独立"接电话"，进而触发 N 次冗余的 git status 二次验证
        _process_pending_notifications()

        if not user_input:
            continue

        # 用户提交了新一轮输入：清除已完成 worker 记录（保留运行中），
        # 面板完成态随之消失，不再常驻输入框上方
        worker_manager.clear_finished()

        if user_input.startswith("!") and len(user_input) > 1:
            import subprocess
            result = subprocess.run(user_input[1:].lstrip(),
                                    shell=True, capture_output=True, text=True)
            if result.stdout:
                console.print(result.stdout.rstrip())
            if result.stderr:
                console.print(f"[red]{result.stderr.rstrip()}[/red]")
            continue

        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        # Slash commands
        # 仅当 / 开头的 token 是**已注册的命令名**时才走命令分支；
        # 否则（如用户问"/query 这个接口的路径是什么"）按普通输入交给 LLM，
        # 避免把用户的自然语言误判为 "Unknown command"。
        parsed = parse_command(user_input)
        if parsed and is_known_command(parsed[0]):
            name, cmd_args = parsed
            # 用 try/except 包住命令执行：斜杠命令（如 /compact）内部可能阻塞在
            # 同步网络调用（OpenAI 非流式 create）上，用户按 Ctrl+C 时 KeyboardInterrupt
            # 会从 ssl.recv 一路抛出。若不在此拦截，异常会逃出主循环，导致程序退出
            # （PyInstaller 打包后还会显示 "Failed to execute script"）。
            # 普通对话路径有 run_query 内部的 catch 兜底，命令路径之前是裸调，是结构性缺口。
            # 注：当前所有内置命令都遵循"先调 API、后改本地状态"的顺序，中途取消不会留下半状态。
            try:
                handle_command(name, cmd_args, cmd_ctx)
                # /plan <desc> 可能设置 pending_query，触发一次模型查询
                if cmd_ctx.pending_query:
                    query = cmd_ctx.pending_query
                    cmd_ctx.pending_query = None
                    run_query(engine, query, print_mode=False, permissions=permissions)
            except KeyboardInterrupt:
                # 清空可能残留的 pending_query，避免下一轮被误触发
                cmd_ctx.pending_query = None
                # 兜底回滚：斜杠命令内部若已向 session_store 写入半态（如调 LLM 时
                # 留下孤立 tool_use），这里必须显式 cancel_turn；run_query 的兜底
                # 够不到这条命令路径。多调一次是幂等的（checkpoint 为 None 时 no-op）。
                engine.cancel_turn()
                console.print("\n[dim yellow]⏹ Command cancelled[/dim yellow]")
            except Exception as e:
                # 命令执行抛业务异常（典型场景：/compact 调 LLM 失败、网络断开、鉴权
                # 失败、PTL 兜底 3 次全失败、dirty session 触发 400 等）。
                # 不接住的话异常会逃出主循环 → 整个 TUI 当场退出（PyInstaller 打包后
                # 还会显示 "Failed to execute script"），用户被踢出会话。
                # 仅 catch Exception（不含 KeyboardInterrupt/SystemExit）：保留 Ctrl+C
                # 与正常退出路径。
                # 不调用 engine.cancel_turn()：内置命令均不通过 engine.submit() 推进
                # 对话轮次，cancel_turn 会按上一轮成功 turn 的 checkpoint 误删有效历史。
                cmd_ctx.pending_query = None
                console.print(f"\n[red]Command failed:[/red] {_escape(str(e))}")
            continue

        # Step 6：把"按相关性精选的记忆"作为 <system-reminder> 前缀注入到 user_input。
        # 只在交互式普通对话路径生效；slash 命令路径不走这里，不会被污染。
        # plan mode 跳过（与 extract / dream 同策略，避免计划阶段噪音）。
        # 失败 / 无匹配 → 返回空串，prefix + user_input 自然降级为 user_input。
        memory_prefix = ""
        if not plan_manager.is_active:
            try:
                from features.find_relevant_memories import build_relevant_memories_prefix, will_need_side_query
                # extract_model 是 Step 6 性能优化：用更便宜的小模型跑 selector；
                # 未配置（空串）则回退到主模型，行为与早期版本一致
                selector_model = app_config.extract_model or app_config.model
                _needs_llm = will_need_side_query(user_input, memory_dir)
                if _needs_llm:
                    _mem_spinner = SpinnerManager(console)
                    _mem_spinner.start("Searching memories…", SPINNER_MEMORY)
                try:
                    memory_prefix = build_relevant_memories_prefix(
                        user_input, memory_dir, engine._client, selector_model,
                    )
                finally:
                    if _needs_llm:
                        _mem_spinner.stop()
            except Exception:
                # 任何意外（例如 side-query 模型 404）都不应阻塞用户提问
                memory_prefix = ""

        # 普通对话路径同样需要 API 异常兜底：402 余额不足、401 鉴权、provider 5xx、网络断开
        # 等都会从 openai SDK 经 engine.submit() 一路冒上来，run_query 内部只接 Abort/Ctrl+C。
        # 不接住的话异常会逃出主循环 → 整个 TUI 当场退出。处理方式与 worker 通知路径同源。
        try:
            run_query(engine, memory_prefix + user_input, print_mode=False, permissions=permissions)
        except KeyboardInterrupt:
            engine.cancel_turn()
            console.print("\n[dim yellow]⏹ Turn cancelled[/dim yellow]")
        except Exception as e:
            engine.cancel_turn()
            console.print(f"\n[red]Query failed:[/red] {_escape(str(e))}")

        # 用户 query 完成后，自动处理在此期间完成的 worker 通知。
        # 循环收敛：coordinator 收到通知后可能 spawn 新 worker，新 worker 可能
        # 在本轮回复期间完成，所以 drain 到队列为空才停止。
        # 硬上限 MAX_AUTO_DRAIN_ROUNDS 防止 coordinator 无限 spawn→complete 循环。
        MAX_AUTO_DRAIN_ROUNDS = 5
        for _ in range(MAX_AUTO_DRAIN_ROUNDS):
            if not _process_pending_notifications():
                break

        # 从 assistant 输出中提取 <system_reminder> 标签，追加到当天日志
        for tag in extract_memory_tags(engine.last_assistant_text()):
            append_to_daily_log(memory_dir, tag)

        # Step 5：后台抽取本轮新增对话里值得持久化的偏好 / 事实。fire-and-forget；
        # 内部已做：节流（<1 条新消息跳过）/ 互斥（重叠运行跳过）/ 主智能体已写则跳过。
        # plan mode 下跳过，与 auto-dream 一致：避免计划过程中的临时讨论被当成记忆。
        # 不再有"已保存"通知：后台 print 会撞进主线程 spinner/流式输出造成 UI 错位
        # （"⠦ Thinking…💾 Saved ..."），且记忆系统有 /memory 显式入口可查，通知是
        # 多余信息。
        if not plan_manager.is_active:
            from features.extract_memories import execute_extract_memories
            execute_extract_memories(engine.get_messages(), app_config, memory_dir)

        # auto-dream 门检查（plan mode 下跳过）
        if not plan_manager.is_active and app_config.auto_dream:
            if _trigger_auto_dream_bg(app_config, memory_dir, session_store):
                console.print("[dim]Dreaming in background…[/dim]")

        # 每轮结束后检查是否需要自动压缩（plan mode 下跳过；熔断后跳过）
        if not plan_manager.is_active and consecutive_compact_failures < MAX_CONSECUTIVE_COMPACT_FAILURES:
            messages = engine.get_messages()
            if should_compact(messages, model=app_config.model):
                console.print("[dim]Auto-compacting conversation context…[/dim]")
                # 只捕获 Exception（不含 KeyboardInterrupt/SystemExit）：保留用户 Ctrl+C
                # 中断 autocompact 的既有逃逸路径，不把"用户取消"误计入熔断失败次数。
                try:
                    handle_command("compact", "", cmd_ctx)
                    consecutive_compact_failures = 0  # 成功 → 计数清零
                except Exception as e:
                    consecutive_compact_failures += 1
                    console.print(
                        f"[yellow]Auto-compact failed "
                        f"({consecutive_compact_failures}/{MAX_CONSECUTIVE_COMPACT_FAILURES}): "
                        f"{_escape(str(e))}[/yellow]"
                    )
                    if consecutive_compact_failures >= MAX_CONSECUTIVE_COMPACT_FAILURES:
                        console.print(
                            "[yellow]Auto-compact disabled for this session after repeated failures. "
                            "Use /compact to retry manually.[/yellow]"
                        )

    # 退出时追加会话摘要到当天日志
    messages = engine.get_messages()
    if messages:
        append_to_daily_log(memory_dir, f"Session ended. {len(messages)} messages exchanged.")

    # 退出时关闭所有 MCP server 子进程
    shutdown_mcp()

    # 退出时打印费用摘要
    if cost_tracker.total_cost_usd > 0 or cost_tracker.last_input_tokens > 0:
        console.print(f"\n[dim]{cost_tracker.format_cost()}[/dim]")


if __name__ == "__main__":
    main()
