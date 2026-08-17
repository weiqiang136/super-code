"""Run a single query turn with TUI feedback (spinner, markdown streaming)."""
from __future__ import annotations

import sys

from rich.console import Console
from rich.markup import escape as _escape

from core.engine import AbortedError, Engine, _REJECT_MESSAGE, _SIBLING_REJECT_MESSAGE
from tui.keylistener import EscListener
from tui.rendering import (StreamingMarkdown, SpinnerManager, tool_preview, collapsed_tool_summary,
                           SPINNER_THINKING, SPINNER_COMPACT, SPINNER_PREPARING, SPINNER_WORKING)

console = Console()


def run_query(engine: Engine, user_input: str, print_mode: bool,
              permissions=None, quiet: bool = False) -> None:

    listener = EscListener(on_cancel=engine.abort)  # 创建一个 Esc 键监听器，当用户按下 Esc 键时自动调用 engine.abort() 方法来中断引擎正在执行的任务
    spinner = SpinnerManager(console)
    md_stream = StreamingMarkdown(console)
    first_text = True       # 标记是否是第一次接收到文本内容，以便在首次收到文本时停止加载动画并切换到流式渲染模式
    streaming = False       # 追踪当前是否处于流式文本输出状态，配合 Esc 键监听实现中途取消功能。
    # 键(key)是 tool_use_id（API 分配的唯一 id），值(value)是 (工具名, 用于终端显示的带箭头格式化字符串)。
    # 使用 id 而非 "tool_name(preview)" 字符串作 key，避免同名工具（如两个 Grep 用相同 pattern）
    # 因 key 碰撞导致第二条记录覆盖第一条，进而使 pending_tools 永远无法清空，spinner 卡死。
    pending_tools: dict[str, tuple[str, str]] = {}

    try:
        with listener:
            if not quiet:
                spinner.start("Thinking…", SPINNER_THINKING)

            for event in engine.submit(user_input):
                if not quiet and streaming and listener.pressed:
                    md_stream.flush()
                    spinner.stop()
                    engine.cancel_turn()
                    console.print("\n[dim yellow]⏹ Turn cancelled (Esc)[/dim yellow]")
                    return

                if event[0] == "thinking":
                    if not quiet and first_text:
                        spinner.start("Thinking…", SPINNER_THINKING)

                elif event[0] == "compact":
                    # 轮内紧急压缩：在工具链中途触发，显示专用提示
                    if not quiet:
                        md_stream.flush()
                        spinner.start("Compacting context…", SPINNER_COMPACT)

                elif event[0] == "notification":
                    # worker 完成通知：engine 在 mid-turn 注入了通知到对话中
                    if not quiet:
                        md_stream.flush()
                        count = event[1].count("<task-notification>")
                        console.print(
                            f"[dim]Worker completed ({count}).[/dim]" if count > 1
                            else "[dim]Worker completed.[/dim]"
                        )
                        spinner.start("Thinking…", SPINNER_THINKING)

                elif event[0] == "text":
                    if quiet:
                        continue
                    if first_text:
                        spinner.stop()
                        streaming = True
                        first_text = False
                    if print_mode:
                        print(event[1], end="", flush=True)
                    else:
                        md_stream.feed(event[1])

                elif event[0] == "waiting":
                    if not quiet:
                        md_stream.flush()
                    streaming = False
                    if not quiet:
                        spinner.start("Preparing tool call…", SPINNER_PREPARING)

                elif event[0] == "tool_call":
                    if not quiet:
                        spinner.stop()
                        streaming = False
                        # event 格式: ("tool_call", tool_name, tool_input, activity, tool_use_id)
                        # 用 tool_use_id 作 key，保证同名工具（如两个 Grep）各自独立追踪，不会 key 碰撞
                        _, tool_name, tool_input, activity, tool_id = event
                        preview = tool_preview(tool_name, tool_input)
                        line = f"↳ {tool_name}({preview})"
                        pending_tools[tool_id] = (tool_name, line)

                elif event[0] == "tool_executing":
                    if not quiet:
                        # event 格式: ("tool_executing", tool_name, tool_input, activity, tool_use_id)
                        _, tool_name, tool_input, activity, tool_id = event
                        # 交互式工具（AskUserQuestion）执行期间需要独占 terminal 等用户输入，
                        # Rich Live spinner 的后台重绘会与 input() 的行编辑抢光标控制，导致
                        # 用户按键被吞、卡住直到按 Enter 才返回。这里显式不启 spinner，让
                        # 工具自己拥有 terminal；工具返回后 tool_result 路径正常 stop()。
                        if tool_name == "AskUserQuestion":
                            spinner.stop()
                        else:
                            n = len(pending_tools)
                            if n > 1:
                                names = [tn for tn, _ in pending_tools.values()]
                                spinner.start(collapsed_tool_summary(names), SPINNER_WORKING)
                            else:
                                _, line = pending_tools.get(tool_id, ("", f"↳ {tool_name}"))
                                activity_text = activity or f"Running {tool_name}…"
                                spinner.start(f"{line} … {activity_text}", SPINNER_WORKING)

                elif event[0] == "tool_result":
                    if not quiet:
                        spinner.stop()
                        # event 格式: ("tool_result", tool_name, tool_input, result, tool_use_id)
                        # 用 tool_use_id pop，精确匹配对应的 tool_call，不受工具名或参数重复影响
                        _, tool_name, tool_input, result, tool_id = event
                        tname, line = pending_tools.pop(tool_id, (tool_name, f"↳ {tool_name}"))
                        if result.is_error:
                            console.print(f"  [dim]{_escape(line)}[/dim] [red]✗[/red]", highlight=False)
                            # REJECT_MESSAGE / SIBLING_REJECT_MESSAGE 是给 LLM 看的硬
                            # 指令体（"STOP what you are doing..."），不是用户视角的错误
                            # 信息——用户主动拒的工具再把这串硬指令红字读一遍很多余。
                            # 真实工具失败（Bash 报错、Read 文件不存在等）仍然正常显示。
                            if result.content not in (_REJECT_MESSAGE, _SIBLING_REJECT_MESSAGE):
                                console.print(f"    [red]{_escape(result.content[:200])}[/red]")
                        else:
                            console.print(f"  [dim]{_escape(line)}[/dim] [green]✓[/green]", highlight=False)
                        if pending_tools:
                            names = [tn for tn, _ in pending_tools.values()]
                            spinner.start(collapsed_tool_summary(names), SPINNER_WORKING)
                        else:
                            streaming = False
                            spinner.start("Thinking…", SPINNER_THINKING)
                            first_text = True

                elif event[0] == "error":
                    if not quiet:
                        md_stream.flush()
                        spinner.stop()
                        console.print(f"\n[bold red]{_escape(event[1])}[/bold red]")

                elif event[0] == "turn_aborted_by_deny":
                    # engine 检测到本轮发生过 deny → 硬结束 turn，给一行收尾提示，
                    # 避免用户看到一堆 ✗ 后突然回输入框疑惑。
                    if not quiet:
                        md_stream.flush()
                        spinner.stop()
                        console.print("[dim yellow]⏹ Stopped after tool use rejected. Type your next message.[/dim yellow]")

            md_stream.flush()
            spinner.stop()
    except (AbortedError, KeyboardInterrupt):
        md_stream.flush()
        spinner.stop()
        if not isinstance(sys.exc_info()[1], AbortedError):
            engine.cancel_turn()
        if not quiet:
            console.print("\n[dim yellow]⏹ Turn cancelled[/dim yellow]")
        return
    finally:
        md_stream.flush()
        spinner.stop()

    if not print_mode:
        console.print()



















