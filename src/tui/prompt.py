"""带边框的输入框 + 斜杠命令补全。"""
from __future__ import annotations

import os
import sys
from typing import Callable

from prompt_toolkit.application import Application as PTApp
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window, FloatContainer, Float
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text


def _cursor_distance_to_bottom() -> int:
    """返回光标到终端窗口底部的行数。获取失败返回一个大值（不添加 spacer）。"""
    if sys.platform != 'win32':
        return 999  # 非 Windows 暂不处理

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE

    class _COORD(ctypes.Structure):
        _fields_ = [('X', wintypes.SHORT), ('Y', wintypes.SHORT)]

    class _SMALL_RECT(ctypes.Structure):
        _fields_ = [('Left', wintypes.SHORT), ('Top', wintypes.SHORT),
                    ('Right', wintypes.SHORT), ('Bottom', wintypes.SHORT)]

    class _CSBI(ctypes.Structure):
        _fields_ = [('dwSize', _COORD), ('dwCursorPosition', _COORD),
                    ('wAttributes', wintypes.WORD), ('srWindow', _SMALL_RECT),
                    ('dwMaximumWindowSize', _COORD)]

    csbi = _CSBI()
    if kernel32.GetConsoleScreenBufferInfo(h, ctypes.byref(csbi)):
        return max(0, csbi.srWindow.Bottom - csbi.dwCursorPosition.Y)
    return 999


class SlashCommandCompleter(Completer):
    """输入 / 时触发斜杠命令补全。"""

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith('/'):
            return

        query = text[1:].lower()

        # 内置命令
        from commands import _COMMAND_TABLE
        for name, desc, _ in _COMMAND_TABLE:
            if not query or name.startswith(query):
                yield Completion(
                    f'/{name}',
                    start_position=-len(text),
                    display=f'/{name}',
                    display_meta=desc,
                )

        # 动态 skill 命令
        try:
            from features.skills import list_skills
            builtin_names = {name for name, _, _ in _COMMAND_TABLE}
            for skill in list_skills(user_invocable_only=True):
                if skill.name in builtin_names:
                    continue
                if not query or skill.name.startswith(query):
                    yield Completion(
                        f'/{skill.name}',
                        start_position=-len(text),
                        display=f'/{skill.name}',
                        display_meta=skill.description[:40] if skill.description else 'skill',
                    )
        except Exception:
            pass


slash_completer = SlashCommandCompleter()


def bordered_prompt(
    con: Console,
    history: FileHistory | None = None,
    completer: Completer | None = None,
    mode_ref: list | None = None,       # [False]=normal, [True]=plan，可变列表在闭包间共享状态
    on_mode_toggle=None,                # 切换模式时的回调，由 app.py 传入，负责调用 plan_manager
    session_title: str = "",            # 当前会话标题，/rename 设置后显示在上边框
    ctx_usage: list | None = None,      # [已用 token, 窗口 token]，None 时不显示占用率
    worker_status_cb: Callable[[], list[dict]] | None = None,  # 返回运行中 worker 状态列表，None 时不显示进度面板
) -> str:
    """带上下边框的输入框，输入 / 时弹出补全菜单，Shift+Tab 切换模式。

    Raises KeyboardInterrupt on Ctrl+C, EOFError on Ctrl+D with empty buffer.
    """
    if mode_ref is None:
        mode_ref = [False]

    # ===== 粘贴占位符机制 =====
    # 注册表：[(占位符文本, 原始粘贴内容), ...]
    _paste_registry: list[tuple[str, str]] = []
    _paste_counter = 0                   # 自增编号，确保每个占位符唯一
    _last_text = ""                      # 上一次 buffer 文本，用于 diff 检测

    def _accept(b):
        """提交时将所有粘贴占位符展开为原始内容。"""
        text = b.text
        for placeholder, actual in _paste_registry:
            text = text.replace(placeholder, actual)
        get_app().exit(result=text)
        return True

    buf = Buffer(
        history=history,
        completer=completer,
        complete_while_typing=False,
        accept_handler=_accept,
    )

    def _trigger_completion_next_tick():
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon(lambda: buf.start_completion(select_first=False))
        except RuntimeError:
            pass

    def _on_text_changed(_buf):
        """文本变化回调：检测多行粘贴并替换为占位符；同时处理 / 补全触发。"""
        nonlocal _last_text, _paste_counter

        current = _buf.text

        # --- 粘贴检测：通过 diff 找到本次插入的文本 ---
        if current != _last_text and len(current) > len(_last_text):
            old = _last_text
            # 找到新旧文本的公共前缀
            i = 0
            while i < len(old) and i < len(current) and old[i] == current[i]:
                i += 1
            # 找到新旧文本的公共后缀
            j_old = len(old) - 1
            j_new = len(current) - 1
            while j_old >= i and j_new >= i and old[j_old] == current[j_new]:
                j_old -= 1
                j_new -= 1
            inserted = current[i:j_new + 1]

            # 多行插入 → 判定为粘贴，替换为占位符
            if '\n' in inserted:
                _paste_counter += 1
                n_lines = inserted.count('\n') + 1
                placeholder = f'[Pasted text #{_paste_counter} +{n_lines} lines]'
                _paste_registry.append((placeholder, inserted))

                # 用占位符替换粘贴内容（临时移除回调避免递归触发）
                new_text = current[:i] + placeholder + current[j_new + 1:]
                _buf.on_text_changed -= _on_text_changed
                try:
                    _buf.text = new_text
                    _buf.cursor_position = i + len(placeholder)
                finally:
                    _buf.on_text_changed += _on_text_changed

                _last_text = new_text
                return

        _last_text = current

        # 原有逻辑：输入 / 时触发补全
        if current.lstrip().startswith('/'):
            _trigger_completion_next_tick()

    buf.on_text_changed += _on_text_changed

    _BAR = "\u2500"          # ─  box-drawing horizontal
    _TITLE_PREFIX = " 📝 "
    _TITLE_SUFFIX = " "

    def _render_top_bar(width: int, title: str, color: str) -> list[tuple[str, str]]:
        """渲染上边框：有标题时居中显示 📝 {title}，无标题时全 ─ 填充。"""
        if not title:
            fill = _BAR * max(0, width - 1)
            return [(color, f'{_BAR}{fill}')]

        min_frame = 4  # 标题两端至少保留 ──（各 2 字符）
        # 标题最大列宽 = 终端宽度 - 边框开销（cell_len 按显示列宽计，中文/emoji 占 2 列，
        # len() 只数字符数会低估宽度导致标题溢出边框），再加 0.6 比例上限，
        # 防止超宽终端上标题撑满整条边框、左右 ─ 填充几乎消失。
        max_title = min(width - cell_len(_TITLE_PREFIX) - cell_len(_TITLE_SUFFIX) - min_frame,
                        int(width * 0.6))
        if max_title <= 0:
            fill = _BAR * max(0, width - 1)
            return [(color, f'{_BAR}{fill}')]

        # 先清洗换行符再截断：自动标题取自第一条用户消息（session.py _generate_title），
        # 多行消息会带 \n。若 \n 带入 FormattedText，右侧 ─ 填充会被挤到第二行，
        # 而 _top 的 Window height=1 只显示第一行 → 上边框右半"缺一块"。
        # 注意清洗必须在 truncate 之前：Rich truncate 按列宽截断时 \n 占 0 列，
        # 截断后再 replace 成空格会放大列宽（0→1 列/个），照样撑爆 remaining。
        _title_text = Text(title.replace("\r\n", " ").replace("\n", " ").replace("\r", " "))
        _title_text.truncate(max_title, overflow="ellipsis")  # 原地修改，返回 None，不能链式
        display_title = _title_text.plain
        title_segment = f"{_TITLE_PREFIX}{display_title}{_TITLE_SUFFIX}"
        remaining = width - cell_len(title_segment) - 1  # -1 for leading ─
        left_fill = max(0, remaining // 2)
        right_fill = max(0, remaining - left_fill)
        return [(color, _BAR + _BAR * left_fill + title_segment + _BAR * right_fill)]

    def _top():
        try:
            w = os.get_terminal_size().columns
        except OSError:
            w = 80
        color = 'bold fg:ansiyellow' if mode_ref[0] else 'bold fg:ansiwhite'
        return _render_top_bar(w, session_title, color)

    def _bot():
        try:
            w = os.get_terminal_size().columns
        except OSError:
            w = 80
        # 左：模式标签；右：ctx 进度条 + 百分比；中间 ─ 填充。极简：无快捷键提示。
        mode_label = f"{_BAR} [Plan Mode] " if mode_ref[0] else f"{_BAR} [Normal] "
        base_color = 'fg:ansiyellow' if mode_ref[0] else 'fg:ansiwhite'

        # 上下文占用率：使用最近一次 API 返回的 input_tokens（tokenizer 精确计数），
        # 除以模型 context window 得到占用百分比。<70% 绿 / 70-90% 黄（逼近压缩阈值）/ >=90% 红。
        # 1M 大窗口下小占用会截断成 0%，用 round 四舍五入、不足 1% 显示 <1%。
        # 30 格 ▰/▱ 进度条（每格 3.3%）靠右，百分比在条后同色。
        # 空心部分用同色调暗色 hex（prompt_toolkit 不支持 dim 属性，会 ValueError）。
        _DIM_HEX = {'ansigreen': '#1f5e2a', 'ansiyellow': '#6b5e00', 'ansired': '#7a2020'}
        ctx_filled, ctx_empty, ctx_pct, ctx_color, ctx_dim = "", "", "", None, None
        if ctx_usage is not None and ctx_usage[0]:
            used, window = ctx_usage[0], ctx_usage[1]
            if window:
                pct = min(100, round(used * 100 / window))
                filled = round(pct / (100 / 30))  # 30 格，每格 3.3%
                ctx_filled = "▰" * filled
                ctx_empty = "▱" * (30 - filled)
                ctx_pct = f" {pct}% " if pct else " <1% "
                state = ('ansired' if pct >= 90
                         else 'ansiyellow' if pct >= 70
                         else 'ansigreen')
                ctx_color = f'bold fg:{state}'
                ctx_dim = f'fg:{_DIM_HEX[state]}'

        right_extra = ctx_filled + ctx_empty + ctx_pct
        fill = _BAR * max(0, w - 1 - len(mode_label) - len(right_extra))
        segments: list[tuple[str, str]] = [(base_color, f'{_BAR}{mode_label}{fill}')]
        if ctx_color:
            segments.append((ctx_color, ctx_filled))
            segments.append((ctx_dim, ctx_empty))  # 同色调暗色，整条颜色统一（实心亮/空心暗）
            segments.append((ctx_color, ctx_pct))
        segments.append((base_color, _BAR))
        return segments

    # ===== worker 进度面板（输入框上方，协调者模式常驻）=====
    # 运行中 worker 全展示（彩色实时活动）；完成的保留为 ✓/✗ 状态行不消失
    # （输入框位置稳定，不在 worker 完成时跳动）；完成行保留最近 _WORKER_DONE_MAX
    # 个、更早的折叠；无任何 worker 时显示占位行。数据由外部回调注入（拉取式快照）。
    _WORKER_DONE_MAX = 5       # 完成的 worker 最多保留展示的行数，更早的折叠
    _WORKER_PANEL_MIN_WIDTH = 40   # 终端列宽低于此值时不显示面板（最小内容也会溢出）
    _WORKER_ITEM_PREFIX = " ⚙ "   # 条目前缀（box-drawing 竖线 + 齿轮，延续边框语言）
    _WORKER_MORE_INDENT = "  "      # 折叠行缩进（对齐条目内容区，区别于 ⚙ 条目）
    _WORKER_GRAY = '#888888'        # 完成态灰色（prompt_toolkit 无 dim，用灰 hex）
    _WORKER_DONE_GREEN = '#2e8b57'  # ✓ 对勾暗绿（低调不抢视线）

    def _render_worker_item(wk: dict, width: int) -> list[tuple[str, str]]:
        """单行条目：│ ⚙ 描述 · 状态/活动 · N tools，按状态分支配色。

        running：描述 cyan、活动 yellow、工具数 green（实时动态最醒目）；
        completed/killed/failed：✓/✗ 状态色 + 灰色描述，整行弱化。
        宽度用 cell_len 计显示列宽（中文/emoji 占 2 列），超宽截断。
        """
        desc = (wk.get("description") or "Worker").strip()
        tools = wk.get("tool_uses", 0)
        tools_str = f"{tools} tools"
        sep = " · "
        prefix_w = cell_len(_WORKER_ITEM_PREFIX)
        inner_budget = max(1, width - prefix_w)
        status = wk.get("status", "running")

        if status == "running":
            act = (wk.get("activity") or "Idle").strip()
            d = Text(desc)
            d.truncate(min(24, inner_budget // 3), overflow="ellipsis")
            d_text = d.plain
            fixed_w = cell_len(tools_str) + cell_len(sep) * 2
            a = Text(act)
            a.truncate(max(4, inner_budget - cell_len(d_text) - fixed_w), overflow="ellipsis")
            return [
                ('fg:ansiwhite', _WORKER_ITEM_PREFIX),
                ('fg:ansicyan', d_text + sep),
                ('fg:ansiyellow', a.plain + sep),
                ('fg:ansigreen', tools_str),
            ]

        # 完成态：✓/✗ + 状态词 + 工具数，整行灰色弱化
        if status == "completed":
            mark, mark_style, state_text = "✓", f'fg:{_WORKER_DONE_GREEN}', "已完成"
        elif status == "killed":
            mark, mark_style, state_text = "✗", 'fg:ansiyellow', "已停止"
        elif status == "failed":
            mark, mark_style, state_text = "✗", 'fg:ansired', "失败"
        else:   # idle 等防御分支：无状态词
            mark, mark_style, state_text = "", "", ""
        fixed = cell_len(tools_str)
        fixed += cell_len(sep) * 2 + cell_len(state_text) if state_text else cell_len(sep)
        if mark:
            fixed += cell_len(mark) + 1
        d = Text(desc)
        d.truncate(max(4, inner_budget - fixed), overflow="ellipsis")
        d_text = d.plain
        segments: list[tuple[str, str]] = [('fg:ansiwhite', _WORKER_ITEM_PREFIX)]
        if mark:
            segments.append((mark_style, mark + " "))
        segments.append((_WORKER_GRAY, d_text))
        if state_text:
            segments.append((_WORKER_GRAY, sep + state_text))
        segments.append((_WORKER_GRAY, sep + tools_str))
        return segments

    def _workers_panel() -> list[tuple[str, str]]:
        """面板 text callable：协调者模式常驻；运行中全展示、完成态保留、无任务占位。

        不做标题条/边框线：上边框已有 ─ 线，再铺一条会视觉重复（用户实测反馈）。
        条目自带 │ ⚙ 前缀 + 颜色分段，独立成行已足够区分。
        """
        if worker_status_cb is None:
            return [("", "")]
        try:
            w = os.get_terminal_size().columns
        except OSError:
            w = 80
        try:
            workers = worker_status_cb()
        except Exception:
            return [("", "")]   # 渲染层防御：状态回调异常不应崩掉整个 REPL
        if w < _WORKER_PANEL_MIN_WIDTH:
            return [("", "")]
        if not workers:
            return [(_WORKER_GRAY, f"{_WORKER_ITEM_PREFIX}暂无任务")]
        running = [wk for wk in workers if wk.get("status", "running") == "running"]
        done = [wk for wk in workers if wk.get("status", "running") != "running"]
        # 运行中全展示；完成的保留最近 _WORKER_DONE_MAX 个（spawn 序靠后 = 最近）
        hidden = max(0, len(done) - _WORKER_DONE_MAX)
        shown = running + (done[hidden:] if hidden else done)
        segments: list[tuple[str, str]] = []
        for i, wk in enumerate(shown):
            segments.extend(_render_worker_item(wk, w))
            if i < len(shown) - 1:
                segments.append(("", "\n"))
        if hidden:
            segments.append(("", "\n"))
            segments.append(('fg:ansiyellow',
                             f"{_WORKER_MORE_INDENT}… and {hidden} more"))
        return segments

    def _panel_height() -> Dimension:
        """面板高度 callable：未启用或终端过窄时 0；否则按行数取值，至少 1 行（占位）。"""
        if worker_status_cb is None:
            return Dimension(min=0, preferred=0)
        try:
            w = os.get_terminal_size().columns
        except OSError:
            w = 80
        try:
            workers = worker_status_cb()
        except Exception:
            return Dimension(min=0, preferred=0)
        if w < _WORKER_PANEL_MIN_WIDTH:
            return Dimension(min=0, preferred=0)
        if not workers:
            return Dimension(min=1, preferred=1, max=1)   # 占位行
        n_running = sum(1 for wk in workers if wk.get("status", "running") == "running")
        n_done = len(workers) - n_running
        n = n_running + min(n_done, _WORKER_DONE_MAX)   # 运行中全展示 + 最近完成
        if n_done > _WORKER_DONE_MAX:
            n += 1                                      # 折叠行
        return Dimension(min=1, preferred=n, max=n)

    def _line_prefix(lineno, wrap_count):
        if lineno == 0 and wrap_count == 0:
            color = 'bold fg:ansiyellow' if mode_ref[0] else 'bold fg:ansiwhite'
            return [(color, '> ')]
        return [('', '  ')]

    _MENU_MAX = 8  # CompletionsMenu max_height
    _distance = _cursor_distance_to_bottom()
    _spacer_height = max(0, _MENU_MAX - _distance) if _distance < _MENU_MAX else 0

    _body_windows = [
        # worker 进度面板：置于整个输入组件最顶部（上边框之外），
        # 与输入区之间由现有上边框线自然分隔，独立区块感更强
        Window(FormattedTextControl(_workers_panel), height=_panel_height,
               dont_extend_height=True),
        Window(FormattedTextControl(_top), height=1, dont_extend_height=True),
        Window(
            BufferControl(buffer=buf),
            get_line_prefix=_line_prefix,
            height=Dimension(min=1),
            dont_extend_height=True,
            wrap_lines=True,
        ),
        Window(FormattedTextControl(_bot), dont_extend_height=True),
    ]
    if _spacer_height > 0:
        _body_windows.append(
            Window(FormattedTextControl(lambda: [("", "")]),
                   height=_spacer_height, dont_extend_height=True),
        )

    body = HSplit(_body_windows)

    root = FloatContainer(
        content=body,
        floats=[
            Float(
                xcursor=True, ycursor=True,
                content=CompletionsMenu(max_height=8, scroll_offset=1),
            ),
        ],
    )

    kb = KeyBindings()

    @kb.add('enter')
    def _(event):
        buf.validate_and_handle()

    @kb.add('c-c')
    def _(event):
        event.app.exit(exception=KeyboardInterrupt())

    @kb.add('s-tab')
    def _(event):
        """Shift+Tab 切换 plan mode：调用外部回调（负责 engine 切换），再刷新 UI。"""
        if on_mode_toggle:
            on_mode_toggle()
        else:
            mode_ref[0] = not mode_ref[0]
        event.app.invalidate()

    @kb.add('c-d')
    def _(event):
        if not buf.text:
            event.app.exit(exception=EOFError())

    @kb.add('backspace')
    def _(event):
        """退格键：若光标前紧邻粘贴占位符，则一次性删除整个占位符；
        否则执行默认单字符删除。"""
        cursor_pos = buf.cursor_position
        text_before = buf.text[:cursor_pos]
        # 检查光标前是否以某个占位符结尾
        for placeholder, _actual in _paste_registry:
            if text_before.endswith(placeholder):
                # 一次性删除整个占位符
                new_text = buf.text[:cursor_pos - len(placeholder)] + buf.text[cursor_pos:]
                buf.text = new_text
                buf.cursor_position = cursor_pos - len(placeholder)
                # 从注册表移除该占位符
                _paste_registry[:] = [(p, a) for p, a in _paste_registry if p != placeholder]
                return
        # 非占位符：执行默认单字符删除
        buf.delete_before_cursor(1)

    # refresh_interval：面板心跳。worker 在后台线程更新状态，bordered_prompt 阻塞在
    # app.run() 期间没有其他事件源，必须靠定时 invalidate 才能把 get_running_status()
    # 的最新快照重绘出来。1s 间隔：worker 活动变化粒度是秒级，够跟手且开销可忽略。
    # 用 prompt_toolkit 原生 refresh_interval（内部 async 任务随 run_async 自动启停），
    # 普通模式（cb=None）保持 None = 事件驱动，零空转、零影响。
    app = PTApp(
        layout=Layout(root),
        key_bindings=kb,
        full_screen=False,
        refresh_interval=1.0 if worker_status_cb is not None else None,
    )
    app.layout.focus(buf)
    try:
        return app.run()
    finally:
        # 提交后清理 spacer 区域：上移光标 → 清除到屏幕底，后续输出紧贴下边框
        if _spacer_height > 0:
            sys.stdout.write(f'\033[{_spacer_height}A')
            sys.stdout.write('\033[J')
            sys.stdout.flush()


def pick_session(sessions: list) -> "object | None":
    """交互式会话选择器：上下方向键移动，回车确认，q/Esc 取消。

    参数 sessions: list[SessionMeta]，按 updated_at 降序排列（最新在前）。
    返回选中的 SessionMeta，取消返回 None。
    """
    if not sessions:
        return None

    state = {"idx": 0}  # 用 dict 让内层函数可修改

    def _render_lines():
        """生成列表每行的 FormattedText 片段。"""
        lines = []
        for i, meta in enumerate(sessions):
            from core.session import format_local_time
            updated = format_local_time(meta.updated_at, "%Y-%m-%d %H:%M")
            title = (meta.title or "Untitled")[:50]
            label = f"  {updated}  {title}"
            if i == state["idx"]:
                lines.append(("bold fg:ansigreen", f"> {label}\n"))
            else:
                lines.append(("", f"  {label}\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    def _(event):
        state["idx"] = max(0, state["idx"] - 1)
        event.app.invalidate()

    @kb.add("down")
    def _(event):
        state["idx"] = min(len(sessions) - 1, state["idx"] + 1)
        event.app.invalidate()

    @kb.add("enter")
    def _(event):
        event.app.exit(result=sessions[state["idx"]])

    @kb.add("q")
    @kb.add("c-c")
    @kb.add("escape")
    def _(event):
        event.app.exit(result=None)

    header = Window(
        FormattedTextControl(lambda: [("bold", "Select a session  (↑↓ move · Enter confirm · q cancel)\n")]),
        height=1, dont_extend_height=True,
    )
    body = Window(
        FormattedTextControl(_render_lines),
        dont_extend_height=False,
    )
    layout = Layout(HSplit([header, body]))

    app = PTApp(layout=layout, key_bindings=kb, full_screen=False, refresh_interval=None)
    return app.run()
