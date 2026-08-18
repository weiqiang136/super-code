"""带边框的输入框 + 斜杠命令补全。"""
from __future__ import annotations

import os
import sys

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

        # 按显示列宽截断 + 省略号（… 宽度由 Text.truncate 自动计入），存储的完整标题不受影响
        _title_text = Text(title)
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

    def _line_prefix(lineno, wrap_count):
        if lineno == 0 and wrap_count == 0:
            color = 'bold fg:ansiyellow' if mode_ref[0] else 'bold fg:ansiwhite'
            return [(color, '> ')]
        return [('', '  ')]

    _MENU_MAX = 8  # CompletionsMenu max_height
    _distance = _cursor_distance_to_bottom()
    _spacer_height = max(0, _MENU_MAX - _distance) if _distance < _MENU_MAX else 0

    _body_windows = [
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

    app = PTApp(
        layout=Layout(root),
        key_bindings=kb,
        full_screen=False,
        refresh_interval=None,
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
