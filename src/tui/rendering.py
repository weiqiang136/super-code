"""Streaming markdown renderer and spinner manager for the TUI."""
from __future__ import annotations

import re

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown as RichMarkdown
from rich.spinner import Spinner
from rich.text import Text

_BLOCK_BOUNDARY_RE = re.compile(r"\n(?=\n|\#{1,6} |```|---|\* |- |\d+\. )")

# 各 UI 场景专属 spinner 风格（Rich 内置）与配色，让不同状态一眼可辨、告别全场景单一灰白 dots。
# 元组格式: (动画风格名, Rich 颜色名)，start() 里解包。
SPINNER_THINKING = ("dots8", "cyan")           # 思考中：点阵旋转（青，冷静）
SPINNER_COMPACT = ("arc", "yellow")            # 上下文压缩：圆弧转动（黄，整理收拢）
SPINNER_PREPARING = ("line", "bright_blue")    # 准备工具调用：短线脉冲（蓝，待命）
SPINNER_WORKING = ("bouncingBar", "green")     # 工具执行中：弹跳条（绿，干活推进）
SPINNER_MEMORY = ("star", "magenta")           # 记忆检索：星光闪烁（品红，联想检索）


class StreamingMarkdown:
    def __init__(self, console: Console):
        self._console = console
        self._buf = ""
        self._stable_len = 0
        self._live: Live | None = None

    def feed(self, chunk: str) -> None:
        self._buf += chunk
        self._render()

    def _render(self) -> None:  # 智能区分并分别渲染稳定文本和不完整文本：将已完成的 Markdown 块直接打印到终端，而将正在接收中的不完整部分用 Live 组件动态刷新以实现打字机效果。
        text = self._buf
        boundary = self._stable_len
        for m in _BLOCK_BOUNDARY_RE.finditer(text, self._stable_len):
            boundary = m.start()
        if boundary > self._stable_len:
            if self._live is not None:
                self._live.stop()
                self._live = None
            stable_text = text[self._stable_len:boundary]
            self._console.print(RichMarkdown(stable_text), end="")
            self._stable_len = boundary
        unstable = text[self._stable_len:]
        if unstable:
            if self._live is None:
                self._live = Live(RichMarkdown(unstable), console=self._console,
                                  refresh_per_second=8, transient=True)
                self._live.start()
            else:
                self._live.update(RichMarkdown(unstable))

    def flush(self) -> None:    # 强制完成所有剩余文本的渲染并清空缓冲区，确保流式传输结束时没有内容遗漏。
        if self._live is not None:
            self._live.stop()
            self._live = None
        remaining = self._buf[self._stable_len:]
        if remaining:
            self._console.print(RichMarkdown(remaining), end="")
        self._buf = ""
        self._stable_len = 0


class SpinnerManager:
    def __init__(self, console: Console):
        self._console = console
        self._live: Live | None = None
        self._spinner: Spinner | None = None
        self._spinner_name: str | tuple | None = None   # 当前 (风格名, 颜色) 配置（Rich Spinner 不保存 name，需自己记录）

    def start(self, text: str = "Thinking…", spinner: str | tuple[str, str] = "dots"):
        # 启动带提示文本的加载动画，每秒刷新12次且结束后自动消失。
        # spinner 参数：Rich 动画风格名，或 (风格名, 颜色) 元组（见上方 SPINNER_* 常量）；
        # 只传风格名时默认青色。帧与文字同色，去掉 dim 避免灰白。
        name, color = spinner if isinstance(spinner, tuple) else (spinner, "cyan")
        # 幂等：Live 已在运行时只原地换文本，不销毁重建。思考模型按 token 高频
        # 产生 thinking 事件，若每次都 stop+重建（清屏+重启刷新线程），spinner 会
        # 忽隐忽现、出现"没反应"的空窗。复用同一个 Spinner 实例避免动画相位重置。
        # 但请求的风格/颜色与当前不同（场景切换）时必须重建，否则换不了动画。
        if self._live is not None and self._spinner is not None:
            if self._spinner_name != spinner:
                self._live.stop()
                self._live = None
                self._spinner = None
            else:
                self._spinner.text = Text(text, style=color)
                self._live.update(self._spinner)
                return
        self._spinner = Spinner(name, text=Text(text, style=color), style=color)
        self._spinner_name = spinner
        self._live = Live(
            self._spinner,
            console=self._console, refresh_per_second=12, transient=True,
        )
        self._live.start()

    def stop(self):                     # 停止并清除当前正在显示的加载动画。
        if self._live is not None:
            self._live.stop()
            self._live = None


def tool_preview(tool_name: str, tool_input: dict) -> str:    # 根据工具类型生成简洁的工具调用预览文本，用于在界面上显示即将执行的操作摘要。
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        return cmd[:80] + ("…" if len(cmd) > 80 else "")
    if tool_name in ("Read", "Edit", "Write"):
        fp = tool_input.get("file_path", "")
        from pathlib import Path
        return Path(fp).name if fp else ""
    if tool_name in ("Glob", "Grep"):
        pat = tool_input.get("pattern", "")
        p = tool_input.get("path", "")
        return f"{pat} in {p}" if p else pat
    return ""


def collapsed_tool_summary(tool_names: list[str]) -> str:  # 是将多个工具调用按类型统计并生成简洁的汇总描述文本，用" · "连接显示。
    from collections import Counter
    counts = Counter(tool_names)
    _LABELS = {
        "Read": ("Reading {n} files", "Reading file"),
        "Glob": ("Searching {n} patterns", "Searching"),
        "Grep": ("Searching {n} patterns", "Searching"),
        "Bash": ("Running {n} commands", "Running command"),
        "Edit": ("Editing {n} files", "Editing file"),
        "Write": ("Writing {n} files", "Writing file"),
    }
    parts = []
    for name, n in counts.items():
        plural, singular = _LABELS.get(name, (f"{name} ×{{n}}", name))
        parts.append(plural.format(n=n) if n > 1 else singular)
    return " · ".join(parts) + "…"