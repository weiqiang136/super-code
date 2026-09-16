"""全局兜底 Console：Rich markup 解析失败时退化成纯文本重打，不让整轮对话崩掉。

为什么需要：模型/外部可控文本（工具输入、异常消息、shell 输出、图片路径）经常被
直接拼进 `console.print()`。`Console.print` 默认 `markup=True`，会把这类文本当 Rich
标签解析，而 Rich 的标签正则 `((\\*)\[([a-z#/@][^[]*?)])` 只约束 `[` 后的**第一个**
字符——所以 `[/\\]`、`[/...]`、`[/b]` 一律被当成**闭合标签**；找不到对应 open tag 时
抛 `MarkupError: closing tag '...' at position N doesn't match any open tag`。
异常会从工具/权限层一路冒到 turn 顶层 → 整轮对话失败（"Query failed: ..."），
且会话可能被自己讨论 `[/...]` 的输出反复毒死（权限提示弹一次炸一次）。

逐个补 `rich.markup.escape()` 容易漏（新加的 print 又会引入新窟窿），所以在这里做
进程级兜底：`print` 失败时用 `markup=False` 重打一遍——只丢该行颜色，内容与 turn
都保住。交互路径（权限提示 / AskUserQuestion）另在源头 escape，避免 Panel 类
renderable 走到兜底时留下半截输出。
"""
from __future__ import annotations

from rich.console import Console
from rich.errors import MarkupError


class SafeConsole(Console):
    """`Console` 的容错版本：任何 renderable 触发 MarkupError 都降级为纯文本重打。"""

    def print(self, *objects, **kwargs) -> None:
        try:
            super().print(*objects, **kwargs)
        except MarkupError:
            # str renderable 在真正输出前就会在 _collect_renderables 里抛错（无半截输出）；
            # Panel 内含 str 的情况可能已写出部分边框，重打一次可读性仍优于整轮崩掉。
            # 只重试一次：markup=False 后仍失败说明不是 markup 问题，按原样抛出。
            kwargs["markup"] = False
            super().print(*objects, **kwargs)
