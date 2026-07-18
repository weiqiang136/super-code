"""Plan mode — explore-before-implement workflow."""
from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.engine import Engine
    from core.tool import Tool
    from core.permissions import PermissionChecker

# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------

_ADJECTIVES = [
    "amber", "azure", "bold", "bright", "calm", "clear", "cool", "crisp",
    "dark", "deep", "eager", "fair", "fast", "fierce", "gentle", "golden",
    "lucky", "merry", "noble", "pale", "proud", "quick", "quiet", "sharp",
    "silent", "sleek", "soft", "steady", "swift", "warm", "wild", "wise",
]

_NOUNS = [
    "arrow", "brook", "castle", "cloud", "comet", "coral", "crane", "dawn",
    "delta", "dove", "dream", "eagle", "ember", "falcon", "fern", "flame",
    "forge", "frost", "harbor", "hawk", "hill", "island", "lake", "leaf",
    "maple", "moon", "ocean", "peak", "pine", "river", "shore", "spark",
    "stone", "storm", "summit", "trail", "valley", "wave", "willow",
]


def _generate_slug() -> str:
    """生成 plan 文件名（形如 lucky-conjuring-island）。"""
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}-{random.choice(_NOUNS)}"


def _get_plans_dir() -> Path:
    """返回 plan 文件存储目录。"""
    plans_dir = Path.home() / ".config" / "super-code" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    return plans_dir


# ---------------------------------------------------------------------------
# PlanModeManager
# ---------------------------------------------------------------------------

class PlanModeManager:
    """管理 plan mode 的生命周期：进入、退出、文件管理、提示词注入。

    先创建 PlanModeManager，再通过 bind_engine() 绑定 engine，
    避免循环依赖（engine 需要 plan_tools，plan_tools 需要 manager）。
    """

    def __init__(self) -> None:
        self._engine: Engine | None = None                  # 主 engine
        self._permissions: PermissionChecker | None = None  # 权限检查器
        self._active: bool = False                          # 是否处于 plan mode
        self._plan_file: Path | None = None                 # 当前 plan 文件路径
        self._saved_tools: list[Tool] | None = None         # 进入前保存的工具列表
        self._saved_prompt: str | None = None               # 进入前保存的系统提示词

    def bind_engine(self, engine: Engine) -> None:
        """绑定主 engine（构造后调用，避免循环依赖）。"""
        self._engine = engine

    def set_permissions(self, permissions: PermissionChecker) -> None:
        """设置权限检查器。"""
        self._permissions = permissions

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def plan_file_path(self) -> str | None:
        return str(self._plan_file) if self._plan_file else None

    def _sanitize_plan_tool_history(self, normal_tool_names: set[str]) -> None:
        """清理历史消息中 plan 专属工具的 tool_use/tool_result，避免退出后 API 报错。

        OpenAI API 要求：assistant 消息中 tool_calls 引用的工具名，必须出现在本次请求的
        tools 参数中。plan 专属工具（如 EnterPlanMode/ExitPlanMode）退出后已从工具集移除，
        若历史中仍有其 tool_use/tool_result，API 会因找不到对应 schema 而报错或挂起。

        关键约束：此方法可能在工具执行过程中被调用（例如 LLM 调用 ExitPlanModeTool，
        ExitPlanModeTool.execute() 触发 exit()，此时 ExitPlanMode 的 tool_use 已在历史中，
        但其 tool_result 尚未追加）。若此时把该 tool_use 转为 text，engine 随后追加的
        tool_result 就会出现在没有 tool_calls 的 assistant 消息之后，触发 API 400 错误：
        "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"

        因此只清理那些在历史中**已有对应 tool_result** 的 plan 专属工具对，
        对于只有 tool_use 尚无 tool_result 的（正在执行中的调用），保持原样不动。

        处理策略：
        - 先收集历史中所有 tool_result 的 tool_use_id（已完成的调用集合）
        - assistant 消息：只将"已完成"的 plan 专属工具 tool_use 转为 text block
        - user 消息（tool_results）：移除对应的 plan 专属 tool_result 条目；
          若该 user 消息的所有条目都被移除，则整条消息也删除
        """
        messages = self._engine._messages

        # 第零遍：收集历史中所有已存在的 tool_result 的 tool_use_id（即已完成的工具调用）
        # 只有这些 id 对应的 tool_use 才能安全清理，正在执行中（尚无 tool_result）的不能动
        completed_tool_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = block.get("tool_use_id", "")
                    if tid:  # 跳过空 id，避免误匹配
                        completed_tool_ids.add(tid)

        # 第一遍：找出"已完成"的 plan 专属工具 tool_use id，并将其转为 text block
        plan_tool_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new_blocks = []
            for block in content:
                tid = block.get("id", "") if isinstance(block, dict) else ""
                is_plan_tool = (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") not in normal_tool_names
                    and tid in completed_tool_ids  # 只处理已完成的调用
                )
                if is_plan_tool:
                    plan_tool_ids.add(tid)
                    # 转为 text block，保留调用记录供 LLM 理解上下文
                    new_blocks.append({
                        "type": "text",
                        "text": f"[Plan tool used: {block.get('name')}]",
                    })
                else:
                    new_blocks.append(block)
            msg["content"] = new_blocks

        if not plan_tool_ids:
            return

        # 第二遍：清理 user 消息中对应的 tool_result 条目
        to_delete = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            # 只处理全部为 tool_result 的 user 消息（即工具结果消息）
            if not all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                continue
            filtered = [b for b in content if b.get("tool_use_id") not in plan_tool_ids]
            if not filtered:
                # 所有条目都是 plan 专属工具的结果，整条消息可删除
                to_delete.append(i)
            else:
                msg["content"] = filtered

        # 倒序删除，避免索引偏移
        for i in reversed(to_delete):
            del messages[i]

    def get_plan_content(self) -> str | None:
        """读取 plan 文件内容。"""
        if self._plan_file and self._plan_file.exists():
            try:
                return self._plan_file.read_text(encoding="utf-8")
            except OSError:
                return None
        return None

    # -- enter / exit -------------------------------------------------------

    def enter(self) -> str:
        """进入 plan mode：创建 plan 文件，切换为只读工具集，注入提示词。"""
        assert self._engine is not None, "PlanModeManager not bound to engine"

        if self._active:
            return f"Already in plan mode. Plan file: {self._plan_file}"

        # 决定本次 plan 模式使用的文件路径：
        #
        # 复用策略（修复"plan 文件换代"bug）：
        #   若 self._plan_file 已存在（即同一进程中之前进入过 plan 模式，
        #   exit() 不会清空此字段），且该文件仍在磁盘上，则复用同一个文件。
        #   这样用户在 plan → normal → plan 来回切换时，LLM 看到的始终是
        #   同一个 plan 文件，避免出现：
        #     - 历史中 LLM 提及旧文件名 → 新进入后 LLM 试图编辑旧文件
        #     - 权限层只允许编辑新文件 → 编辑被拒 → turn cancelled
        #
        # 新建策略：
        #   首次进入（self._plan_file is None）或上次的文件已被外部删除时，
        #   生成不重复的新 slug 文件名。
        #
        # 跨进程不会复用：self._plan_file 是实例属性，新 super-code 进程
        # 重新构造 PlanModeManager 时 self._plan_file 从 None 开始——
        # 这是合理的会话隔离边界。
        if self._plan_file is not None and self._plan_file.exists():
            path = self._plan_file
        else:
            plans_dir = _get_plans_dir()
            for _ in range(10):
                slug = _generate_slug()
                path = plans_dir / f"{slug}.md"
                if not path.exists():
                    break
        self._plan_file = path

        # 保存当前 engine 状态
        self._saved_tools = list(self._engine._tools.values())
        self._saved_prompt = self._engine.system_prompt

        # 构建 plan mode 工具集（只读 + plan 工具）
        from tools.plan_tools import EnterPlanModeTool, ExitPlanModeTool
        from tools.ask_user import AskUserQuestionTool
        from tools.file_read import FileReadTool
        from tools.glob_tool import GlobTool
        from tools.grep_tool import GrepTool
        from tools.file_edit import FileEditTool
        from tools.file_write import FileWriteTool

        plan_tools: list[Tool] = [
            FileReadTool(), GlobTool(), GrepTool(),
            FileEditTool(), FileWriteTool(),   # 权限层限制只能编辑 plan 文件
            AskUserQuestionTool(),
            EnterPlanModeTool(self),
            ExitPlanModeTool(self),
        ]
        self._engine.set_tools(plan_tools)

        # 注入 plan mode 系统提示词段落
        from core.context import get_plan_mode_section
        self._engine.system_prompt = self._saved_prompt + "\n\n" + get_plan_mode_section(str(self._plan_file))

        self._active = True

        # 切换权限模式
        if self._permissions is not None:
            self._permissions.enter_plan_mode()

        # 注入通知消息，让 LLM 明确感知已进入 plan mode。
        # 仅修改 system_prompt 不够，历史消息会压制 system prompt 的变化。
        self._engine._messages.append({
            "role": "user",
            "content": "[System: Plan mode is now active. You must NOT edit files or run commands. Only read files and write to the plan file.]",
        })

        return f"Entered plan mode. Plan file: {self._plan_file}"

    def exit(self) -> tuple[str, str | None]:
        """退出 plan mode：恢复工具集和系统提示词，返回 (message, plan_content)。"""
        assert self._engine is not None, "PlanModeManager not bound to engine"

        if not self._active:
            return ("Not in plan mode.", None)

        plan_content = self.get_plan_content()

        # 先恢复权限，再恢复工具
        if self._permissions is not None:
            self._permissions.exit_plan_mode()

        if self._saved_tools is not None:
            self._engine.set_tools(self._saved_tools)
        if self._saved_prompt is not None:
            self._engine.system_prompt = self._saved_prompt

        self._active = False

        # 清理历史消息中 plan 专属工具的 tool_use/tool_result，有两条路径：
        #
        # 路径A：Shift+Tab 手动切换（两轮对话之间，engine 没有正在进行的 submit()）
        #   此时所有工具调用都已完成，tool_results 已全部 append 到 _messages，
        #   可以立即清理。**这是当前唯一会被实际触发的路径**：
        #   ExitPlanModeTool 已重构为"信号工具"（plan_tools.py），不再调用 exit()；
        #   而 Shift+Tab 回调仅在 bordered_prompt 等待输入时生效（tui/app.py），
        #   此时 engine 不可能在 submit() 中。
        #
        # 路径B（防御性保留）：理论上 exit() 在 engine.submit() 运行期间被调用的场景
        #   engine._turn_start_len is not None 表示 submit() 正在运行，
        #   当前轮的 tool_results 尚未 append 到 _messages（engine 在所有工具执行完后
        #   才统一 append）。若此时立即清理，正在执行中的 tool_use 因找不到对应
        #   tool_result 而会被跳过，进而清理失效，下一轮 API 调用报 400。
        #   解决方案：向 engine 注册一次性回调，engine append tool_results 后触发清理。
        #   当前实现中没有真实代码路径走到这里（见上面对路径A的说明），但为防御未来
        #   引入新的 exit() 触发点（如 mid-stream 信号、其它工具触发等）而保留。
        normal_tool_names = {t.name for t in (self._saved_tools or [])}
        if self._engine._turn_start_len is None:
            # 路径A：立即清理
            self._sanitize_plan_tool_history(normal_tool_names)
        else:
            # 路径B：延迟清理，等 engine append tool_results 后触发
            self._engine._post_tool_hooks.append(
                lambda: self._sanitize_plan_tool_history(normal_tool_names)
            )

        self._saved_tools = None
        self._saved_prompt = None

        # 注入通知消息，让 LLM 明确感知模式已切换。
        # 不能截断历史，因为 plan 阶段的分析结论需要保留供后续实现使用。
        self._engine._messages.append({
            "role": "user",
            "content": "[System: Plan mode has ended. You are now in normal mode with full tool access. You can implement the plan discussed above.]",
        })

        plan_path = str(self._plan_file) if self._plan_file else "unknown"

        if plan_content:
            msg = (
                f"Exited plan mode. Plan saved to: {plan_path}\n\n"
                f"## Plan Content:\n{plan_content}\n\n"
                "You can now review the plan, request modifications, or proceed with implementation."
            )
        else:
            msg = "Exited plan mode. No plan file was written."

        return (msg, plan_content)
