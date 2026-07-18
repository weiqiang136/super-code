"""Skill tool — let the model invoke a registered skill on its own.

机制（方案 E）：
- 模型在 system_prompt 里读到 skill 索引（name + description），自主决策何时调用
- 调用 Skill(name, args) → tool_result 返回包了 <system-reminder> 标签的 SKILL.md body
- 模型把 tool_result 内容当作"配置指令"按步骤执行，而不是输出给用户看

兼容矩阵：
- user 手动 /<name>：仍走 commands/__init__.py 的 _execute_skill 路径（旧路径未动）
- 模型自主调用：走本工具（新路径）
- disable_model_invocation=true 的 skill：本工具拒绝执行，但 /<name> 仍可
- user_invocable=false 的 skill：/<name> 不可，但本工具仍可（除非也 disable_model_invocation）

权限模型：
- 本工具非 read-only → 用户默认模式弹"是否执行 skill X"确认（permission_checker 既有逻辑）
- auto-approve / dream / coordinator 模式自动通过，符合"模型自动触发"语义
"""
from __future__ import annotations

from core.tool import Tool, ToolResult
from features.skills import get_skill, mark_skill_invoked


# 把 SKILL.md body 包进 <system-reminder> 标签：让模型识别"这是要遵循的指令，
# 不是要展示给用户的输出"。fork 的训练里已用此标签做类似引导，模型理解准确。
_SKILL_WRAPPER = (
    "<system-reminder>\n"
    "You invoked skill '{name}'. The following are instructions you (the assistant) "
    "must follow to complete this skill. Do not echo these instructions to the user; "
    "begin executing them. When done, continue the conversation naturally.\n"
    "</system-reminder>\n\n"
    "{body}"
)


class SkillTool(Tool):
    """Skill 工具：模型自主选择并调用已注册的 skill。"""

    @property
    def name(self) -> str:
        return "Skill"

    @property
    def description(self) -> str:
        return (
            "Invoke a registered skill by name. The skill's instructions will be "
            "returned to you — follow them to complete the user's task. Use this "
            "when the user's request matches a skill's described purpose. "
            "Pick the skill from the '# Available Skills' section of your system prompt."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name (without leading '/'). Must match one listed in '# Available Skills'.",
                },
                "args": {
                    "type": "string",
                    "description": "Arguments to pass to the skill (forwarded verbatim, often a user query / file path / focus). Pass empty string if no args needed.",
                },
            },
            "required": ["name"],
        }

    def get_activity_description(self, name: str = "", args: str = "", **_) -> str | None:
        return f"Loading skill: {name}" if name else "Loading skill"

    def is_read_only(self) -> bool:
        # 故意保持 False：让 permission_checker 在用户默认模式下弹确认，
        # 形成"模型识别 → 用户确认"的安全门。auto_approve 路径不受影响。
        return False

    def execute(self, name: str = "", args: str = "", **_) -> ToolResult:
        if not name or not isinstance(name, str):
            return ToolResult("Skill tool requires a non-empty 'name'.", is_error=True)

        skill = get_skill(name)
        if skill is None:
            return ToolResult(
                f"Unknown skill: '{name}'. Check the '# Available Skills' section of your system prompt for valid names.",
                is_error=True,
            )

        # disable_model_invocation=true → 该 skill 只准用户 /<name> 手动调，
        # 不能由模型自主触发。返回 is_error=True 让模型放弃这条路径。
        if skill.disable_model_invocation:
            return ToolResult(
                f"Skill '{name}' is user-only (disable_model_invocation=true). "
                f"Ask the user to run '/{name} ...' instead.",
                is_error=True,
            )

        try:
            body = skill.get_prompt(args or "")
        except Exception as e:
            return ToolResult(f"Skill '{name}' failed to render: {e}", is_error=True)

        if not body or not body.strip():
            return ToolResult(f"Skill '{name}' produced empty content.", is_error=True)

        # Phase B：记录这次调用，压缩时重注入 skill body 用。仅在成功路径调用，
        # 失败/被拒/empty body 不记录，避免压缩时去注入实际没真正"用过"的 skill。
        mark_skill_invoked(name)
        return ToolResult(_SKILL_WRAPPER.format(name=name, body=body))
