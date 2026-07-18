from __future__ import annotations

from core.tool import Tool, ToolResult


class AskUserQuestionTool(Tool):
    @property
    def name(self) -> str:
        return "AskUserQuestion"

    @property
    def description(self) -> str:
        return (
            "Ask the user a question with predefined options. Use this to gather "
            "preferences, clarify ambiguous instructions, or get decisions on "
            "implementation choices. Each question has 2-4 options plus an automatic "
            "'Other' option for free-form input."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["label", "description"],
                                },
                                "minItems": 2,
                                "maxItems": 4,
                            },
                            "multiSelect": {"type": "boolean", "default": False},
                        },
                        "required": ["question", "options"],
                    },
                    "minItems": 1,
                    "maxItems": 4,
                }
            },
            "required": ["questions"],
        }

    def is_read_only(self) -> bool:
        return True

    def execute(self, **kwargs) -> ToolResult:
        questions = kwargs.get("questions", [])
        if not questions:
            return ToolResult(content="No questions provided.", is_error=True)

        # 用 prompt_toolkit 而不是裸 input()：主 TUI 用 prompt_toolkit Application
        # (bordered_prompt) 之后会调整 terminal 状态，加上 Rich Live spinner 的后台
        # 重绘竞争，裸 input() 会出现按键被吞、必须按 Enter 才解锁的诡异现象。
        # prompt_toolkit 的 prompt() 启动时会再次接管 terminal，与既有 TUI 链路兼容。
        # 失败时（如非交互式环境）回退到 input() 保持原行为。
        from prompt_toolkit import prompt as pt_prompt

        def _ask(label: str) -> str:
            try:
                return pt_prompt(label).strip()
            except Exception:
                return input(label).strip()

        answers: list[str] = []
        for q in questions:
            question_text = q.get("question", "")
            options = q.get("options", [])
            labels = [o["label"] for o in options]

            from rich.console import Console
            console = Console()
            console.print(f"\n[bold]{question_text}[/bold]")
            for i, o in enumerate(options, 1):
                desc = o.get("description", "")
                console.print(f"  {i}) {o['label']}" + (f" — {desc}" if desc else ""))
            console.print(f"  {len(options)+1}) Other")

            while True:
                try:
                    raw = _ask("  Choice: ")
                except (EOFError, KeyboardInterrupt):
                    return ToolResult(content="User cancelled the question.", is_error=True)
                if raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(options):
                        answers.append(f"{question_text} => {labels[idx]}")
                        break
                    if idx == len(options):
                        try:
                            other = _ask("  Enter your answer: ")
                        except (EOFError, KeyboardInterrupt):
                            return ToolResult(content="User cancelled the question.", is_error=True)
                        answers.append(f"{question_text} => {other}")
                        break
                elif raw:
                    answers.append(f"{question_text} => {raw}")
                    break

        return ToolResult(content="User answered:\n" + "\n".join(answers))
