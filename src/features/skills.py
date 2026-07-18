"""Skill system — load, register, and execute SKILL.md-based skills."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Skill definition
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    description: str = ""
    when_to_use: str = ""
    user_invocable: bool = True
    context: str = "inline"         # "inline" 注入当前对话 / "fork" 独立子会话
    argument_hint: str = ""
    source: str = "project"         # "bundled" / "project" / "user"
    skill_root: str | None = None   # 用于 $SKILL_DIR 变量替换的基础目录

    # ── 扩展字段 ───────────────────────────────────────────────────────────
    # 解析后仅存储，目标 1（字段对齐）阶段不强制行为；目标 2（自动触发）启用后才生效。
    # 这样外部 SKILL.md 可以直接复制进来不报错、字段不丢失。
    allowed_tools: list[str] = field(default_factory=list)  # 限制 skill 可调用工具，空列表 = 不限制
    model: str = ""                                          # 指定 skill 偏好的模型，空 = 沿用主对话
    disable_model_invocation: bool = False                  # True = 仅允许用户 /<name> 手动触发

    _prompt_text: str = ""
    _prompt_fn: Callable[[str], str] | None = None

    def get_prompt(self, args: str = "") -> str:
        """返回最终提示词，替换变量。"""
        if self._prompt_fn is not None:
            return self._prompt_fn(args)
        text = self._prompt_text
        text = text.replace("$ARGUMENTS", args)
        if self.skill_root:
            text = text.replace("${CLAUDE_SKILL_DIR}", self.skill_root)
        if args and self.argument_hint:
            text = text.replace(f"${{{self.argument_hint}}}", args)
        return text


# ---------------------------------------------------------------------------
# YAML frontmatter parser（最小实现，无 PyYAML 依赖）
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

#标识哪些字段被认为是列表类型，可以采用逗号分隔
_LIST_FIELDS = {"allowed_tools"}


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """将 SKILL.md 文本拆分为 (frontmatter_dict, body)。

    支持 YAML 续行：以空格/制表符缩进开头的行视为前一字段的续行，
    用单个空格连接拼到前一字段值上。典型场景：
        description: Use this skill whenever the user wants to do anything with
          PDF files. This includes reading or extracting text/tables ...
    没有续行支持时 description 只保留第一行，模型读不到完整触发条件。
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    body = text[m.end():]

    # 第一遍：识别"key: value"行 + 续行（缩进开头），把多行 value 拼回单行
    pairs: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        # 续行判定：行首是空白 + 已有累积字段 → 拼到前一字段
        if line[:1] in (" ", "\t") and pairs:
            cont = line.strip()
            if cont:
                key, prev = pairs[-1]
                pairs[-1] = (key, (prev + " " + cont).strip() if prev else cont)
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        pairs.append((key.strip().lower().replace("-", "_"), val.strip()))

    # 第二遍：类型推断
    meta: dict[str, Any] = {}
    for key, val in pairs:
        if val.lower() in ("true", "yes"):
            meta[key] = True
        elif val.lower() in ("false", "no"):
            meta[key] = False
        elif key in _LIST_FIELDS and "," in val:
            # 仅已知列表字段才按逗号切分，避免 description/when_to_use 等被误切
            meta[key] = [v.strip() for v in val.split(",") if v.strip()]
        elif (val.startswith('"') and val.endswith('"')) or \
             (val.startswith("'") and val.endswith("'")):
            meta[key] = val[1:-1]
        else:
            meta[key] = val
    return meta, body


def _ensure_str(val: Any, default: str = "") -> str:
    """将任意类型安全转为字符串（列表则 join）。"""
    if val is None:
        return default
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val)


def _skill_from_frontmatter(meta: dict[str, Any], body: str,
                             name: str, source: str,
                             skill_root: str | None = None) -> Skill:
    """根据解析的 frontmatter 和 body 构建 Skill 对象。"""
    # argument-hint 是标准字段名（经 _parse_frontmatter 规范化后变成
    # argument_hint）；早期版本写的是 arguments，两者都兼容，前者优先。
    arg_hint = meta.get("argument_hint")
    if not arg_hint:
        arg_hint = meta.get("arguments")

    # allowed-tools：标准 frontmatter 通常是 "Read, Write, Bash" 这样的逗号串。
    # _parse_frontmatter 对已知列表字段会切成 list，但用户也可能写成单值字符串，
    # 这里统一规整为 list[str]。
    raw_allowed = meta.get("allowed_tools")
    if isinstance(raw_allowed, list):
        allowed_tools = [str(t).strip() for t in raw_allowed if str(t).strip()]
    elif isinstance(raw_allowed, str) and raw_allowed.strip():
        allowed_tools = [t.strip() for t in raw_allowed.split(",") if t.strip()]
    else:
        allowed_tools = []

    return Skill(
        name=_ensure_str(meta.get("name"), name),
        description=_ensure_str(meta.get("description")),
        when_to_use=_ensure_str(meta.get("when_to_use")),
        user_invocable=meta.get("user_invocable", True),
        context=_ensure_str(meta.get("context"), "inline"),
        argument_hint=_ensure_str(arg_hint),
        source=source,
        skill_root=skill_root,
        allowed_tools=allowed_tools,
        model=_ensure_str(meta.get("model")),
        disable_model_invocation=bool(meta.get("disable_model_invocation", False)),
        _prompt_text=body.strip(),
    )


# ---------------------------------------------------------------------------
# Skill registry（全局注册表）
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Skill] = {}

# 本进程内"已被调用过的 skill 名称"集合。压缩流程会读取它做 Phase B 重注入，
# 让模型在压缩之后仍知道之前调用过哪些 skill 的指令内容。
# 进程级状态（不持久化、不跨会话）：/resume 一个老 session 时该集合为空，
# 老对话里的 skill body 仍在 messages 中能进入 history 被总结，无需重注入。
_INVOKED_SKILLS: set[str] = set()


def mark_skill_invoked(name: str) -> None:
    """记录一次 skill 调用。两条调用路径（用户 /<name> 与模型 SkillTool）都会调本函数。"""
    if name:
        _INVOKED_SKILLS.add(name)


def get_invoked_skills() -> list[str]:
    """返回本进程内调用过的 skill 名称（按字母序）。压缩重注入用。"""
    return sorted(_INVOKED_SKILLS)


def clear_invoked_skills() -> None:
    """清空调用记录。测试或 /clear 命令调用。"""
    _INVOKED_SKILLS.clear()


def register_skill(skill: Skill) -> None:
    """将 skill 注册到全局注册表。"""
    _REGISTRY[skill.name] = skill


def get_skill(name: str) -> Skill | None:
    """按名称查找 skill。"""
    return _REGISTRY.get(name)


def list_skills(user_invocable_only: bool = True) -> list[Skill]:
    """返回所有已注册的 skill，可选只返回用户可调用的。"""
    skills = list(_REGISTRY.values())
    if user_invocable_only:
        skills = [s for s in skills if s.user_invocable]
    return sorted(skills, key=lambda s: (s.source != "bundled", s.name))


def clear_skills(source: str | None = None) -> None:
    """清除注册表，可选只清除指定来源的 skill。"""
    if source is None:
        _REGISTRY.clear()
    else:
        for k in [k for k, v in _REGISTRY.items() if v.source == source]:
            del _REGISTRY[k]


# ---------------------------------------------------------------------------
# Skill discovery from disk
# ---------------------------------------------------------------------------

def load_skills_from_dir(skills_dir: Path, source: str = "project") -> list[Skill]:
    """扫描 skills_dir 下的 <name>/SKILL.md 并注册每个 skill。"""
    loaded: list[Skill] = []
    if not skills_dir.is_dir():
        return loaded
    for entry in sorted(skills_dir.iterdir()):
        skill = None
        if entry.is_dir():
            skill_md = entry / "SKILL.md"
            if not skill_md.exists():
                md_files = list(entry.glob("*.md"))
                skill_md = md_files[0] if md_files else None
            if skill_md is None:
                continue
            try:
                text = skill_md.read_text(encoding="utf-8")
            except Exception:
                continue
            meta, body = _parse_frontmatter(text)
            skill = _skill_from_frontmatter(meta, body, name=entry.name,
                                            source=source, skill_root=str(entry))
        elif entry.suffix == ".md" and entry.is_file():
            try:
                text = entry.read_text(encoding="utf-8")
            except Exception:
                continue
            meta, body = _parse_frontmatter(text)
            skill = _skill_from_frontmatter(meta, body, name=entry.stem,
                                            source=source, skill_root=str(entry.parent))
        if skill and skill._prompt_text:
            register_skill(skill)
            loaded.append(skill)
    return loaded


def discover_skills(cwd: str | None = None) -> list[Skill]:
    """从标准位置发现并注册 skill。

    搜索顺序（同名 skill 后扫描者覆盖前者）：
      1. 用户级(HOME)：~/.config/super-code/skills/
      2. 便携级(exe同级)：<exe目录>/skills/
      3. 项目级(当前目录)：{cwd}/.super-code/skills/
    """
    loaded: list[Skill] = []
    home = Path.home()
    loaded.extend(load_skills_from_dir(home / ".config" / "super-code" / "skills", source="user"))
    from core.config import get_portable_dir
    loaded.extend(load_skills_from_dir(get_portable_dir() / "skills", source="portable"))
    if cwd:
        loaded.extend(load_skills_from_dir(Path(cwd) / ".super-code" / "skills", source="project"))
    return loaded


# ---------------------------------------------------------------------------
# System prompt section
# ---------------------------------------------------------------------------

def build_skills_prompt_section() -> str:
    """生成 skill 列表文本，拼接到系统提示词中，让模型知道可用的 skill。

    输出含两段：
      1) 用法说明：告诉模型如何用 Skill 工具自主调用，以及和用户手动 /<name> 的差异
      2) skill 索引：每个 skill 的 name + description(+ when_to_use)，仅作触发判据

    disable_model_invocation=true 的 skill 仍出现在索引里，便于模型识别用户意图
    后建议用户手动 /<name>；SkillTool.execute 会真正拒绝模型对它们的调用。
    """
    skills = list_skills(user_invocable_only=False)
    if not skills:
        return ""
    lines = [
        "# Available Skills",
        "",
        "When the user's request matches one of these skills, prefer invoking it "
        "via the Skill tool: `Skill(name=\"<skill-name>\", args=\"<user args>\")`. "
        "The user can also trigger any skill manually by typing `/<skill-name> args`.",
        "",
    ]
    for s in skills:
        desc = s.description or "(no description)"
        line = f"- {s.name}: {desc}"
        if s.when_to_use:
            line += f" — {s.when_to_use}"
        if s.disable_model_invocation:
            line += "  [user-only: suggest /<name> instead of calling Skill tool]"
        lines.append(line)
    return "\n".join(lines)
