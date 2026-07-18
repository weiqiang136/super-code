"""扫描记忆目录，输出 header 清单（filename + mtime + description + type）。

本模块只负责扫描与格式化，不做选择 / 写入 / 注入；为下游 Step 5（后台抽取）和
Step 6（相关性精选）提供共享原语。

为什么只读前 30 行：
    单个记忆文件可能很大；我们只需要顶部的 YAML frontmatter，按行读到 30 行即可截断，
    避免把全量内容拉进内存。

为什么按 mtime 倒序 + 上限 200：
    记忆目录会随时间累积，老旧记忆相关性低。倒序 + 截断让 Step 6 的
    side-query manifest 体积可控。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# 复用 skills 模块已有的极简 YAML frontmatter 解析器（无 PyYAML 依赖）。
# 该函数虽以下划线开头属"模块私有"，但项目内多处复用属于既定约定；
# 此处显式 import 保证与 SKILL.md 解析行为完全一致（包括续行处理）。
from features.skills import _parse_frontmatter
from features.memory_types import MemoryType, parse_memory_type

# 单次扫描返回的最大文件数。超出后按 mtime 倒序截断。
MAX_MEMORY_FILES = 200

# 只读取文件开头多少行用于解析 frontmatter。YAML frontmatter 紧贴文件首部，
# 30 行足以覆盖最复杂的多字段续行场景，同时显著降低 IO。
FRONTMATTER_MAX_LINES = 30

# MEMORY.md 是"索引"而非"记忆"，由 features.memory.load_memory_index 单独处理，
# 这里必须排除以免被 Step 6 的 selector 误当作可选条目。
ENTRYPOINT_NAME = "MEMORY.md"

# KAIROS 模式（features.memory 已实现）下，logs/YYYY/MM/*.md 是 append-only 日志，
# 没有 frontmatter，扫描进来全是噪音；显式跳过该子树。
_LOGS_DIRNAME = "logs"


@dataclass(frozen=True)
class MemoryHeader:
    """单条记忆的 header 摘要，不含正文。

    - filename: 相对 memory_dir 的 POSIX 路径（跨平台稳定，便于在 prompt 中展示）
    - file_path: 绝对路径，供下游真正打开文件
    - mtime_ms: 毫秒级时间戳（与 JS Date.now() 对齐，方便后续 Step 4 memory_age 复用）
    - description / type: 从 frontmatter 提取，缺失即 None（不报错）
    """
    filename: str
    file_path: Path
    mtime_ms: float
    description: str | None
    type: MemoryType | None


def _read_frontmatter_only(path: Path, max_lines: int = FRONTMATTER_MAX_LINES) -> str:
    """只读前 max_lines 行。逐行迭代 + 早停，避免把整个文件加载到内存。"""
    chunks: list[str] = []
    # errors="replace" 与 features.memory.load_memory_index 行为保持一致：
    # 编码异常不应让整个扫描流程中断。
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            chunks.append(line)
    return "".join(chunks)


def scan_memory_files(memory_dir: Path) -> list[MemoryHeader]:
    """递归扫描 memory_dir 下所有 .md 文件并返回 header 列表。

    排除规则：
        1. 顶层及子目录中的 MEMORY.md（索引文件）
        2. logs/ 子树（KAIROS append-only 日志）
        3. 单文件读取异常（OSError）—— 静默跳过，不影响其它文件

    返回：按 mtime 倒序、最多 MAX_MEMORY_FILES 条。
        目录不存在 → 返回空列表（首次启动场景）。
    """
    if not memory_dir.exists():
        return []

    headers: list[MemoryHeader] = []
    # rglob 在不存在的目录上会抛 FileNotFoundError；已在上面 exists() 守卫过
    for path in memory_dir.rglob("*.md"):
        # 排除索引文件（无论在哪一层）
        if path.name == ENTRYPOINT_NAME:
            continue

        # 计算相对路径；理论上 rglob 出来的 path 必然在 memory_dir 之下，
        # try/except 是对符号链接 / 边界情况的兜底
        try:
            relative = path.relative_to(memory_dir)
        except ValueError:
            continue

        # 排除 logs/ 子树（取顶层目录名判断，深度无关）
        if relative.parts and relative.parts[0] == _LOGS_DIRNAME:
            continue

        try:
            stat = path.stat()
            head_text = _read_frontmatter_only(path)
        except OSError:
            # 文件被并发删除 / 权限不足 / 损坏，跳过即可
            continue

        # _parse_frontmatter 在无 frontmatter 时返回 ({}, full_text)，不会抛异常
        meta, _body = _parse_frontmatter(head_text)

        # description 在 frontmatter 中可能是 str / bool / list（_parse_frontmatter 会做类型推断）；
        # 这里只接受 str 且非空，其它情况降级为 None
        desc_raw = meta.get("description")
        description = desc_raw.strip() if isinstance(desc_raw, str) and desc_raw.strip() else None

        headers.append(
            MemoryHeader(
                filename=relative.as_posix(),
                file_path=path,
                # st_mtime 是 float 秒，× 1000 得毫秒；保持 float 不强转 int，
                # 让 Step 4 memory_age 在做天数差时不会因精度丢失
                mtime_ms=stat.st_mtime * 1000.0,
                description=description,
                type=parse_memory_type(meta.get("type")),
            )
        )

    headers.sort(key=lambda h: h.mtime_ms, reverse=True)
    return headers[:MAX_MEMORY_FILES]


def format_memory_manifest(memories: list[MemoryHeader]) -> str:
    """格式化为单行清单文本，给 Step 5/6 的 LLM prompt 用。

    输出样例（每行一条，无尾空行）::

        - [feedback] feedback_pytest.md (2026-06-08T12:34:56+00:00): use pytest, not unittest
        - notes.md (2026-06-01T09:00:00+00:00)

    缺 type → 不带 `[xxx]` 前缀；缺 description → 不带末尾冒号。
    """
    lines: list[str] = []
    for m in memories:
        tag = f"[{m.type}] " if m.type else ""
        # UTC ISO 时间戳，秒级精度，毫秒级精度对模型来说是噪音。
        ts = datetime.fromtimestamp(m.mtime_ms / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")
        if m.description:
            lines.append(f"- {tag}{m.filename} ({ts}): {m.description}")
        else:
            lines.append(f"- {tag}{m.filename} ({ts})")
    return "\n".join(lines)
