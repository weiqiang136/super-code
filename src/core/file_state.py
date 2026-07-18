"""Snippet 系统核心数据模型 —— 会话级文件状态 & 编辑凭证追踪。

Phase 1: 基础数据模型 + read 集成。
Phase 2: edit 接入。
Phase 3: write 刷新 + engine 集成 + /resume 重建。
Phase 4: compact 失效 + 边界情况。

设计原则：
- read 时创建 snippet（含行范围 + 文件版本号），作为"编辑凭证"
- edit 必须在 snippet 限定的行范围内搜索替换
- 文件被修改后旧 snippet 自动失效（version 不匹配）
- 会话恢复时从 JSONL 历史重建注册表
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class FileState:
    """会话内文件的缓存状态。"""

    file_path: str
    content: str
    mtime: float          # 读取时的文件修改时间
    version: int = 1      # 会话内修改次数（每次 edit/write 后 +1）
    encoding: str = "utf-8"
    line_endings: str = "LF"  # "LF" | "CRLF"（Phase 4 编辑时保留换行符用）


@dataclass
class FileSnippet:
    """一次 read 操作生成的"编辑凭证"。

    edit 必须携带有效的 snippet_id，且替换仅在 [start_line, end_line] 范围内搜索。
    file_version 用于检测"自读取后文件是否被修改过"。
    """

    id: str               # "snp_3" 或 "full_0"
    file_path: str
    start_line: int       # 1-based
    end_line: int         # 1-based, inclusive
    file_version: int     # 创建时的文件版本号
    scope_type: str       # "full"（全文件）或 "snippet"（部分）


# ---------------------------------------------------------------------------
# 模块级注册表（按 session_id 隔离）
# ---------------------------------------------------------------------------

# file_path → FileState
_file_states: Dict[str, Dict[str, FileState]] = {}

# snippet_id → FileSnippet
_snippets: Dict[str, Dict[str, FileSnippet]] = {}

# file_path → 当前 version
_file_versions: Dict[str, Dict[str, int]] = {}

# 每个 session 的 snippet 计数器
_snippet_counters: Dict[str, int] = {}
_full_counters: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def _ensure_session(session_id: str) -> None:
    """惰性初始化 session 的所有 dict。"""
    if session_id not in _file_states:
        _file_states[session_id] = {}
        _snippets[session_id] = {}
        _file_versions[session_id] = {}
        _snippet_counters[session_id] = 0
        _full_counters[session_id] = 0


def clear_session_state(session_id: str) -> None:
    """清空指定会话的所有状态。"""
    _file_states.pop(session_id, None)
    _snippets.pop(session_id, None)
    _file_versions.pop(session_id, None)
    _snippet_counters.pop(session_id, None)
    _full_counters.pop(session_id, None)


# --- 文件版本 ---


def get_file_version(session_id: str, file_path: str) -> int:
    """获取文件当前版本号（未记录则返回 0）。"""
    return _file_versions.get(session_id, {}).get(_normalize(file_path), 0)


def _bump_file_version(session_id: str, file_path: str) -> int:
    """自增版本号并返回新值。"""
    key = _normalize(file_path)
    versions = _file_versions.setdefault(session_id, {})
    versions[key] = versions.get(key, 0) + 1
    return versions[key]


# --- 文件状态 ---


def record_file_state(
    session_id: str,
    file_path: str,
    content: str,
    mtime: float = 0.0,
    *,
    bump_version: bool = False,
) -> FileState:
    """记录文件当前状态。bump_version=True 时自增版本号（edit/write 后调用）。

    首次记录时版本号默认1为 1（非 0），确保 snippet 版本校验的初始状态正确。
    """
    _ensure_session(session_id)
    key = _normalize(file_path)
    versions = _file_versions[session_id]
    if key not in versions:
        versions[key] = 1  # 首次记录初始化为 1
    if bump_version:
        _bump_file_version(session_id, file_path)
    state = FileState(
        file_path=key,
        content=content,
        mtime=mtime,
        version=get_file_version(session_id, file_path),
    )
    _file_states[session_id][key] = state
    return state


def get_file_state(session_id: str, file_path: str) -> Optional[FileState]:
    """获取文件状态（未记录则返回 None）。"""
    return _file_states.get(session_id, {}).get(_normalize(file_path))


def file_was_read(session_id: str, file_path: str) -> bool:
    """文件是否在当前会话中被读过。"""
    return get_file_state(session_id, file_path) is not None


# --- Snippet ---


def create_snippet(
    session_id: str,
    file_path: str,
    start_line: int,
    end_line: int,
    *,
    scope_type: str = "snippet",
) -> FileSnippet:
    """为一次 read 操作创建编辑凭证。

    读全文件时 scope_type="full"，部分读时 scope_type="snippet"。
    """
    _ensure_session(session_id)
    key = _normalize(file_path)
    version = get_file_version(session_id, file_path)

    if scope_type == "full":
        _full_counters[session_id] += 1
        sid = f"full_{_full_counters[session_id] - 1}"
    else:
        _snippet_counters[session_id] += 1
        sid = f"snp_{_snippet_counters[session_id] - 1}"

    snippet = FileSnippet(
        id=sid,
        file_path=key,
        start_line=start_line,
        end_line=end_line,
        file_version=version,
        scope_type=scope_type,
    )
    _snippets[session_id][sid] = snippet
    return snippet


def get_snippet(session_id: str, snippet_id: str) -> Optional[FileSnippet]:
    """按 id 查找 snippet。"""
    return _snippets.get(session_id, {}).get(snippet_id)


def is_snippet_stale(session_id: str, snippet: FileSnippet) -> bool:
    """snippet 是否已过期（文件版本升级）。"""
    current_version = get_file_version(session_id, snippet.file_path)
    return current_version > snippet.file_version


def invalidate_snippets_for_file(session_id: str, file_path: str) -> None:
    """让某个文件的所有已存在 snippet 失效（通过将 file_version 设为极高值）。

    注意：已存在的 snippet 实例不会变，但后续 is_snippet_stale() 会因为
    file_version 不匹配而返回 True。
    """
    _ensure_session(session_id)
    key = _normalize(file_path)
    # 将版本号设为极大值，让所有旧 snippet 失效
    _file_versions.setdefault(session_id, {})[key] = 999_999_999


def invalidate_all_snippets(session_id: str) -> None:
    """失效当前会话所有已存在 snippet（Phase 4: compact 后调用）。

    将所有已追踪文件的版本号设为极大值，让 compact 前的所有 snippet 全部过期。
    模型必须重新 read 才能获得新的有效 snippet。
    """
    _ensure_session(session_id)
    for key in list(_file_versions.get(session_id, {})):
        _file_versions[session_id][key] = 999_999_999


# --- 会话恢复（Phase 3 用到，提前定义接口）---


def rebuild_snippet(
    session_id: str,
    snippet_id: str,
    file_path: str,
    start_line: int,
    end_line: int,
    scope_type: str = "snippet",
) -> Optional[FileSnippet]:
    """从 JSONL 历史重建一个 snippet（会话恢复时使用）。"""
    _ensure_session(session_id)
    key = _normalize(file_path)
    version = get_file_version(session_id, file_path)

    snippet = FileSnippet(
        id=snippet_id,
        file_path=key,
        start_line=start_line,
        end_line=end_line,
        file_version=version,
        scope_type=scope_type,
    )
    _snippets[session_id][snippet_id] = snippet

    # 调整计数器，避免 id 冲突
    if snippet_id.startswith("full_"):
        try:
            num = int(snippet_id.split("_", 1)[1]) + 1
            _full_counters[session_id] = max(_full_counters.get(session_id, 0), num)
        except ValueError:
            pass
    elif snippet_id.startswith("snp_"):
        try:
            num = int(snippet_id.rsplit("_", 1)[1]) + 1
            _snippet_counters[session_id] = max(_snippet_counters.get(session_id, 0), num)
        except ValueError:
            pass

    return snippet


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _normalize(file_path: str) -> str:
    """用 resolve 后的绝对路径做 key，避免同一个文件因相对/绝对路径不同被记成两条。"""
    try:
        return str(Path(file_path).resolve())
    except Exception:
        return file_path
