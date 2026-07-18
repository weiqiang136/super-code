"""文件路径保护：硬编码的禁止读写路径列表（不可配置的安全基线）。

Phase 3 在 Edit/Write/Read/Bash 工具中插入路径检查，堵住「绕开 Bash
直接用文件工具写敏感路径」的旁路。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from core.config import get_portable_dir


# ============================================================
# 受保护路径初始化（模块加载时执行一次）
# ============================================================

def _home() -> str:
    return os.path.expanduser("~")


def _resolve(p: str) -> str:
    """展开 ~ + realpath，取规范绝对路径；路径不存在时回退到 expanduser。"""
    expanded = os.path.expanduser(p)
    try:
        return os.path.realpath(expanded)
    except OSError:
        return os.path.normpath(expanded)


# ---------- 禁止写入 ----------
_portable = str(get_portable_dir())
_DENIED_WRITE_PREFIXES: list[str] = [
    # 防沙箱逃逸：super-code 全局配置文件（仅保护 super-code.json 自身，
    # 不保护 mcp.json / skills / plans / projects 等子目录）
    _resolve("~/.config/super-code/super-code.json"),
    # 防沙箱逃逸：便携分发目录中的关键配置文件
    _resolve(os.path.join(_portable, "super-code.json")),
    # SSH / GPG 密钥目录
    _resolve("~/.ssh"),
    _resolve("~/.gnupg"),
]

if sys.platform == "win32":
    windir = os.environ.get("SystemRoot", "C:\\Windows")
    _DENIED_WRITE_PREFIXES.extend([
        _resolve(windir),
        _resolve(windir + "\\System32"),
        _resolve("~\\AppData\\Roaming"),
    ])
else:
    _DENIED_WRITE_PREFIXES.extend([
        _resolve("/etc"),
        _resolve("/usr"),
        _resolve("/bin"),
        _resolve("/sbin"),
        _resolve("/boot"),
        _resolve("/dev"),
    ])

# 去重 + 排序
_DENIED_WRITE_PREFIXES = sorted(set(_DENIED_WRITE_PREFIXES))

# ---------- 禁止读取 ----------
_DENIED_READ_PREFIXES: list[str] = [
    _resolve("~/.ssh"),
    _resolve("~/.gnupg"),
]

if sys.platform == "win32":
    _DENIED_READ_PREFIXES.extend([
        _resolve("~\\AppData\\Roaming"),
    ])
else:
    _DENIED_READ_PREFIXES.extend([
        _resolve("/proc"),
        _resolve("/etc/ssl/private"),
    ])

_DENIED_READ_PREFIXES = sorted(set(_DENIED_READ_PREFIXES))


# ============================================================
# 公开 API
# ============================================================

def check_path(file_path: str, operation: str = "write") -> tuple[bool, str]:
    """检查路径是否受保护。

    Args:
        file_path: 文件路径（相对或绝对，支持 ~ 展开）
        operation: "write" 或 "read"

    Returns:
        (True, "") 表示允许，(False, reason) 表示拒绝。
    """
    if not file_path:
        return True, ""

    try:
        normalized = _resolve(file_path)
    except Exception:
        # 路径解析失败（如包含非法字符），仍然拒绝以策安全
        return False, f"Sandbox blocked: invalid path: {file_path[:80]}"

    prefixes = _DENIED_WRITE_PREFIXES if operation == "write" else _DENIED_READ_PREFIXES

    for prefix in prefixes:
        # 精确匹配前缀本身，或前缀后跟分隔符
        if normalized == prefix or _path_starts_with(normalized, prefix):
            preview = file_path[:80] + ("..." if len(file_path) > 80 else "")
            return False, f"Sandbox blocked: {operation} to protected path: {preview}"

    return True, ""


def _path_starts_with(path: str, prefix: str) -> bool:
    """检查 path 是否在 prefix 之下（跨平台分隔符安全）。"""
    if not path.startswith(prefix):
        return False
    rest = path[len(prefix):]
    if not rest:
        return True
    seps = (os.sep, os.altsep) if os.altsep else (os.sep,)
    return rest[0] in seps
