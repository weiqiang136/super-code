"""沙箱模块：命令黑名单过滤（非 OS 级隔离）。

启用方式：--sandbox CLI 参数，或 super-code.json 中 sandbox.enabled = true。

目录结构：
    config.py          — SandboxConfig dataclass，从 super-code.json 解析
    blacklist.py       — 命令黑名单规则 + SandboxManager
    path_protection.py — 文件路径保护（禁止写入/读取的目录）
"""
from __future__ import annotations

from .blacklist import SandboxManager
from .config import SandboxConfig

__all__ = ["SandboxManager", "SandboxConfig"]
