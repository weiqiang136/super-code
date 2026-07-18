"""沙箱配置：SandboxConfig dataclass + 从 super-code.json 解析。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SandboxConfig:
    """沙箱所有可配置项。"""

    enabled: bool = False
    # 命令级黑名单（精确匹配命令名，如 "mkfs"）
    blocked_commands: list[str] = field(default_factory=list)
    # 正则黑名单（额外规则，追加到内置规则之后）
    extra_patterns: list[tuple[str, str]] = field(default_factory=list)
    # 跳过沙箱检查的命令前缀（如 "docker", "bazel"）
    excluded_commands: list[str] = field(default_factory=list)
    # 沙箱激活时自动批准 Bash 调用（因为命令已被黑名单保护）
    auto_approve_if_sandboxed: bool = False
    # 网络外发域名白名单（非空时：只允许列表内的域名外发）
    allowed_domains: list[str] = field(default_factory=list)
    # 网络外发域名黑名单（命中直接拒绝，优先级高于白名单）
    denied_domains: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SandboxConfig":
        """从 super-code.json 的 sandbox 段创建配置。"""
        if not d:
            return cls()
        return cls(
            enabled=bool(d.get("enabled", False)),
            blocked_commands=list(d.get("blocked_commands") or []),
            extra_patterns=list(d.get("extra_patterns") or []),
            excluded_commands=list(d.get("excluded_commands") or []),
            auto_approve_if_sandboxed=bool(d.get("auto_approve_if_sandboxed", False)),
            allowed_domains=list(d.get("allowed_domains") or []),
            denied_domains=list(d.get("denied_domains") or []),
        )
