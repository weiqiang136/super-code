"""命令黑名单：规则编译 + 检查入口。

内置规则按类别组织，每项 (re.Pattern, description)。
外部可通过 SandboxConfig.extra_patterns 追加更多规则。
"""
from __future__ import annotations

import re
import logging

from .config import SandboxConfig
from .path_protection import check_path as _check_path
from .network import check_network as _check_network

logger = logging.getLogger(__name__)

# ============================================================
# 内置黑名单规则（30+ 条，按类别分组）
# ============================================================

_BUILTIN_RULES: list[tuple[re.Pattern[str], str]] = []

def _add(pattern: str, desc: str) -> None:
    _BUILTIN_RULES.append((re.compile(pattern, re.I), desc))

# ---- 文件系统破坏 ----
_add(r'\brm\b[^|;&]*-(?:[a-z]*r[a-z]*f|[a-z]*f[a-z]*r)', "recursive force delete (rm -rf)")
_add(r'\brm\b[^|;&]*/(?:\s|$)', "delete root /")
_add(r'\b(?:rd|rmdir)\b[^|;&]*/[sS]', "recursive rmdir (Windows)")
_add(r'\bdel\b[^|;&]*/[sSq]', "recursive del (Windows)")
_add(r'>\s*(?:\w:)?[/\\](?:[Ww]indows[/\\]?)?[Ss]ystem32', "overwrite System32")
_add(r'>\s*/etc/(?:passwd|shadow|sudoers|hosts)\b', "overwrite system file")
_add(r'>\s*/dev/(?:sd|hd|nvme|mmcblk|disk|loop)\w*', "overwrite disk device")

# ---- 磁盘/分区操作 ----
_add(r'\b(?:newfs|mkfs|fsck)\b', "filesystem create/check")
_add(r'\bdiskutil\s+eraseDisk\b', "disk format (macOS)")
_add(r'\bformat\s+[a-zA-Z]:', "disk format (Windows)")
_add(r'\bdd\b.*\bof=/dev/(?:sd|hd|nvme|mmcblk)\w*', "raw disk write (dd)")

# ---- 系统配置修改 ----
_add(r'\bchmod\b[^|;&]*[47]77', "chmod 777 (world-writable)")
_add(r'\bchmod\b[^|;&]*/(?:etc|usr|bin|sbin|boot)\b', "chmod on system dir")
_add(r'\bchown\b[^|;&]*/(?:etc|usr|bin|sbin|boot)\b', "chown on system dir")

# ---- 代码执行 / 环境变量注入 ----
# eval + 网络下载/远程执行才算危险；bare eval "$(brew shellenv)" 是合法初始化
_add(r'\beval\b.*\b(?:curl|wget|nc|ssh|bash\s+-c|sh\s+-c)\b', "eval with remote code")
_add(r'\bLD_PRELOAD\b', "LD_PRELOAD injection")
_add(r'\bDYLD_INSERT_LIBRARIES\b', "DYLD_INSERT_LIBRARIES injection")
_add(r'\bLD_LIBRARY_PATH\b.*=', "LD_LIBRARY_PATH override")

# source with command substitution（动态路径）；不拦 source /path/to/venv/bin/activate
_add(r'\bsource\s+(?:\$\(|`|\$\{)', "source with dynamic path")

# ---- 包管理器全局安装 ----
_add(r'\bpip3?\b.*install\b(?!.*--user)', "pip install (global)")
_add(r'\bnpm\b.*install\b.*-g\b', "npm install -g")
_add(r'\bapt\b.*install\b', "apt install")
_add(r'\byum\b.*install\b', "yum install")
_add(r'\bbrew\b.*install\b', "brew install")

# ---- 远程传输 / 数据外泄 ----
_add(r'\bssh\b.*\b(?:root|admin)@', "ssh to privileged user")
_add(r'\bscp\b', "scp (remote file copy)")
_add(r'\brsync\b.*(?:@|::|rsync://)', "rsync to remote host")
_add(r'\bnc\b\s+-[lL]', "netcat listen mode")
_add(r'\bnc\b.*\s+-e\b', "netcat exec mode")

# ---- 持久化 / 定时任务 ----
_add(r'\bcrontab\b', "crontab modification")
_add(r'\bat\b\s', "at job scheduling")
_add(r'\bsystemctl\b\s+enable\b', "systemctl enable")

# ---- 网络/防火墙 ----
_add(r'\biptables\b', "iptables modification")
_add(r'\bnft\b\s', "nftables modification")

# ---- 挂载操作 ----
_add(r'\bmount\b\s', "mount filesystem")
_add(r'\bumount\b\s', "unmount filesystem")

# ---- 裸设备/内存访问 ----
_add(r'>\s*/dev/(?:mem|kmem|port)\b', "raw memory/port access")

# ---- 编码混淆绕过 ----
_add(r'base64\s+(?:-d|--decode).*\|?\s*(?:bash|sh|zsh|python|perl|ruby)\b', "base64 decode → execute")
_add(r'xxd\s+-r\b.*\|?\s*(?:bash|sh|zsh)\b', "xxd decode → execute")

# ---- Fork bomb ----
_add(r':\(\)\s*\{.*:\s*\|.*:.*&.*\}', "fork bomb")

del _add  # 清理辅助函数，避免污染模块命名空间


# ============================================================
# SandboxManager
# ============================================================

class SandboxManager:
    """命令黑名单过滤器。

    使用方式：
        config = SandboxConfig(enabled=True, excluded_commands=["docker"])
        sandbox = SandboxManager(config)
        allowed, reason = sandbox.check("rm -rf /")
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config

    def check(self, command: str) -> tuple[bool, str]:
        """检查命令是否允许执行。

        返回 (True, "") 表示允许，(False, reason) 表示拒绝。
        """
        config = self._config

        if config:
            # 1. 排除列表优先：excluded_commands 开头的命令跳过所有检查（白名单豁免）
            for exc in config.excluded_commands:
                if command.strip().startswith(exc):
                    return True, ""

            # 2. 命令精确黑名单（blocked_commands）
            cmd_name = command.strip().split()[0] if command.strip() else ""
            if cmd_name and cmd_name in config.blocked_commands:
                preview = _preview(command)
                reason = f"Sandbox blocked: forbidden command '{cmd_name}': {preview}"
                logger.info(reason)
                return False, reason

        # 3. 内置正则规则
        for pattern, desc in _BUILTIN_RULES:
            if pattern.search(command):
                preview = _preview(command)
                reason = f"Sandbox blocked: {desc}: {preview}"
                logger.info(reason)
                return False, reason

        # 4. 用户追加的正则规则
        if config and config.extra_patterns:
            for pat_str, desc in config.extra_patterns:
                try:
                    if re.search(pat_str, command, re.I):
                        preview = _preview(command)
                        reason = f"Sandbox blocked: {desc}: {preview}"
                        logger.info(reason)
                        return False, reason
                except re.error:
                    logger.warning("Invalid sandbox extra pattern: %s", pat_str)

        return True, ""

    def check_path(self, file_path: str, operation: str = "write") -> tuple[bool, str]:
        """检查文件路径是否受保护（禁止写入/读取的目录）。

        operation: "write" 或 "read"
        返回 (True, "") 表示允许，(False, reason) 表示拒绝。
        """
        return _check_path(file_path, operation)

    def check_network(self, command: str) -> tuple[bool, str]:
        """检查命令中网络外发操作的目标域名是否合规。

        依赖 config 中的 allowed_domains / denied_domains 配置。
        两者都为空时不做任何限制（向后兼容）。
        返回 (True, "") 表示允许，(False, reason) 表示拒绝。
        """
        if self._config is None:
            return True, ""
        return _check_network(command, self._config.allowed_domains, self._config.denied_domains)


def _preview(command: str, max_len: int = 80) -> str:
    return command[:max_len] + ("..." if len(command) > max_len else "")
