"""网络外发控制：从命令中提取目标域名，按黑白名单过滤。

Phase 4 — 防止 curl/wget/Invoke-WebRequest 将数据外泄到未授权域名。
"""
from __future__ import annotations

import re
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# 网络外发命令匹配（curl、wget、PowerShell Invoke-WebRequest/iwr）
# 使用 \b 单词边界匹配，覆盖 /usr/bin/curl 这类带路径的调用
_NET_CMD_RE = re.compile(
    r'\b(?:curl(?:\.exe)?|wget(?:\.exe)?|Invoke-WebRequest|iwr)\b',
    re.I,
)

# 从命令中提取 URL（带 scheme 的完整 URL，或带路径/端口的裸域名）
_URL_RE = re.compile(
    r'(?:https?|ftp)://[^\s"\';&|`$()<>]+'              # http(s)://host/path
    r'|'                                                  # 或
    r'(?:^|\s)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}'  # 裸域名
    r'(?::\d+(?:/\S*)?|/\S+)',                           # 必须有路径或端口（区分文件参数）
    re.I,
)

# 检测 curl/wget 的文件上传参数（-d @file、--data @file、-T file、--upload-file file、--post-file file）
# 上传本地文件是数据外泄的关键指标
_FILE_UPLOAD_RE = re.compile(
    r'(?:-d|--data(?:-raw|-binary|-urlencode)?)[\s=]*@\S|'   # -d@file、-d @file、--data=@file
    r'(?:-T|--upload-file)\s+\S|'                              # -T file、--upload-file file
    r'--post-file[=\s]\S',                                     # --post-file=path、--post-file path
    re.I,
)


def check_network(command: str, allowed: list[str], denied: list[str]) -> tuple[bool, str]:
    """检查命令中网络外发操作的目标域名是否合规。

    Args:
        command: 完整的 shell 命令
        allowed: 域名白名单（非空时只允许列表中的域名）
        denied: 域名黑名单（命中直接拒绝）

    Returns:
        (True, "") 允许执行，(False, reason) 拒绝。
    """
    # 1. 判断是否为网络外发命令
    if not _NET_CMD_RE.search(command):
        return True, ""

    # 2. 文件上传检查（独立于域名配置，始终生效）
    if _FILE_UPLOAD_RE.search(command):
        preview = command[:80] + ("..." if len(command) > 80 else "")
        reason = f"Sandbox blocked: file upload via network command: {preview}"
        logger.info(reason)
        return False, reason

    # 3. 无域名配置 → 不限制（文件上传已在上一步拦截）
    if not allowed and not denied:
        return True, ""

    # 4. 提取命令中所有 URL（用 finditer，避免 capturing group 导致 findall 只返回 scheme）
    urls = [m.group(0) for m in _URL_RE.finditer(command)]
    if not urls:
        return True, ""

    # 5. 对每个 URL 提取域名并检查
    for raw_url in urls:
        url_str = raw_url.strip()
        hostname = _extract_hostname(url_str)

        if not hostname:
            continue

        # 记录请求 URL（用于日志和错误信息，脱敏处理 query 参数）
        preview = _safe_url_preview(url_str)

        # 6. 黑名单优先
        for denied_domain in denied:
            if _domain_match(hostname, denied_domain):
                reason = f"Sandbox blocked: network request to denied domain '{hostname}': {preview}"
                logger.info(reason)
                return False, reason

        # 7. 白名单检查（白名单非空时，未命中 = 拒绝）
        if allowed:
            matched = any(_domain_match(hostname, a) for a in allowed)
            if not matched:
                reason = f"Sandbox blocked: network request to unlisted domain '{hostname}': {preview}"
                logger.info(reason)
                return False, reason

    return True, ""


def _extract_hostname(url_str: str) -> str:
    """从 URL 中提取主机名。

    处理两种格式：
      - http(s)://host/path  → urlparse 正常解析
      - host/path, host:8080/path  → 裸域名，需手动提取
    """
    if "://" in url_str:
        try:
            parsed = urlparse(url_str)
            return parsed.hostname or ""
        except Exception:
            return ""
    # 裸域名：取第一个 '/' 或 ':' 之前的部分作为 hostname
    bare = url_str.split("/")[0]  # "host:8080" or "host"
    return bare.split(":")[0]      # "host"


def _domain_match(hostname: str, rule: str) -> bool:
    """域名匹配：精确匹配或子域名匹配。

    rule="example.com" 匹配 example.com 和 *.example.com
    rule=".example.com" 等价于 rule="example.com"
    """
    rule = rule.lstrip(".")
    return hostname == rule or hostname.endswith("." + rule)


def _safe_url_preview(url: str, max_len: int = 80) -> str:
    """URL 预览（截断前先尝试去掉 query 参数，避免 token 信息泄漏到日志）。"""
    try:
        parsed = urlparse(url)
        clean = f"{parsed.scheme}://{parsed.hostname}{parsed.path or ''}"
        if clean and len(clean) > 20:
            return clean[:max_len] + ("..." if len(clean) > max_len else "")
    except Exception:
        pass
    return url[:max_len] + ("..." if len(url) > max_len else "")
