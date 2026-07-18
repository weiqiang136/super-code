"""WebFetch tool: 获取 URL 内容并转换为纯文本，供 LLM 阅读。

使用标准库实现，无需额外依赖。
HTML 解析采用 html.parser，提取可读正文，过滤 script/style 等噪音标签。
"""
from __future__ import annotations

import urllib.request
import urllib.error
from html.parser import HTMLParser

from core.tool import Tool, ToolResult

# 单次返回的最大字符数，避免撑爆上下文
_MAX_CHARS = 20_000

# 这些标签的内容对 LLM 无意义，直接跳过
_SKIP_TAGS = {"script", "style", "noscript", "head", "meta", "link"}


class _TextExtractor(HTMLParser):
    """最小化 HTML → 纯文本提取器。"""

    def __init__(self):
        super().__init__()
        self._skip_depth = 0   # 当前处于需跳过的标签嵌套深度
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        # 块级标签结束时补换行，保留段落结构
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # 合并连续空白行，保留可读段落
        lines = [line.strip() for line in raw.splitlines()]
        cleaned: list[str] = []
        blank = 0
        for line in lines:
            if line:
                blank = 0
                cleaned.append(line)
            else:
                blank += 1
                if blank <= 1:          # 最多保留一个空行
                    cleaned.append("")
        return "\n".join(cleaned).strip()


class WebFetchTool(Tool):
    name = "WebFetch"
    description = (
        "Fetches the content of a URL and returns it as plain text. "
        "Use this to read documentation, web pages, or any HTTP/HTTPS resource. "
        "HTML is converted to readable text; non-HTML responses are returned as-is."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The HTTP/HTTPS URL to fetch",
            },
            "prompt": {
                "type": "string",
                "description": "Optional: what information you are looking for (for context only, does not filter output)",
            },
        },
        "required": ["url"],
    }

    def is_read_only(self) -> bool:
        # 只读网络请求，无需权限提示
        return True

    def get_activity_description(self, **kwargs) -> str | None:
        url = kwargs.get("url", "")
        return f"Fetching {url}" if url else None

    def execute(self, url: str, prompt: str = "", **kwargs) -> ToolResult:
        # 只允许 http/https，防止 file:// 等本地协议被滥用
        if not url.startswith(("http://", "https://")):
            return ToolResult(content="Error: Only http/https URLs are supported.", is_error=True)

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; SuperCode/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                content_type = resp.headers.get_content_type() or ""
                raw_bytes = resp.read(500_000)   # 最多读 500KB，避免大文件阻塞
        except urllib.error.HTTPError as e:
            return ToolResult(content=f"HTTP Error {e.code}: {e.reason}", is_error=True)
        except urllib.error.URLError as e:
            return ToolResult(content=f"URL Error: {e.reason}", is_error=True)
        except TimeoutError:
            return ToolResult(content="Error: Request timed out after 15s", is_error=True)
        except Exception as e:
            return ToolResult(content=f"Error: {e}", is_error=True)

        # 解码：优先用响应头声明的编码，fallback utf-8
        charset = resp.headers.get_content_charset() or "utf-8"
        try:
            text = raw_bytes.decode(charset, errors="replace")
        except LookupError:
            text = raw_bytes.decode("utf-8", errors="replace")

        # HTML 转纯文本；其他类型（JSON、纯文本等）直接返回
        if "html" in content_type:
            extractor = _TextExtractor()
            extractor.feed(text)
            text = extractor.get_text()

        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + f"\n\n... (truncated, {len(text)} chars total)"

        return ToolResult(content=text)
