"""WebSearch tool: search the web using Bing and return structured results.

Uses Bing.com search — zero dependencies, pure stdlib. Works in China.
"""
from __future__ import annotations

import html
import os
import re
import urllib.request
import urllib.error
import urllib.parse
from html.parser import HTMLParser

from core.tool import Tool, ToolResult

_MAX_RESULTS = 10
_DEFAULT_RESULTS = 5
_MAX_CHARS = 3_000


class _BingParser(HTMLParser):
    """Parse Bing search result page HTML.

    Bing results are in <li class="b_algo"> blocks:
      <h2><a href="URL">title</a></h2>
      <div class="b_caption"><p>snippet</p></div>

    Uses depth counters to handle nested elements inside b_caption
    (e.g. <div class="b_caption"><div class="f">...</div></div>).
    """

    def __init__(self):
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_algo = False
        self._in_title = False           # inside <h2><a>
        self._in_caption = False          # inside <div class="b_caption">
        self._caption_depth = 0           # 嵌套 div 深度计数
        self._text: list[str] = []
        self._current_href = ""
        self._current_title = ""
        self._current_snippet = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = dict(attrs)
        cls = d.get("class", "")

        if tag == "li" and "b_algo" in cls.split():
            self._in_algo = True
            self._in_title = False
            self._in_caption = False
            self._caption_depth = 0
            self._current_href = ""
            self._current_title = ""
            self._current_snippet = ""

        elif self._in_algo and not self._current_href:
            # 第一个 <h2><a href="..."> 就是结果标题链接
            if tag == "h2":
                self._in_title = True
            elif self._in_title and tag == "a":
                self._current_href = d.get("href", "")
                self._text = []

        elif self._in_algo:
            if tag == "div" and "b_caption" in cls.split():
                self._in_caption = True
                self._caption_depth = 1
            elif self._in_caption and tag == "div":
                self._caption_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag == "a":
            self._current_title = "".join(self._text).strip()
        elif self._in_title and tag == "h2":
            self._in_title = False
        elif self._in_caption and tag == "div":
            self._caption_depth -= 1
            if self._caption_depth > 0:
                return  # 内层 </div>，跳过
            self._in_caption = False
            # 外层 b_caption 的 </div>，收尾
            if self._current_href:
                self.results.append({
                    "title": self._current_title,
                    "href": self._current_href,
                    "snippet": self._current_snippet,
                })
            self._in_algo = False
        elif self._in_algo and tag == "li":
            # 有些结果可能没有 b_caption（如视频/图片结果），在 </li> 时兜底
            if self._current_href and not self._current_snippet:
                self.results.append({
                    "title": self._current_title,
                    "href": self._current_href,
                    "snippet": self._current_snippet,
                })
            self._in_algo = False

    def handle_data(self, data: str) -> None:
        if self._in_title and self._current_href:
            self._text.append(data)
        elif self._in_caption:
            self._current_snippet += data

    def finalize(self) -> None:
        if self._in_algo and self._current_href:
            self.results.append({
                "title": self._current_title,
                "href": self._current_href,
                "snippet": self._current_snippet.strip(),
            })


class WebSearchTool(Tool):
    name = "WebSearch"
    description = (
        "Searches the web using Bing and returns results as structured text. "
        "Use this to find documentation, code examples, or any information on the web. "
        "Each result includes a title, URL, and text snippet."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "max_results": {
                "type": "integer",
                "description": f"Maximum number of results (1-{_MAX_RESULTS}, default {_DEFAULT_RESULTS})",
                "default": _DEFAULT_RESULTS,
            },
        },
        "required": ["query"],
    }

    def is_read_only(self) -> bool:
        return True

    def get_activity_description(self, **kwargs) -> str | None:
        query = kwargs.get("query", "")
        return f"Searching: {query}" if query else None

    def execute(self, query: str, max_results: int = _DEFAULT_RESULTS, **kwargs) -> ToolResult:
        if not query.strip():
            return ToolResult(content="Error: query must not be empty.", is_error=True)

        max_results = min(max(max_results, 1), _MAX_RESULTS)

        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.bing.com/search?q={encoded_query}"

        # 代理支持：读取环境变量 HTTPS_PROXY / HTTP_PROXY
        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        opener = (
            urllib.request.build_opener(
                urllib.request.ProxyHandler({"https": proxy_url, "http": proxy_url})
            )
            if proxy_url
            else urllib.request.build_opener()
        )

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )
            with opener.open(req, timeout=15) as resp:
                raw_bytes = resp.read(200_000)
        except urllib.error.HTTPError as e:
            return ToolResult(content=f"Search error: HTTP {e.code} {e.reason}", is_error=True)
        except urllib.error.URLError as e:
            return ToolResult(content=f"Search error: {e.reason}", is_error=True)
        except TimeoutError:
            return ToolResult(content="Search error: request timed out after 15s", is_error=True)
        except Exception as e:
            return ToolResult(content=f"Search error: {e}", is_error=True)

        charset = resp.headers.get_content_charset() or "utf-8"
        try:
            raw_html = raw_bytes.decode(charset, errors="replace")
        except LookupError:
            raw_html = raw_bytes.decode("utf-8", errors="replace")

        parser = _BingParser()
        try:
            parser.feed(raw_html)
            parser.finalize()
        except Exception:
            return ToolResult(content="Search error: failed to parse results.", is_error=True)

        results = parser.results[:max_results]

        if not results:
            return ToolResult(content="No results found.")

        lines = []
        total = 0
        for i, r in enumerate(results, 1):
            title = r["title"] or "(no title)"
            href = r["href"]
            # 清理 HTML 标签和实体
            snippet = html.unescape(re.sub(r"<[^>]+>", "", r["snippet"].strip()))
            line = f"{i}. **{title}**\n   {href}\n   {snippet}"
            total += len(line)
            if total > _MAX_CHARS:
                if i == 1:
                    lines.append(line[:_MAX_CHARS] + "...")
                break
            lines.append(line)

        return ToolResult(content="\n\n".join(lines))
