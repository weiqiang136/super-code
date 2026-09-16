"""Grep content 模式长行锚定截断的测试。

背景（2026-09-14 esigning 事故）：minified idx.js 整个文件一行，
Grep head_limit 数行不数字符，两条结果各 ~31.4 万 token 打爆 1M 窗口。
"""
import sys
sys.path.insert(0, "src")

import re
from types import SimpleNamespace

from tools.grep_tool import (
    GREP_LINE_WINDOW_CHARS,
    GrepTool,
    _enforce_total_cap,
    _merge_match_windows,
    _split_rg_prefix,
    _truncate_content_line,
)


def test_merge_single_window():
    """单个命中 → ±半窗。"""
    assert _merge_match_windows(1_200_000, [(500_000, 500_010)]) == [
        (499_000, 501_010)]


def test_merge_nearby_spans_into_one_window():
    """相邻命中（窗口重叠或 gap ≤ 200）→ 合并为一个窗口。"""
    spans = [(1000, 1010), (3000, 3010), (5000, 5010)]
    assert _merge_match_windows(50_000, spans) == [(0, 6010)]


def test_merge_distant_spans_stay_separate():
    """远距命中保持独立窗口。"""
    spans = [(0, 10), (20_000, 20_010), (40_000, 40_010)]
    ws = _merge_match_windows(50_000, spans)
    assert len(ws) == 3
    assert ws[0] == (0, 1010)


def test_merge_skips_zero_width_spans():
    """零宽匹配（如裸 lookahead）跳过，防空窗口。"""
    assert _merge_match_windows(50_000, [(500, 500), (5000, 5010)]) == [
        (4000, 6010)]


RG_API = re.compile(r"api/[a-zA-Z0-9/_.-]{3,60}")


# ---------------------------------------------------------------------------
# _split_rg_prefix
# ---------------------------------------------------------------------------

def test_split_prefix_windows_path_with_lineno():
    """Windows 盘符路径 + 行号前缀正确拆分（非贪婪不误吃内容）。"""
    prefix, content = _split_rg_prefix(
        r"D:\code\esigning\idx.js:3:function oC(e,t){}")
    assert prefix == r"D:\code\esigning\idx.js:3:"
    assert content == "function oC(e,t){}"


def test_split_prefix_context_line_dash_separator():
    """rg 上下文行用 `-` 分隔符，同样能拆。"""
    prefix, content = _split_rg_prefix("src/a.py-12-context text")
    assert prefix == "src/a.py-12-"
    assert content == "context text"


def test_split_prefix_no_match_returns_whole_line():
    """拆不出前缀（如未开 -n）→ 整行当 content。"""
    prefix, content = _split_rg_prefix("just some long line")
    assert prefix == ""
    assert content == "just some long line"


# ---------------------------------------------------------------------------
# _truncate_content_line
# ---------------------------------------------------------------------------

def test_short_line_untouched():
    """短行（≤ 2000 字符）原样返回。"""
    line = "x" * 100
    assert _truncate_content_line("f.js:1:", line, RG_API) == "f.js:1:" + line


def test_long_line_head_fallback_when_regex_none():
    """rg 方言编译失败（regex=None）→ 头部截断 + 「命中位置未知」标注。"""
    content = "y" * 5000
    out = _truncate_content_line("f.js:1:", content, None)
    assert out.startswith("f.js:1:" + "y" * GREP_LINE_WINDOW_CHARS)
    assert "line truncated" in out
    assert "match position unknown" in out


def test_long_line_no_match_head_fallback():
    """行内匹配不到（rg/Python 方言差异）→ 同样头部截断兜底。"""
    content = "z" * 5000
    out = _truncate_content_line("f.js:1:", content, RG_API)
    assert out.startswith("f.js:1:" + "z" * GREP_LINE_WINDOW_CHARS)
    assert "line truncated" in out


def test_minified_line_window_anchored_on_match():
    """核心场景：120 万字符 minified 行，命中点 ±1000 开窗。"""
    content = "q" * 100_000 + "api/doc/detail" + "q" * 100_000
    out = _truncate_content_line("idx.js:3:", content, RG_API)
    assert "api/doc/detail" in out                     # 命中点必在窗口内
    assert "…" in out                                  # 两侧省略标记
    assert "200,014 chars" in out                      # 标注原始总长
    assert len(out) < GREP_LINE_WINDOW_CHARS + 500     # 输出压到窗口量级


def test_two_distant_matches_two_windows():
    """两个远距命中 → 两个独立窗口都输出，标注总数。"""
    content = "q" * 9000 + "api/a" + "q" * 9000 + "api/a" + "q" * 9000
    out = _truncate_content_line("idx.js:3:", content, RG_API)
    assert out.count("api/a") == 2
    assert "2 match(es)" in out
    assert "2 window(s)" in out


def test_more_matches_than_quota_note_shown():
    """命中窗口数超配额 3 → 只显示前 3 个 + 剩余引导。"""
    content = ""
    for i in range(4):
        content += "q" * 10000 + "api/unique" + str(i) + "pad"
    out = _truncate_content_line("idx.js:3:", content, re.compile(r"api/unique\d"))
    assert out.count("api/unique") == 3        # 只显示前 3 个窗口
    assert "4 match(es)" in out                # 告知行内总数
    assert "output_mode" in out                # 引导词（count / narrow pattern）


def test_huge_match_span_window_clamped():
    """病态模式回归（2026-09-15 实测）：api/[a-z]+ 吞掉 5 万小写填充串，
    span 本身巨长 → 单窗必须 clamp 到 ~3000 字符内，超额报数。"""
    content = "x" * 50_000 + "api/gamma" + "x" * 50_000
    out = _truncate_content_line("idx.js:3:", content, re.compile(r"api/[a-z]+"))
    assert "api/gamma" in out                          # 命中点必在窗口内
    assert len(out) < 5_000                            # 单窗被压住（原实现 52K+）
    assert "rendered partially" in out                 # span 截断标注
    assert "50,009" in out                             # 告知真实 span 长度


# ---------------------------------------------------------------------------
# _enforce_total_cap
# ---------------------------------------------------------------------------

def test_total_cap_truncates_and_notes():
    """40 行 × 1000 字符 > 32K → 丢行 + 标注剩余。"""
    parts = ["L" * 1000] * 40
    out = _enforce_total_cap(parts)
    assert "truncated at 32,768 chars" in out
    assert out.count("\n") >= 30   # 大部分行保留


def test_total_cap_under_no_note():
    parts = ["L" * 1000] * 10
    out = _enforce_total_cap(parts)
    assert "truncated" not in out


def test_total_cap_hard_slices_oversized_single_part():
    """回归（2026-09-15 实测）：单行文件 → 单个巨 part，第一个 part 无条件
    放行导致 32K 总闸被穿透 → 必须对超顶 part 硬切。"""
    parts = ["L" * 40_000]
    out = _enforce_total_cap(parts)
    assert len(out) < 40_000
    assert "hard-truncated at 32,768 chars" in out


# ---------------------------------------------------------------------------
# GrepTool.execute / _python_grep 集成
# ---------------------------------------------------------------------------

def _fake_run(stdout: str):
    return lambda *a, **k: SimpleNamespace(stdout=stdout, stderr="", returncode=0)


def test_execute_content_mode_truncates_minified(tmp_path, monkeypatch):
    """端到端：minified 单行 content 结果被锚定截断。"""
    big = "q" * 100_000 + "api/doc/detail" + "q" * 100_000
    f = tmp_path / "idx.js"
    f.write_text(big, encoding="utf-8")
    monkeypatch.setattr("tools.grep_tool.subprocess.run",
                        _fake_run(f"{f}:1:{big}\n"))
    tool = GrepTool()
    res = tool.execute(pattern="api/doc/detail", path=str(f),
                       output_mode="content")
    assert "api/doc/detail" in res.content
    assert "line truncated" in res.content
    assert len(res.content) < 5000


def test_execute_files_mode_untouched(tmp_path, monkeypatch):
    """files_with_matches 模式完全不受截断影响。"""
    f = tmp_path / "idx.js"
    f.write_text("q" * 100_000 + "api/doc/detail" + "q" * 100_000,
                 encoding="utf-8")
    monkeypatch.setattr("tools.grep_tool.subprocess.run",
                        _fake_run(str(f) + "\n"))
    tool = GrepTool()
    res = tool.execute(pattern="api/doc/detail", path=str(f),
                       output_mode="files_with_matches")
    assert res.content == str(f)


def test_python_grep_fallback_truncates(tmp_path):
    """rg 缺失走 Python fallback 时同样截断。"""
    big = "q" * 100_000 + "api/doc/detail" + "q" * 100_000
    f = tmp_path / "idx.js"
    f.write_text(big, encoding="utf-8")
    tool = GrepTool()
    res = tool._python_grep("api/doc/detail", str(f), None, False, "content")
    assert "api/doc/detail" in res.content
    assert "line truncated" in res.content
