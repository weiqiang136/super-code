"""
test_cmd_memory.py — Step 10 /memory 命令升级。

测试策略：
    - 用真实 tmp_path 当 memory_dir，写若干 .md 文件
    - 用 StringIO Console 捕获打印
    - monkeypatch subprocess.run 验证编辑器调用而不真的弹编辑器

覆盖路径：
    - 无 memory_dir → 警告
    - 无参数 → 打印 MEMORY.md（旧行为）
    - 无参数 + 无 MEMORY.md → "No memories consolidated yet"
    - list → 列出 topic
    - list + 空目录 → "No topic memories yet"
    - 数字范围内 → 调编辑器
    - 数字越界 → 警告
    - 数字 + 空目录 → 警告
    - substring 命中一个 → 调编辑器
    - substring 命中多个 → 提示 + 打开第一个
    - substring 无命中 → 警告
    - $EDITOR 未设 + 编辑器不存在 → 友好警告
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from commands import _cmd_memory


def _ctx(memory_dir: Path) -> SimpleNamespace:
    """构造一个最小 CommandContext（dataclass 用 SimpleNamespace 替代以避免循环依赖）。"""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200, color_system=None)
    return SimpleNamespace(
        console=console,
        memory_dir=memory_dir,
        _buf=buf,  # 测试用：取打印内容
    )


def _output(ctx: SimpleNamespace) -> str:
    return ctx._buf.getvalue()


def _write(path: Path, name: str, desc: str = "", body: str = "body",
           mtime: float | None = None) -> Path:
    fp = path / f"{name}.md"
    fp.write_text(
        f"---\nname: {name}\ndescription: {desc}\ntype: feedback\n---\n{body}\n",
        encoding="utf-8",
    )
    if mtime is not None:
        os.utime(fp, (mtime, mtime))
    return fp


# ---------------------------------------------------------------------------
# 无 memory_dir
# ---------------------------------------------------------------------------

def test_no_memory_dir_warns():
    ctx = SimpleNamespace(console=Console(file=io.StringIO(), color_system=None),
                          memory_dir=None, _buf=None)
    ctx._buf = ctx.console.file
    _cmd_memory(ctx, "")
    assert "not available" in _output(ctx)


# ---------------------------------------------------------------------------
# 无参数：保持旧行为
# ---------------------------------------------------------------------------

def test_no_arg_prints_index(tmp_path: Path):
    (tmp_path / "MEMORY.md").write_text("- [Test](test.md) — hook\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "")
    out = _output(ctx)
    assert "[Test](test.md)" in out
    # 提示信息出现
    assert "/memory list" in out


def test_no_arg_empty_memory_md(tmp_path: Path):
    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "")
    assert "No memories consolidated yet" in _output(ctx)


# ---------------------------------------------------------------------------
# list 子命令
# ---------------------------------------------------------------------------

def test_list_shows_numbered_entries(tmp_path: Path):
    _write(tmp_path, "alpha", desc="first one", mtime=2_000_000)
    _write(tmp_path, "beta", desc="second one", mtime=1_000_000)
    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "list")
    out = _output(ctx)
    assert "Available memories (2)" in out
    # 按 mtime 倒序：alpha 在前
    a_pos = out.find("alpha.md")
    b_pos = out.find("beta.md")
    assert a_pos != -1 and b_pos != -1 and a_pos < b_pos
    # 编号 1./2. 都出现
    assert "1." in out and "2." in out


def test_list_empty_directory(tmp_path: Path):
    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "list")
    assert "No topic memories" in _output(ctx)


# ---------------------------------------------------------------------------
# 数字
# ---------------------------------------------------------------------------

def _stub_subprocess(monkeypatch) -> list[list[str]]:
    """把 subprocess.run 替换为记录调用的桩，返回成功；返回收到的 cmd 列表。"""
    captured: list[list[str]] = []

    def fake_run(cmd, *a, **kw):
        captured.append(list(cmd))
        return SimpleNamespace(returncode=0)

    # _open_in_editor 内部用 `import subprocess` —— 必须 patch sys.modules 里的那个
    import subprocess as _sp
    monkeypatch.setattr(_sp, "run", fake_run)
    return captured


def test_number_opens_correct_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EDITOR", "fake-editor")
    _write(tmp_path, "alpha", mtime=2_000_000)
    _write(tmp_path, "beta", mtime=1_000_000)
    cmds = _stub_subprocess(monkeypatch)

    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "1")

    # mtime 倒序 → alpha 是 #1
    assert len(cmds) == 1
    assert cmds[0][0] == "fake-editor"
    assert cmds[0][1].endswith("alpha.md")


def test_number_out_of_range(tmp_path: Path, monkeypatch):
    _write(tmp_path, "only")
    cmds = _stub_subprocess(monkeypatch)
    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "5")
    assert "out of range" in _output(ctx)
    assert cmds == []  # 不应调用编辑器


def test_number_on_empty_directory(tmp_path: Path, monkeypatch):
    cmds = _stub_subprocess(monkeypatch)
    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "1")
    assert "No topic memories to open" in _output(ctx)
    assert cmds == []


# ---------------------------------------------------------------------------
# substring
# ---------------------------------------------------------------------------

def test_substring_single_match_opens(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EDITOR", "fake-editor")
    _write(tmp_path, "feedback-no-mock", desc="no mocking DB")
    _write(tmp_path, "user-prefers-uv")
    cmds = _stub_subprocess(monkeypatch)

    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "uv")
    assert len(cmds) == 1
    assert cmds[0][1].endswith("user-prefers-uv.md")


def test_substring_matches_description(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EDITOR", "fake-editor")
    _write(tmp_path, "feedback-x", desc="no mocking the database in tests")
    _write(tmp_path, "other")
    cmds = _stub_subprocess(monkeypatch)

    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "mocking")
    assert len(cmds) == 1
    assert cmds[0][1].endswith("feedback-x.md")


def test_substring_case_insensitive(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EDITOR", "fake-editor")
    _write(tmp_path, "Feedback-Important")
    cmds = _stub_subprocess(monkeypatch)
    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "IMPORTANT")
    assert len(cmds) == 1


def test_substring_multiple_matches_opens_first(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EDITOR", "fake-editor")
    _write(tmp_path, "feedback-a", mtime=2_000_000)
    _write(tmp_path, "feedback-b", mtime=1_000_000)
    cmds = _stub_subprocess(monkeypatch)

    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "feedback")
    out = _output(ctx)
    assert "2 matches" in out
    # 第一个 = mtime 最大 = feedback-a
    assert cmds[0][1].endswith("feedback-a.md")


def test_substring_no_match(tmp_path: Path, monkeypatch):
    _write(tmp_path, "alpha")
    cmds = _stub_subprocess(monkeypatch)
    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "nonexistent")
    assert "No memory matches" in _output(ctx)
    assert cmds == []


# ---------------------------------------------------------------------------
# editor fallback
# ---------------------------------------------------------------------------

def test_editor_not_found_friendly_message(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EDITOR", "definitely-not-an-editor-xyz")
    _write(tmp_path, "only")

    # 让 subprocess.run 抛 FileNotFoundError 模拟编辑器不在 PATH
    import subprocess as _sp
    def boom(*a, **kw):
        raise FileNotFoundError("not found")
    monkeypatch.setattr(_sp, "run", boom)

    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "1")
    out = _output(ctx)
    assert "Editor not found" in out
    assert "definitely-not-an-editor-xyz" in out


def test_editor_nonzero_exit_reported(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EDITOR", "fake-editor")
    _write(tmp_path, "only")
    import subprocess as _sp
    monkeypatch.setattr(_sp, "run", lambda *a, **kw: SimpleNamespace(returncode=2))

    ctx = _ctx(tmp_path)
    _cmd_memory(ctx, "1")
    assert "exited with code 2" in _output(ctx)
