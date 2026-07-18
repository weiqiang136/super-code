"""
test_memory_dir.py — Step 2 项目级记忆目录隔离的测试。

覆盖：
- _sanitize_path：基础替换、空字符串、超长截断 + hash 后缀、稳定性
- get_memory_dir：
  * git 仓库 → 返回 <base>/projects/<sanitized-root>/memory/
  * 同一 git 仓库的不同子目录 → 返回相同路径（worktree 子目录场景）
  * 非 git 目录 → 回退到 GLOBAL_MEMORY_DIR
  * git 命令缺失 / 报错 → 同样回退（mock subprocess）
- MEMORY_DIR 向后兼容常量仍存在且等于 GLOBAL_MEMORY_DIR
"""
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from features import memory as memory_mod
from features.memory import (
    GLOBAL_MEMORY_DIR,
    MAX_SANITIZED_LENGTH,
    MEMORY_DIR,
    _sanitize_path,
    get_memory_dir,
)


# ---------------------------------------------------------------------------
# _sanitize_path
# ---------------------------------------------------------------------------

def test_sanitize_replaces_non_alnum():
    assert _sanitize_path("/Users/foo/my-project") == "-Users-foo-my-project"
    # Windows 风格路径同样工作
    assert _sanitize_path("C:\\Users\\foo") == "C--Users-foo"
    # 已经是 slug 的字符串保持不变
    assert _sanitize_path("abc123") == "abc123"


def test_sanitize_empty_string():
    assert _sanitize_path("") == ""


def test_sanitize_short_path_no_hash_suffix():
    short = "a" * MAX_SANITIZED_LENGTH  # 恰好 200 字符
    assert _sanitize_path(short) == short
    # 不应包含 hash 分隔
    assert len(_sanitize_path(short)) == MAX_SANITIZED_LENGTH


def test_sanitize_long_path_truncates_with_hash():
    long = "/" + "x" * 300  # 远超 200
    result = _sanitize_path(long)
    # 形式：<200 字符截断>-<8 字符 hex hash>
    assert len(result) == MAX_SANITIZED_LENGTH + 1 + 8
    assert result[MAX_SANITIZED_LENGTH] == "-"
    # 后 8 位是 hex
    assert all(c in "0123456789abcdef" for c in result[-8:])


def test_sanitize_long_path_deterministic():
    """同一输入必须产生同一结果（用于 git_root 稳定映射到同一记忆目录）。"""
    long = "/" + "y" * 400
    assert _sanitize_path(long) == _sanitize_path(long)


def test_sanitize_long_paths_differ_by_hash():
    """长度截断后两个不同输入仍能通过 hash 后缀区分。"""
    a = "/foo/" + "z" * 400
    b = "/bar/" + "z" * 400
    assert _sanitize_path(a) != _sanitize_path(b)


# ---------------------------------------------------------------------------
# get_memory_dir
# ---------------------------------------------------------------------------

def _init_git_repo(path: Path) -> None:
    """在 path 初始化一个最小 git 仓库（只需 init，无需 commit）。"""
    subprocess.run(
        ["git", "init", "-q"],
        cwd=str(path),
        check=True,
        capture_output=True,
        timeout=10,
    )


def _git_available() -> bool:
    """检测当前环境是否能跑 git，跑不了就跳过依赖 git 的测试。"""
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=3)
        return True
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


pytestmark_git = pytest.mark.skipif(not _git_available(), reason="git not available")


@pytestmark_git
def test_get_memory_dir_in_git_repo(tmp_path, monkeypatch):
    """git 仓库 → 返回项目级路径，包含 projects/<sanitized>/memory 结构。"""
    monkeypatch.setattr(memory_mod, "BASE_CONFIG_DIR", tmp_path / "config")
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    result = get_memory_dir(repo)

    # 必须落在 projects/<sanitized>/memory 下，且不等于 GLOBAL_MEMORY_DIR
    assert result.parent.parent == tmp_path / "config" / "projects"
    assert result.name == "memory"
    assert result != memory_mod.GLOBAL_MEMORY_DIR


@pytestmark_git
def test_get_memory_dir_same_repo_subdir(tmp_path, monkeypatch):
    """同一仓库内的子目录必须解析到同一记忆目录（git rev-parse 找到同一 root）。"""
    monkeypatch.setattr(memory_mod, "BASE_CONFIG_DIR", tmp_path / "config")
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)

    assert get_memory_dir(repo) == get_memory_dir(sub)


@pytestmark_git
def test_get_memory_dir_different_repos_differ(tmp_path, monkeypatch):
    """两个不同 git 仓库必须解析到不同记忆目录。"""
    monkeypatch.setattr(memory_mod, "BASE_CONFIG_DIR", tmp_path / "config")
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    _init_git_repo(repo_a)
    _init_git_repo(repo_b)

    assert get_memory_dir(repo_a) != get_memory_dir(repo_b)


def test_get_memory_dir_non_git_falls_back_to_global(tmp_path, monkeypatch):
    """非 git 目录 → 回退到全局目录，保留旧行为。"""
    monkeypatch.setattr(memory_mod, "BASE_CONFIG_DIR", tmp_path / "config")
    # 同时替换 GLOBAL_MEMORY_DIR 以匹配上面 BASE_CONFIG_DIR 的新值；
    # 但 GLOBAL_MEMORY_DIR 是模块加载时计算的常量，运行时不会自动更新，
    # 因此这里直接断言"非 git 时返回的就是模块当前 GLOBAL_MEMORY_DIR"即可。
    non_git = tmp_path / "loose"
    non_git.mkdir()

    assert get_memory_dir(non_git) == memory_mod.GLOBAL_MEMORY_DIR


def test_get_memory_dir_git_missing_falls_back(monkeypatch, tmp_path):
    """git 可执行文件不存在 → 静默回退到全局目录。"""
    monkeypatch.setattr(memory_mod, "BASE_CONFIG_DIR", tmp_path / "config")

    def boom(*a, **kw):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(memory_mod.subprocess, "run", boom)

    # 无论传什么路径，都应回退
    assert get_memory_dir(tmp_path) == memory_mod.GLOBAL_MEMORY_DIR


def test_get_memory_dir_git_timeout_falls_back(monkeypatch, tmp_path):
    """git 超时 → 静默回退（防止主流程被慢/卡住的 git 阻塞）。"""
    monkeypatch.setattr(memory_mod, "BASE_CONFIG_DIR", tmp_path / "config")

    def slow(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="git", timeout=3)

    monkeypatch.setattr(memory_mod.subprocess, "run", slow)
    assert get_memory_dir(tmp_path) == memory_mod.GLOBAL_MEMORY_DIR


def test_get_memory_dir_default_uses_cwd(monkeypatch, tmp_path):
    """cwd=None 应该取 Path.cwd()。"""
    monkeypatch.setattr(memory_mod, "BASE_CONFIG_DIR", tmp_path / "config")
    # 把 cwd 切到非 git 目录，预期回退
    non_git = tmp_path / "loose"
    non_git.mkdir()
    monkeypatch.chdir(non_git)

    assert get_memory_dir() == memory_mod.GLOBAL_MEMORY_DIR


# ---------------------------------------------------------------------------
# 向后兼容
# ---------------------------------------------------------------------------

def test_memory_dir_constant_backward_compat():
    """MEMORY_DIR 旧常量仍存在并等于全局目录，保护任何 `from features.memory import MEMORY_DIR` 的旧代码。"""
    assert MEMORY_DIR == GLOBAL_MEMORY_DIR
