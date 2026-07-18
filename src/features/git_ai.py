"""Git AI 集成 — 在 AI 编辑文件前后调用 git ai checkpoint，
用于统计 AI 代码占比（git-ai status）。

工作原理：
  编辑前：checkpoint agent-v1 type=human  → 把上次 AI 写入到现在的人工改动标记为 human
  编辑后：checkpoint agent-v1 type=ai_agent → 把本次 AI 编辑标记为 AI

只对写操作工具（Edit、Write、Bash）触发，读操作不触发。
git-ai 未安装时静默跳过，不影响正常使用。

⚠️ git-ai daemon 要求：checkpoint 数据只存在 daemon 内存中，不持久化。
  如果 daemon 在 checkpoint 和 git commit 之间重启（电脑重启、进程崩溃等），
  commit 将丢失归属数据（显示 untracked 100%）。因此 super-code 在启动时
  和每次 checkpoint 前都会主动调用 git-ai bg start 确保 daemon 在运行。

调试：设置环境变量 SUPER_CODE_DEBUG_GIT_AI=1 可将错误信息打印到 stderr。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)

# git-ai 可执行文件路径，优先用 PATH 查找，找不到则用已知安装路径
_GIT_AI_EXE: str | None = None
_GIT_AI_CHECKED = False

# 缓存的 git root（本进程生命周期内不变），用于确保 repo_working_dir 和
# commit hook 看到的路径一致。Path.cwd() 可能是子目录，必须解析为 git root。
_GIT_ROOT: str | None = None

# daemon 是否已在本 session 中启动过（避免每次 checkpoint 都调 bg start）
_DAEMON_ENSURED = False

# 只对这些工具触发 checkpoint
WRITE_TOOLS = {"Edit", "Write", "Bash"}

# 设置 SUPER_CODE_DEBUG_GIT_AI=1 可启用 stderr 诊断输出
_DEBUG = os.environ.get("SUPER_CODE_DEBUG_GIT_AI") == "1"


def _get_exe() -> str | None:
    """返回 git-ai 可执行文件路径，找不到返回 None。结果缓存，只检测一次。"""
    global _GIT_AI_EXE, _GIT_AI_CHECKED
    if _GIT_AI_CHECKED:
        return _GIT_AI_EXE
    _GIT_AI_CHECKED = True
    # 先从 PATH 查找
    found = shutil.which("git-ai")
    if not found:
        # fallback：已知 Windows 安装路径
        fallback = os.path.expanduser(r"~\.git-ai\bin\git-ai.exe")
        if os.path.isfile(fallback):
            found = fallback
    if found:
        _log.debug("git-ai found at: %s", found)
    else:
        _log.debug("git-ai not found – checkpoints disabled")
    _GIT_AI_EXE = found
    return _GIT_AI_EXE


def _git_root(cwd_hint: str) -> str | None:
    """通过 git rev-parse --show-toplevel 获取 git 根目录。

    repo_working_dir 必须等于 git root，否则 commit hook 无法匹配 checkpoint 数据
    导致 git-ai stats 显示 untracked 100%。结果缓存，进程生命周期内只查询一次。
    """
    global _GIT_ROOT
    if _GIT_ROOT is not None:
        return _GIT_ROOT
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
            cwd=cwd_hint or None,
        )
        if r.returncode == 0 and r.stdout.strip():
            _GIT_ROOT = r.stdout.strip()
            return _GIT_ROOT
    except Exception:
        pass
    # 回退：git 不可用时用传入的目录
    _GIT_ROOT = cwd_hint
    return _GIT_ROOT


def _rel_path(file_path: str, git_root: str) -> str:
    """将 file_path 转为相对 git root 的路径，与 git diff 输出格式一致。

    Windows 上 os.path.abspath 保留输入大小写，而 git rev-parse 返回实际大小写，
    必须用 normcase 做大小写不敏感比较，否则 LLM 传小写盘符会导致 startswith 失败。
    """
    try:
        abs_path = os.path.abspath(file_path)
        root = os.path.abspath(git_root)
        if os.path.normcase(abs_path).startswith(os.path.normcase(root)):
            return abs_path[len(root):].lstrip(os.sep).replace("\\", "/")
    except Exception:
        pass
    return file_path.replace("\\", "/")


def _ensure_daemon(exe: str, cwd: str | None = None) -> None:
    """确保 git-ai 后台 daemon 在运行。

    git-ai 的 checkpoint 数据只存在 daemon 内存中，daemon 重启后数据丢失。
    如果 daemon 没运行，checkpoint 命令会自动启动一个，但在某些场景下
    （如电脑刚重启后第一次调用）可能启动不够及时导致 checkpoint 丢失。
    主动调用 bg start 可以确保 daemon 提前就绪。

    每次调用会检查 daemon 状态，如果已在运行则 no-op。
    """
    global _DAEMON_ENSURED
    if _DAEMON_ENSURED:
        return
    try:
        subprocess.run(
            [exe, "bg", "start"],
            capture_output=True, timeout=5,
            cwd=cwd,
        )
        _DAEMON_ENSURED = True
    except Exception:
        pass


def ensure_daemon() -> None:
    """供外部调用（如 app.py 启动时）主动启动 daemon。"""
    exe = _get_exe()
    if not exe:
        return
    _ensure_daemon(exe)


def _run(payload: dict, exe: str, cwd: str | None = None) -> None:
    """把 payload 序列化为 JSON 通过 stdin 传给 git-ai checkpoint agent-v1。

    失败时记录日志，绝不抛出异常影响主流程。
    设置 SUPER_CODE_DEBUG_GIT_AI=1 可将错误打印到 stderr。
    """
    try:
        data = json.dumps(payload, ensure_ascii=False)
        r = subprocess.run(
            [exe, "ai", "checkpoint", "agent-v1", "--hook-input", "stdin"],
            input=data.encode("utf-8"),
            capture_output=True,
            timeout=10,
            cwd=cwd,
        )
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace").strip()
            _log.debug("git-ai checkpoint agent-v1 rc=%d stderr=%s", r.returncode, err)
            if _DEBUG:
                import sys
                print(f"[git-ai] checkpoint failed (rc={r.returncode}): {err}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        _log.debug("git-ai checkpoint agent-v1 timed out after 10s")
        if _DEBUG:
            import sys
            print("[git-ai] checkpoint timed out after 10s", file=sys.stderr)
    except Exception:
        _log.debug("git-ai checkpoint agent-v1 error", exc_info=True)
        if _DEBUG:
            import sys, traceback
            traceback.print_exc(file=sys.stderr)


def before_edit(repo_dir: str, file_path: str) -> None:
    """编辑文件前调用：把上次 AI 写入到现在的人工改动标记为 human。"""
    exe = _get_exe()
    if not exe:
        return
    root = _git_root(repo_dir)
    if not root:
        return
    _ensure_daemon(exe, root)
    _run({
        "type": "human",
        "repo_working_dir": root.replace("\\", "/"),
        # 告知 git-ai 只 diff 这个文件，速度提升 50-100x
        "will_edit_filepaths": [_rel_path(file_path, root)] if file_path else [],
    }, exe, root)


def after_edit(repo_dir: str, file_path: str,
               messages: list[dict], model: str, session_id: str) -> None:
    """编辑文件后调用：把本次 AI 编辑标记为 AI。

    messages: engine._messages，会自动过滤掉 tool_result（git-ai 不接受）。
    """
    exe = _get_exe()
    if not exe:
        return
    root = _git_root(repo_dir)
    if not root:
        return
    _ensure_daemon(exe, root)
    transcript = _build_transcript(messages)
    _run({
        "type": "ai_agent",
        "repo_working_dir": root.replace("\\", "/"),
        "transcript": transcript,
        "agent_name": "super-code",
        "model": model,
        "conversation_id": session_id,
        "edited_filepaths": [_rel_path(file_path, root)] if file_path else [],
    }, exe, root)


def _build_transcript(messages: list[dict]) -> dict[str, Any]:
    """把 engine 消息历史转换为 git-ai 要求的 transcript 格式。

    规则：
    - role=user 且 content 是 list（tool_result）→ 跳过（git-ai 不接受）
    - role=user 且 content 是 str → type=user
    - role=assistant → type=assistant（取文本部分）
    - content 里的 tool_use block → type=tool_use
    """
    out: list[dict[str, Any]] = []
    ts = datetime.now(timezone.utc).isoformat()

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, list):
                # tool_result 消息，跳过
                continue
            out.append({"type": "user", "text": str(content), "timestamp": ts})

        elif role == "assistant":
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        out.append({"type": "assistant", "text": block.get("text", ""), "timestamp": ts})
                    elif block.get("type") == "tool_use":
                        out.append({
                            "type": "tool_use",
                            "name": block.get("name", ""),
                            "input": block.get("input", {}),
                            "timestamp": ts,
                        })
            else:
                out.append({"type": "assistant", "text": str(content or ""), "timestamp": ts})

    return {"messages": out}
