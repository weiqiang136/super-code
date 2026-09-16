"""Session persistence — JSONL-based conversation storage.

Each session is a pair of files under ~/.config/super-code/sessions/{sanitized_cwd}/:
  {session_id}.jsonl      — one JSON object per message (append-only)
  {session_id}.meta.json  — lightweight metadata for fast listing
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SESSIONS_ROOT = Path.home() / ".config" / "super-code" / "sessions"

# 记忆注入 / skill 注入会把 <system-reminder>...</system-reminder> 前缀拼进第一条
# user 消息。贪婪匹配到最后一个闭合标签：前缀内部可能嵌套 freshness 的小块
# system-reminder，非贪婪会提前停在内层闭合标签处，残留文本污染标题。
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*</system-reminder>\s*", re.DOTALL)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SessionMeta:
    session_id: str
    title: str
    cwd: str
    model: str
    created_at: str
    updated_at: str
    message_count: int = 0
    mode: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_cwd(cwd: str) -> str:         # 绝对路径转为安全的目录名称
    """Convert an absolute path to a safe directory name."""
    name = re.sub(r"[^a-zA-Z0-9]", "-", cwd) # 使用正则表达式，把所有不是英文大小写字母和数字的字符替换成 -
    name = re.sub(r"-+", "-", name).strip("-")
    if len(name) > 80:
        h = hashlib.sha1(cwd.encode()).hexdigest()[:8]  # 超过80字符，追加8位哈希
        name = name[:80] + "-" + h
    return name


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_local_time(iso_str: str, fmt: str = "%Y-%m-%d %H:%M") -> str:
    # 把存盘的 ISO 时间（带或不带 tz；无 tz 视为 UTC，兼容历史会话）转为系统本地时区显示
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
    except ValueError:
        return iso_str[:len(fmt)]  # 解析失败兜底返回原串前缀，避免崩
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime(fmt)


def _serialize_content(content: Any) -> Any:        # 把复杂的 content 转成 JSON 可以保存的普通数据。递归的处理
    """Recursively convert SDK objects to plain dicts for JSON serialization."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_serialize_content(item) for item in content]
    if hasattr(content, "model_dump"):          # Pydantic BaseModel (Anthropic SDK)
        return content.model_dump()
    if isinstance(content, dict):
        return {k: _serialize_content(v) for k, v in content.items()}
    return content


def _serialize_message(msg: dict) -> dict:              # 返回消息字典的JSON副本，专门处理content字段
    """Return a JSON-safe copy of a message dict."""
    out: dict[str, Any] = {}
    for key, val in msg.items():
        out[key] = _serialize_content(val) if key == "content" else val
    return out


def _extract_text(content: Any) -> str:         # 尽力提取消息内容中的纯文本
    """Best-effort plain text extraction from message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(getattr(block, "text", ""))
        return " ".join(parts)
    return str(content)


def _generate_title(content: Any) -> str:       # 根据第一条用户消息创建会话标题
    """Create a short title from the first user message."""
    text = _extract_text(content).strip()
    # 剥离注入层：标题应取自用户真实输入，而不是记忆检索 / skill 的
    # <system-reminder> 前缀（该前缀由 app.py 拼在 user_input 前面一起落盘）
    text = _SYSTEM_REMINDER_RE.sub("", text, count=1)
    if not text:
        return "(untitled)"
    if len(text) <= 80:
        return text
    truncated = text[:80]
    last_space = truncated.rfind(" ")
    return (truncated[:last_space] if last_space > 40 else truncated) + "…"


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

class SessionStore:
    """Manages JSONL persistence for a single session."""

    def __init__(self, cwd: str, model: str,
                 session_id: str | None = None,
                 mode: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex
        self.cwd = cwd
        self.model = model
        self.mode = mode
        self._dir = _SESSIONS_ROOT / _sanitize_cwd(cwd)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._dir / f"{self.session_id}.jsonl"
        self._meta_path = self._dir / f"{self.session_id}.meta.json"
        self._message_count = 0
        self._title: str = ""
        self._created_at: str = _now_iso()
        # ── Turn-level checkpoint（用于配合 engine.cancel_turn 回滚磁盘） ─────
        # 记录某一轮 submit 开始时的 JSONL 字节位置 + 消息计数。一旦该轮被 abort，
        # rollback_to_checkpoint() 会把文件截回这个位置，确保不会留下孤立 tool_use。
        # 单字段元组而非两个字段：保证 (offset, count) 通过单条 STORE_ATTR 原子写入，
        # 避免 KeyboardInterrupt 在两次赋值之间命中导致半态（offset 已设但 count=None，
        # 触发 rollback 守卫的 None 检查，直接 return，孤立 tool_use 永留磁盘）。
        # None 表示当前没有未完成的 turn checkpoint。
        self._checkpoint: tuple[int, int] | None = None

    # -- writing -----------------------------------------------------------

    def append_message(self, message: dict) -> None:    # 将一条消息持久化到JSONL文件中，同时更新元数据
        """Persist one message (append to JSONL)."""
        safe = _serialize_message(message)
        safe["_ts"] = _now_iso()
        with open(self._jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(safe, ensure_ascii=False) + "\n")
        self._message_count += 1

        # Auto-generate title from first user message
        if not self._title and message.get("role") == "user":
            self._title = _generate_title(message.get("content", ""))

        self._save_meta()

    def _save_meta(self) -> None:       # 保存会话元数据到meta.json中
        meta = SessionMeta(
            session_id=self.session_id,
            title=self._title or "(untitled)",
            cwd=self.cwd,
            model=self.model,
            created_at=self._created_at,
            updated_at=_now_iso(),
            message_count=self._message_count,
            mode=self.mode,
        )
        # 原子替换：直接 open(path, "w") 进入瞬间就清空旧文件，Ctrl+C 落在
        # open 之后、json.dump 之前会留下空 meta.json，导致会话元数据丢失。
        # 改为写临时文件 + os.replace（在 Windows / POSIX 上均为原子操作）。
        tmp = self._meta_path.with_name(self._meta_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(asdict(meta), fh, ensure_ascii=False)
        os.replace(tmp, self._meta_path)

    # -- turn checkpoint / rollback ---------------------------------------

    def mark_checkpoint(self) -> None:
        """记录一个"turn 开始"检查点：当前 JSONL 字节偏移 + 消息数。

        engine.submit() 在 turn 入口调用本方法。若该轮被 Ctrl+C 中断，
        cancel_turn() 调用 rollback_to_checkpoint() 即可把磁盘文件截回这里，
        避免留下孤立的 tool_use（缺对应 tool_result），下次 /resume 才不会
        被 OpenAI 拒收：'Messages with role tool must be a response to a
        preceding message with tool_calls'。

        约定：
          - 文件不存在视为偏移 0；rollback 时 truncate 到 0 等于清空。
          - 重复调用会覆盖上一次 checkpoint（正常 turn 完成后不会 rollback，
            下一轮直接覆盖即可，无需主动清理）。
        """
        try:
            offset = self._jsonl_path.stat().st_size if self._jsonl_path.exists() else 0
        except OSError:
            # 文件系统异常时退化为不记 checkpoint：宁可不回滚也别截错位置
            self._checkpoint = None
            return
        # 单句赋值：BUILD_TUPLE 后 STORE_ATTR 是单条字节码，对 KeyboardInterrupt 原子
        self._checkpoint = (offset, self._message_count)

    def rollback_to_checkpoint(self) -> None:
        """把 JSONL 截回最近一次 mark_checkpoint 记录的字节位置。

        与 engine.cancel_turn() 配套：cancel_turn 删内存切片、本方法删磁盘尾巴，
        两者保证内存和磁盘严格一致。

        无 checkpoint 时是 no-op（防御性：不在 None 状态做截断）。
        操作完会同步重写 meta.json，让 message_count 反映真实状态。
        """
        if self._checkpoint is None:
            return
        offset, count = self._checkpoint
        # 关键顺序：先截磁盘、再清状态。若反过来，KeyboardInterrupt 在两步之间命中时
        # checkpoint 会永久丢失，第二次 Ctrl+C 也无法回滚，孤立 tool_use 永留磁盘。
        # 当前顺序下中途被打断：checkpoint 仍在 → 下次 cancel_turn 重做 truncate，
        # 同一 offset 重复截断是 O(1) 幂等 no-op，安全。
        try:
            if self._jsonl_path.exists():
                # r+b 模式打开 + truncate：O(1) 截断，不读取文件内容
                with open(self._jsonl_path, "r+b") as fh:
                    fh.truncate(offset)
            self._message_count = count
            self._save_meta()
        except OSError:
            # 截断失败时不抛——上层 cancel_turn 已经在 except AbortedError 路径上，
            # 再抛只会把原始 AbortedError 掩盖。下次 /resume 仍可能脏，
            # 但至少不会让 abort 路径自身崩溃。
            # 注意：故意保留 checkpoint 不清，让下次 cancel_turn 有机会重试。
            # 绝不能扩大到 except BaseException：那会吞掉 KeyboardInterrupt。
            return
        # truncate + meta 都成功后才清 checkpoint，单句赋值原子
        self._checkpoint = None

    # -- reading (class methods) -------------------------------------------

    @classmethod
    def load_messages(cls, session_id: str, cwd: str) -> list[dict]:    # 从磁盘中读取指定会话的所有消息
        """Read all messages for session_id from disk."""
        path = _SESSIONS_ROOT / _sanitize_cwd(cwd) / f"{session_id}.jsonl"
        if not path.exists():
            return []
        messages: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                obj.pop("_ts", None)
                messages.append(obj)
        return messages

    @classmethod
    def list_sessions(cls, cwd: str) -> list[SessionMeta]:  # 返回指定CWD下可用的会话列表
        """Return available sessions for cwd, most recent first."""
        d = _SESSIONS_ROOT / _sanitize_cwd(cwd)
        if not d.exists():
            return []
        results: list[SessionMeta] = []
        for meta_file in d.glob("*.meta.json"):
            try:
                with open(meta_file, encoding="utf-8") as fh:
                    results.append(SessionMeta(**json.load(fh)))
            except Exception:
                continue
        results.sort(key=lambda m: m.updated_at, reverse=True)
        return results

    @classmethod
    def load_session(cls, session_id: str, cwd: str) -> tuple[SessionMeta | None, list[dict]]:    #加载指定会话的元数据和消息内容。
        """Load metadata + messages for session_id."""
        d = _SESSIONS_ROOT / _sanitize_cwd(cwd)
        meta_path = d / f"{session_id}.meta.json"
        meta = None
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as fh:
                meta = SessionMeta(**json.load(fh))
        return meta, cls.load_messages(session_id, cwd)
