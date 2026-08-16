from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Any

from core.llm import LLMClient
from core.tool import Tool, ToolResult
from core.permissions import PermissionChecker
from features.compact import estimate_tokens, get_context_window


# Windows 终端粘贴 UTF-16 剪贴板时可能把代理对当成两个独立码点喂进 stdin，
# 后续 json.dumps(..., ensure_ascii=False) 写 UTF-8 JSONL 会抛
# UnicodeEncodeError: 'utf-8' codec can't encode ... : surrogates not allowed。
# 在 Engine.submit 入口统一替换为 U+FFFD，保证 _messages、磁盘、LLM 请求三处一致。
_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

# 工具被拒文案。关键词 "STOP what you are doing and wait for the user" 让模型读完该 tool_result
# 后自然结束本轮、不再换其它工具继续骚扰用户——配合 deny → tool_result 路径
# 替代旧的 raise AbortedError + cancel_turn（旧行为会把整轮历史包括用户输入一起截掉）。
_REJECT_MESSAGE = (
    "The user doesn't want to proceed with this tool use. The tool use was rejected "
    "(eg. if it was a file edit, the new_string was NOT written to the file). "
    "STOP what you are doing and wait for the user to tell you how to proceed."
)
# 同一批 tool_use 里某个被拒后，其余未处理 tool_use 走该文案——避免连续弹多次确认，
# 同时保证 tool_use ↔ tool_result 一一配对（不配对下一轮 LLM 调用会 400）。
_SIBLING_REJECT_MESSAGE = (
    "Tool execution skipped because the user rejected an earlier tool call in this batch. "
    "STOP what you are doing and wait for the user to tell you how to proceed."
)


# 轮内压缩触发比例：当估算 token 数达到 context window 的此比例时，在 while 循环
# 内紧急压缩历史消息。0.9 留 10% 余量给压缩后的 compact prompt。
_INTRA_TURN_COMPACT_TRIGGER_RATIO = 0.9


class AbortedError(Exception):
    """Raised when the current turn is aborted by the user (Esc / Ctrl+C)."""


class Engine:
    def __init__(self, tools: list[Tool], system_prompt: str,
                 permission_checker: PermissionChecker,
                 provider: str = "openai",
                 model: str = "gpt-4o",
                 max_tokens: int | None = None,
                 api_key: str | None = None,
                 base_url: str | None = None,
                 effort: str | None = None,
                 session_store=None,
                 cost_tracker=None,
                 repo_dir: str = "",
                 agent_session_id: str = "",
                 timeout: float = 300.0,
                 model_profiles: dict | None = None):
        self._model = model
        self._max_tokens = max_tokens or 131072
        self._model_profiles = model_profiles or {}
        self._client = LLMClient(provider=provider, api_key=api_key, base_url=base_url,
                                 timeout=timeout, model_profiles=self._model_profiles)
        self._tools = {t.name: t for t in tools}
        self._system_prompt = system_prompt
        self._permissions = permission_checker
        self._messages: list[dict] = []
        self._aborted = False
        self._turn_start_len: int | None = None
        self._active_stream = None
        self._session_store = session_store
        self._cost_tracker = cost_tracker  # 费用追踪器，记录每次 API 调用的 token 用量
        # git-ai 钩子参数：worker Engine 不挂 session_store，需要显式传；主 Engine 不传则回退 session_store
        self._repo_dir_override = repo_dir
        self._agent_session_id_override = agent_session_id
        # 一次性回调列表：在每轮 tool_results append 到 _messages 之后触发并清空。
        # 用于 plan_manager.exit() 延迟执行历史清理（避免在工具执行中途清理导致时序问题）。
        self._post_tool_hooks: list = []  # 存储的是callable对象
        # 轮内压缩服务：由 app.py 注入，用于在工具调用链中紧急压缩历史
        self._compact_service = None
        # worker 通知回调：由 app.py 注入，每轮工具执行完成后 drain 通知队列
        self._on_after_tools = None
        # Phase 3: 向 Edit/Read/Write 工具注入当前会话 ID
        self._inject_session_id()

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def last_assistant_text(self) -> str:
        """返回最后一条 assistant 消息的纯文本内容，用于提取 <system_reminder> 标签。"""
        for msg in reversed(self._messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            return str(content) if content else ""
        return ""

    def set_messages(self, messages: list[dict]) -> None:
        self._messages = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
        ]

    def set_session_store(self, session_store) -> None:
        self._session_store = session_store
        self._inject_session_id()

    def set_compact_service(self, compact_service) -> None:
        """注入 CompactService，供轮内紧急压缩使用。"""
        self._compact_service = compact_service

    def set_on_after_tools(self, callback) -> None:
        """注入 worker 通知回调：每轮工具执行完成后调用，返回通知文本注入 _messages。"""
        self._on_after_tools = callback

    def _inject_session_id(self) -> None:
        """Phase 3: 向支持 set_session_id 的工具注入当前会话 ID。"""
        sid = ""
        if self._session_store is not None:
            sid = getattr(self._session_store, "session_id", "")
        if not sid:
            sid = self._agent_session_id_override or ""
        for t in self._tools.values():
            injector = getattr(t, "set_session_id", None)
            if injector is not None:
                injector(sid)

    def rebuild_snippets_from_messages(self) -> int:
        """Phase 3: 从当前 _messages 中扫描 tool_result metadata，重建 snippet 注册表。

        用于 /resume 恢复会话时还原文件状态和 snippet 缓存。
        返回重建的 snippet 数量。
        """
        from core.file_state import record_file_state, rebuild_snippet
        count = 0
        session_id = ""
        if self._session_store is not None:
            session_id = getattr(self._session_store, "session_id", "")
        if not session_id:
            session_id = self._agent_session_id_override or ""
        if not session_id:
            return 0

        for msg in self._messages:
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue
                meta = block.get("metadata")
                if not isinstance(meta, dict):
                    continue

                snippet_id = meta.get("snippet_id") or meta.get("new_snippet_id")
                if not snippet_id:
                    continue

                fp = meta.get("file_path", "")
                sl = meta.get("start_line")
                el = meta.get("end_line")
                st = meta.get("scope_type", "full")
                if not fp or sl is None or el is None:
                    continue

                # record_file_state 从当前磁盘重建（如果文件存在）
                from pathlib import Path as _Path
                p = _Path(fp)
                if p.exists() and p.is_file():
                    try:
                        stat = p.stat()
                        content_text = p.read_text(encoding="utf-8", errors="replace")
                        record_file_state(session_id, fp, content_text, stat.st_mtime)
                    except Exception:
                        pass

                rebuild_snippet(session_id, snippet_id, fp, int(sl), int(el), str(st))
                count += 1

        return count

    def set_tools(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value or ""

    def abort(self):
        self._aborted = True
        if self._active_stream is not None:
            try:
                self._active_stream.close()
            except Exception:
                pass

    def cancel_turn(self):
        if self._turn_start_len is not None:
            del self._messages[self._turn_start_len:]
            self._turn_start_len = None
        # 同步把磁盘 JSONL 截回 turn 开始时记录的 checkpoint：避免被 Ctrl+C 中断的轮次
        # 在磁盘留下孤立 tool_use（缺对应 tool_result），导致下次 /resume 报
        # 'Messages with role tool must be a response to a preceding message with tool_calls'
        if self._session_store:
            self._session_store.rollback_to_checkpoint()

    def _intra_turn_compact(self) -> bool:
        """压缩 _turn_start_len 之前的历史消息，保留本轮消息原封不动。

        仅在 _compact_service 已注入、且有足够历史消息时才执行压缩。
        成功后更新 _turn_start_len 指向新 messages 中本轮开始的位置，
        确保 cancel_turn() 仍能正确截断。

        Returns:
            True 如果压缩成功执行，False 如果跳过（无压缩服务/历史不足/压缩失败）。
        """
        if self._compact_service is None:
            return False
        if self._turn_start_len is None or self._turn_start_len <= 0:
            return False

        history = self._messages[:self._turn_start_len]
        current_turn = self._messages[self._turn_start_len:]

        # 历史消息太少，不值得压缩
        if len(history) < 10:
            return False

        try:
            new_history, _summary = self._compact_service.compact(
                messages=history,
                system_prompt=self._system_prompt,
                # 轮内紧急压缩传 True：剪枝后若已低于自动触发阈值（0.8×窗口），
                # 跳过 LLM 摘要直接返回剪枝结果——轮内场景多为"单轮读大文件"导致，
                # 剪枝往往已经够用，无需再付一次摘要调用。
                skip_if_under_threshold=True,
            )
            self._messages = new_history + current_turn
            # 更新 _turn_start_len，保证 cancel_turn() 截断到正确位置
            self._turn_start_len = len(new_history)
            return True
        except Exception:
            # 压缩失败时静默返回 False，让本轮继续——下一次 LLM 调用
            # 可能因超 context window 失败，但至少不因压缩异常而中断用户操作。
            return False

    def submit(self, user_input: str | list) -> Iterator[tuple]:
        # 清洗 lone surrogate（仅 str 路径，list 路径由内部构造不会含非法码点）
        if isinstance(user_input, str):
            user_input = _LONE_SURROGATE_RE.sub("�", user_input)
        self._aborted = False
        self._turn_start_len = len(self._messages)
        # 记录本轮 JSONL 的字节位置作为 checkpoint；和 _turn_start_len 配对：
        # 一个守内存、一个守磁盘。本调用必须在 user_msg 持久化之前，否则截不掉 user_msg。
        if self._session_store:
            self._session_store.mark_checkpoint()
        user_msg = {"role": "user", "content": user_input}
        self._messages.append(user_msg)
        if self._session_store:
            self._session_store.append_message(user_msg)

        try:
            while True:
                if self._aborted:
                    raise AbortedError()

                # ── 轮内令牌守卫：消息量接近窗口上限时紧急压缩历史 ──
                # 弥补轮间 compact 无法覆盖「单轮内连续读大文件导致消息暴涨」的盲区。
                # 阈值设 0.9（而非 1.0），给压缩后的 compact prompt 留余量。
                estimated = estimate_tokens(self._messages)
                threshold = int(get_context_window(self._model) * _INTRA_TURN_COMPACT_TRIGGER_RATIO)
                if estimated > threshold:
                    yield ("compact",)
                    self._intra_turn_compact()

                tool_uses = []
                tools_schema = [t.to_api_schema() for t in self._tools.values()] if self._tools else None

                with self._client.stream(
                    model=self._model,
                    system_prompt=self._system_prompt,
                    messages=self._messages,
                    tools=tools_schema,
                    max_tokens=self._max_tokens,
                ) as stream:
                    self._active_stream = stream
                    got_text = False
                    waiting_sent = False
                    # Esc 路径：abort() 在子线程关 stream → 主线程 for 循环里抛
                    # httpx.RemoteProtocolError 等网络异常。仅当 _aborted=True 时
                    # 翻译为 AbortedError，让 query.py 走干净的取消路径；
                    # 非 abort 情况下的真实网络故障保持原样抛出。
                    try:
                        for text in stream:
                            if self._aborted:
                                raise AbortedError()
                            if text.startswith("\x00thinking\x00"):
                                yield ("thinking",)
                                continue
                            if text.startswith("\x00toolgen\x00"):
                                if got_text and not waiting_sent:
                                    yield ("waiting",)
                                    waiting_sent = True
                                continue
                            got_text = True
                            yield ("text", text)
                    except AbortedError:
                        raise
                    except Exception:
                        if self._aborted:
                            raise AbortedError()
                        raise

                    if self._aborted:
                        raise AbortedError()
                    if got_text and not waiting_sent:
                        yield ("waiting",)

                    final = stream.final()
                    # 记录本次 API 调用的 token 用量
                    if self._cost_tracker and final.usage:
                        self._cost_tracker.add_usage(self._model, final.usage)
                    if final.content and isinstance(final.content, list):
                        for block in final.content:
                            if _block_type(block) == "tool_use":
                                tool_uses.append(block)

                self._active_stream = None
                asst_msg: dict = {"role": "assistant", "content": final.content}
                # 保留 reasoning_content，DeepSeek 等思考模型要求下一轮原样带回
                if final.reasoning_content:
                    asst_msg["reasoning_content"] = final.reasoning_content
                self._messages.append(asst_msg)
                if self._session_store:
                    self._session_store.append_message(asst_msg)

                if not tool_uses:
                    break

                tool_results = []
                # 本轮是否发生过 deny。任一 batch 出现 deny 即置 True，本轮 LLM 回复
                # 完成后立即 break 外层 while——硬切断"模型不听 REJECT_MESSAGE 里 STOP
                # 指令、继续换工具骚扰用户"的路径（DeepSeek 等指令服从度较弱的模型实测
                # 会这样）。模型仍然能在被拒的下一轮回一句自然语言（用户看到"好的已取消"），
                # 但永远没机会再调任何工具。
                turn_had_deny = False
                batches: list[tuple[bool, list[Any]]] = []
                for tu in tool_uses:
                    t = self._tools.get(_block_name(tu))
                    is_concurrent = t is not None and t.is_read_only()
                    if batches and batches[-1][0] == is_concurrent and is_concurrent:
                        batches[-1][1].append(tu)
                    else:
                        batches.append((is_concurrent, [tu]))

                for is_concurrent, batch in batches:            # 依次处理batches里面的每个元素
                    if self._aborted:
                        raise AbortedError()
                    # 同 turn 内之前的 batch 已经出现 deny → 后续所有 batch 全部 sibling reject
                    # 处理，跳过权限确认。原因：串行工具按 batch 合并规则每个 tool_use 占独立
                    # batch（is_concurrent=False 不合并），sibling_rejected 标志只在 batch 内
                    # 有效，跨 batch 失效——必须在外层用 turn_had_deny 兜底，否则模型一次
                    # 生成 [Bash1, Bash2, Bash3]、用户对 Bash2 点 No 后，Bash3 还会再弹一次。
                    if turn_had_deny:
                        for tu in batch:
                            tid, tn, ti = _block_id(tu), _block_name(tu), _block_input(tu)
                            tool = self._tools.get(tn)
                            act = tool.get_activity_description(**ti) if tool else None
                            yield ("tool_call", tn, ti, act, tid)
                            result = ToolResult(_SIBLING_REJECT_MESSAGE, is_error=True)
                            yield ("tool_result", tn, ti, result, tid)
                            tool_results.append({"type": "tool_result", "tool_use_id": tid,
                                                 "content": result.content, "is_error": result.is_error,
                                                 "metadata": result.metadata})
                        continue

                    if is_concurrent and len(batch) > 1:
                        approved = []
                        denied_results: dict[str, ToolResult] = {}
                        # 一旦本批出现 deny，剩余 tool_use 全部直接标记 sibling reject，
                        # 不再弹权限确认——既保证 tool_use ↔ tool_result 配对（不配对 LLM 400），
                        # 又避免用户连续被弹多次"是否允许"对话框。
                        sibling_rejected = False
                        for tu in batch:
                            tid, tn, ti = _block_id(tu), _block_name(tu), _block_input(tu)
                            tool = self._tools.get(tn)
                            act = tool.get_activity_description(**ti) if tool else None
                            # 事件 tuple 第5位加入 tool_use_id，供 TUI 用唯一 id 追踪工具状态，
                            # 避免同名工具（如两个 Grep）因 key 碰撞导致 pending_tools 无法清空
                            yield ("tool_call", tn, ti, act, tid)
                            if sibling_rejected:
                                denied_results[tid] = ToolResult(_SIBLING_REJECT_MESSAGE, is_error=True)
                                continue
                            if tool and self._permissions.check(tool, ti) == "deny":
                                # 旧行为是 raise AbortedError() → cancel_turn() 截掉整轮历史
                                # （含用户原始输入），UX 上像"系统失忆"。改为构造 is_error 的
                                # tool_result 让 turn 自然走完一轮：模型读到 REJECT_MESSAGE
                                # 里的 "STOP and wait for the user" 会自然结束本轮。
                                denied_results[tid] = ToolResult(_REJECT_MESSAGE, is_error=True)
                                sibling_rejected = True
                                turn_had_deny = True
                            else:
                                approved.append((tu, tool, act))

                        executed_results: dict[str, ToolResult] = {}
                        if approved:
                            for tu, tool, act in approved:
                                yield ("tool_executing", _block_name(tu), _block_input(tu), act, _block_id(tu))
                            with ThreadPoolExecutor(max_workers=min(len(approved), 10)) as pool:
                                futures = {pool.submit(self._execute_tool, tu): tu for tu, _, _ in approved}
                                for f in as_completed(futures):
                                    tu = futures[f]
                                    try:
                                        executed_results[_block_id(tu)] = f.result()
                                    except Exception as exc:
                                        executed_results[_block_id(tu)] = ToolResult(f"Tool execution error: {exc}", is_error=True)

                        for tu in batch:
                            tid, tn, ti = _block_id(tu), _block_name(tu), _block_input(tu)
                            result = denied_results.get(tid) or executed_results.get(tid) or ToolResult("No result", is_error=True)
                            yield ("tool_result", tn, ti, result, tid)
                            tool_results.append({"type": "tool_result", "tool_use_id": tid,
                                                 "content": result.content, "is_error": result.is_error,
                                                 "metadata": result.metadata})
                    else:
                        # 串行批次：用 sibling_rejected 标志跟并发批次同语义——一旦本批
                        # 出现 deny，剩下 tool_use 全部生成 sibling reject 占位 tool_result，
                        # 保证 tool_use ↔ tool_result 配对，同时不再弹后续确认。
                        sibling_rejected = False
                        for tu in batch:
                            if self._aborted:
                                raise AbortedError()
                            tid, tn, ti = _block_id(tu), _block_name(tu), _block_input(tu)
                            tool = self._tools.get(tn)
                            act = tool.get_activity_description(**ti) if tool else None
                            yield ("tool_call", tn, ti, act, tid)
                            if sibling_rejected:
                                result = ToolResult(_SIBLING_REJECT_MESSAGE, is_error=True)
                            elif tool and self._permissions.check(tool, ti) == "deny":
                                # 见并发分支同位置注释：把 raise AbortedError 替换成构造
                                # is_error tool_result，让 turn 自然走完一轮、保留历史。
                                result = ToolResult(_REJECT_MESSAGE, is_error=True)
                                sibling_rejected = True
                                turn_had_deny = True
                            else:
                                yield ("tool_executing", tn, ti, act, tid)
                                result = self._execute_tool(tu)
                            yield ("tool_result", tn, ti, result, tid)
                            tool_results.append({"type": "tool_result", "tool_use_id": tid,
                                                 "content": result.content, "is_error": result.is_error,
                                                 "metadata": result.metadata})

                # 防御性配对兜底：上面所有分支都应保证 tool_uses ↔ tool_results 一一对齐，
                # 但任何后续 refactor 漏一条分支就会让下一轮 LLM 调用收到
                # "tool_use 缺对应 tool_result" 的 400 错误、整个会话死锁（CC 也踩过同类坑，
                # 见 utils/messages.ts:ensureToolResultPairing 注释 CC-1212）。这里做一次
                # 廉价兜底：发现缺失就补占位 tool_result，让会话存活；正常路径走不到。
                _expected_ids = {_block_id(_tu) for _tu in tool_uses}
                _actual_ids = {_tr["tool_use_id"] for _tr in tool_results}
                for _mid in _expected_ids - _actual_ids:
                    tool_results.append({"type": "tool_result", "tool_use_id": _mid,
                                         "content": "Tool execution skipped (internal sibling cancellation).",
                                         "is_error": True, "metadata": None})

                self._messages.append({"role": "user", "content": tool_results})
                if self._session_store:
                    self._session_store.append_message({"role": "user", "content": tool_results})
                # 触发一次性 post_tool_hooks（如 plan_manager 的延迟历史清理）
                # 必须在 tool_results append 之后执行，确保清理逻辑能看到完整的本轮记录
                if self._post_tool_hooks:
                    for hook in self._post_tool_hooks:
                        hook()
                    self._post_tool_hooks.clear()

                # ── Mid-turn worker 通知注入 ──
                # 在工具执行完成后、下一轮 LLM 调用前，检查 worker 完成通知。
                # 有通知则作为 user message 注入 _messages，while 循环自然继续，
                # 下一轮 LLM 调用自动看到通知内容。不需要递归 run_query。
                if self._on_after_tools is not None:
                    injected = self._on_after_tools()
                    if injected:
                        self._messages.append({"role": "user", "content": injected})
                        if self._session_store:
                            self._session_store.append_message(
                                {"role": "user", "content": injected})
                        yield ("notification", injected)

                # 硬约束：本轮发生过 deny → 立即 break 出 while，不再走下一轮 LLM 调用。
                # 专治 DeepSeek 等模型不听 REJECT_MESSAGE 里 STOP 指令、继续换工具骚扰
                # 用户的情况。tool_result 已 append 进 _messages，下一次用户输入时模型
                # 自然会看到本轮被拒的上下文。用户拒绝的视觉反馈靠 ✗ 红字行已足够，
                # 不需要再让模型生成"好的已取消"——后者反而占 token + 让 DeepSeek 有
                # 机会调更多工具。yield 一个事件让 TUI 打印一行"已取消"作为收尾提示，
                # 避免用户看到一堆 ✗ 后突然回到输入框的疑惑感。
                if turn_had_deny:
                    yield ("turn_aborted_by_deny",)
                    break
        except AbortedError:
            self.cancel_turn()
            raise
        finally:
            self._active_stream = None

    def _execute_tool(self, tool_use) -> ToolResult:
        tool = self._tools.get(_block_name(tool_use))
        if tool is None:
            return ToolResult(f"Unknown tool: {_block_name(tool_use)}", is_error=True)

        tool_name = _block_name(tool_use)
        tool_input = _block_input(tool_use)

        # 只对写操作工具触发 git-ai checkpoint，读操作跳过
        from features.git_ai import WRITE_TOOLS, before_edit, after_edit
        is_write = tool_name in WRITE_TOOLS
        file_path = tool_input.get("file_path", "") if is_write else ""
        repo_dir = self._repo_dir_override or (
            self._session_store.cwd if self._session_store else "")

        # Edit 的 file_path 是可选字段，LLM 通常只传 snippet_id 不传 file_path。
        # 若缺失则通过 snippet_id 查文件状态拿到真实路径，否则 checkpoint 会发
        # edited_filepaths=[] 导致 git-ai 无法归属本次修改。
        if is_write and not file_path and tool_name == "Edit":
            _snippet_id = tool_input.get("snippet_id", "")
            if _snippet_id:
                from core.file_state import get_snippet
                _sess = self._agent_session_id_override or (
                    self._session_store.session_id if self._session_store else "unknown")
                _snip = get_snippet(_sess, _snippet_id)
                if _snip:
                    file_path = _snip.file_path

        if is_write and repo_dir:
            before_edit(repo_dir, file_path)

        try:
            result = tool.execute(**tool_input)
        except Exception as e:
            return ToolResult(f"Tool error: {e}", is_error=True)

        if is_write and repo_dir and not result.is_error:
            # 兜底：Edit 成功时 metadata 里也带有 file_path
            resolved_path = file_path or (result.metadata or {}).get("file_path", "")
            session_id = self._agent_session_id_override or (
                self._session_store.session_id if self._session_store else "unknown")
            after_edit(repo_dir, resolved_path, self._messages, self._model, session_id)

        return result


def _block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_name(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("name", ""))
    return str(getattr(block, "name", ""))


def _block_id(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("id", ""))
    return str(getattr(block, "id", ""))


def _block_input(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        value = block.get("input", {})
    else:
        value = getattr(block, "input", {})
    return value if isinstance(value, dict) else {}
