from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Callable
from xml.sax.saxutils import escape

from core.engine import AbortedError, Engine


@dataclass
class WorkerUsage:
    total_tokens: int = 0
    tool_uses: int = 0
    duration_ms: int = 0


@dataclass
class WorkerTask:
    """表示一个后台 worker 任务。"""
    task_id: str
    description: str
    engine: Engine
    status: str = "idle"
    summary: str = ""
    result: str = ""
    usage: WorkerUsage = field(default_factory=WorkerUsage)
    thread: threading.Thread | None = None
    tool_use_count: int = 0       # 已调用工具次数，用于实时状态显示
    current_activity: str = ""    # 当前活动描述，用于实时状态显示


class WorkerManager:
    """管理后台 worker 线程的生命周期：spawn / continue / stop / 通知队列。"""

    def __init__(self, build_worker_engine: Callable[[], Engine]):
        self._build_worker_engine = build_worker_engine
        self._tasks: dict[str, WorkerTask] = {}
        self._lock = threading.Lock()           # 多线程访问 _tasks 需要加锁
        self._notifications: Queue[str] = Queue()  # 线程安全的通知队列

    def spawn(self, *, description: str, prompt: str,
              subagent_type: str = "worker") -> dict[str, str]:
        """启动一个新的 worker 任务，返回 task_id。"""
        if subagent_type != "worker":
            raise ValueError("Only subagent_type='worker' is supported.")

        task = WorkerTask(
            task_id=f"agent-{uuid.uuid4().hex[:8]}",
            description=description.strip() or "Worker task",
            engine=self._build_worker_engine(),  # 每个 worker 独立的 engine 实例
        )
        with self._lock:
            self._tasks[task.task_id] = task
        self._start(task, prompt)
        return {"task_id": task.task_id, "status": "started", "description": task.description}

    def continue_task(self, *, task_id: str, message: str) -> dict[str, str]:
        """继续一个已完成的 worker 任务（SendMessage）。"""
        task = self._get_task(task_id)
        if self._is_running(task):
            raise ValueError("Task is still running. Wait for it to finish before continuing it.")
        self._start(task, message)
        return {"task_id": task.task_id, "status": "started", "description": task.description}

    def stop_task(self, *, task_id: str) -> dict[str, str]:
        """中止一个正在运行的 worker 任务。"""
        task = self._get_task(task_id)
        if not self._is_running(task):
            return {"task_id": task.task_id, "status": task.status or "idle",
                    "description": task.description}
        try:
            task.engine.abort()
        except Exception:
            pass
        return {"task_id": task.task_id, "status": "stopping", "description": task.description}

    def drain_notifications(self) -> list[str]:
        """取出队列中所有已完成任务的通知（非阻塞）。"""
        drained: list[str] = []
        while True:
            try:
                drained.append(self._notifications.get_nowait())
            except Empty:
                return drained

    def has_running_tasks(self) -> bool:
        """是否有正在运行的 worker。"""
        with self._lock:
            return any(self._is_running(t) for t in self._tasks.values())

    def get_running_status(self) -> list[dict]:
        """返回所有正在运行 worker 的实时状态。"""
        with self._lock:
            return [
                {"task_id": t.task_id, "description": t.description,
                 "tool_uses": t.tool_use_count, "activity": t.current_activity}
                for t in self._tasks.values()
                if self._is_running(t)
            ]

    def get_panel_status(self) -> list[dict]:
        """返回进度面板所需的全部 worker 状态（运行中 + 已完成，按 spawn 序）。

        与 get_running_status 的区别：不筛选运行态，附带 status 字段
        （running/completed/killed/failed），供常驻面板显示完成态（✓/✗）。
        线程安全（_lock 保护）；status 的最终写发生在 worker 线程结束处。
        """
        with self._lock:
            return [
                {"task_id": t.task_id, "description": t.description,
                 "tool_uses": t.tool_use_count, "activity": t.current_activity,
                 "status": "running" if self._is_running(t) else t.status}
                for t in self._tasks.values()
            ]

    def _get_task(self, task_id: str) -> WorkerTask:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task id: {task_id}")
        return task

    @staticmethod
    def _is_running(task: WorkerTask) -> bool:
        return task.thread is not None and task.thread.is_alive()

    def _start(self, task: WorkerTask, prompt: str) -> None:
        """在后台线程中启动任务。"""
        task.status = "running"
        task.summary = ""
        task.result = ""
        task.usage = WorkerUsage()
        task.thread = threading.Thread(
            target=self._run_task,          # 线程执行方法
            name=task.task_id,
            args=(task, prompt),
            daemon=True,
        )
        task.thread.start()

    def _run_task(self, task: WorkerTask, prompt: str) -> None:
        """worker 线程主体：消费 engine.submit() 事件流，完成后推送通知。"""
        started = time.monotonic()
        parts: list[str] = []
        total_tokens = 0
        tool_uses = 0
        task.tool_use_count = 0
        task.current_activity = "Initializing…"
        try:
            for event in task.engine.submit(prompt):
                kind = event[0]
                if kind == "text":
                    parts.append(event[1])
                    task.current_activity = "Thinking…"
                elif kind == "tool_call":
                    tool_uses += 1
                    task.tool_use_count = tool_uses
                    tool_name = event[1] if len(event) > 1 else ""
                    task.current_activity = f"Running {tool_name}…"
                elif kind == "tool_result":
                    task.current_activity = "Thinking…"
                elif kind == "error":
                    parts.append(event[1])
            status = "completed"
            summary = f'Agent "{task.description}" completed'
        except AbortedError:
            status = "killed"
            summary = f'Agent "{task.description}" was stopped'
        except Exception as exc:
            status = "failed"
            summary = f'Agent "{task.description}" failed: {exc}'
            parts.append(str(exc))

        task.status = status
        task.summary = summary
        task.current_activity = ""
        task.result = "".join(parts).strip()
        task.usage = WorkerUsage(
            total_tokens=total_tokens,
            tool_uses=tool_uses,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        self._notifications.put(self._render_notification(task))

    def _render_notification(self, task: WorkerTask) -> str:
        """将任务结果序列化为 XML 风格的 <task-notification> 字符串。

            示例：
                <task-notification>
                    <task-id>agent-a1b2c3d4</task-id>
                    <status>completed</status>
                    <summary>Agent &quot;代码重构&quot; completed</summary>
                    <result>已成功将 utils.py 中的函数提取到 helper.py</result>
                    <usage>
                      <total_tokens>1500</total_tokens>
                      <tool_uses>3</tool_uses>
                      <duration_ms>5200</duration_ms>
                    </usage>
                </task-notification>
        """
        parts = [
            "<task-notification>",
            f"<task-id>{escape(task.task_id)}</task-id>",
            f"<status>{escape(task.status)}</status>",
            f"<summary>{escape(task.summary)}</summary>",
        ]
        if task.result:
            parts.append(f"<result>{escape(task.result)}</result>")
        parts.extend([
            "<usage>",
            f"  <total_tokens>{task.usage.total_tokens}</total_tokens>",
            f"  <tool_uses>{task.usage.tool_uses}</tool_uses>",
            f"  <duration_ms>{task.usage.duration_ms}</duration_ms>",
            "</usage>",
            "</task-notification>",
        ])
        return "\n".join(parts)
