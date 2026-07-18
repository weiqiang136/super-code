"""Background thread that listens for the Escape key (Windows + Unix)."""
from __future__ import annotations

import threading
from typing import Callable

try:
    import os, signal, select
    import termios, tty
    _HAS_TERMIOS = True             # Unix/ Mac /Linux
except ImportError:
    _HAS_TERMIOS = False            # Windows


if _HAS_TERMIOS:
    class EscListener:
        def __init__(self, on_cancel: Callable[[], None] | None = None):
            self.pressed = False
            self._on_cancel = on_cancel
            self._stop = threading.Event()
            self._paused = threading.Event()
            self._thread: threading.Thread | None = None
            self._tty_fd: int | None = None
            self._old_settings = None

        def __enter__(self):
            self.pressed = False
            self._stop.clear()
            self._paused.clear()
            try:
                self._tty_fd = os.open("/dev/tty", os.O_RDONLY | os.O_NOCTTY)
            except OSError:
                import sys
                self._tty_fd = sys.stdin.fileno()
            try:
                self._old_settings = termios.tcgetattr(self._tty_fd)
                tty.setcbreak(self._tty_fd)
            except termios.error:
                self._old_settings = None
            self._thread = threading.Thread(target=self._listen, daemon=True)
            self._thread.start()
            return self

        def __exit__(self, *_exc):
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=0.5)
            if self._old_settings is not None and self._tty_fd is not None:
                try:
                    termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, self._old_settings)
                except termios.error:
                    pass
            if self._tty_fd is not None and self._tty_fd > 2:
                try:
                    os.close(self._tty_fd)
                except OSError:
                    pass
            self._tty_fd = None

        def pause(self):
            self._paused.set()

        def resume(self):
            self._paused.clear()

        def _has_data(self, timeout: float) -> bool:
            if self._tty_fd is None:
                return False
            try:
                return bool(select.select([self._tty_fd], [], [], timeout)[0])
            except (OSError, ValueError):
                return False

        def _listen(self):
            while not self._stop.is_set():
                if self._paused.is_set():
                    self._stop.wait(0.05)
                    continue
                if not self._has_data(0.1):
                    continue
                if self._paused.is_set():
                    continue
                try:
                    b = os.read(self._tty_fd, 1)
                except OSError:
                    break
                if not b:
                    break
                if b == b'\x1b':
                    if self._has_data(0.05):
                        continue
                    self.pressed = True
                    os.kill(os.getpid(), signal.SIGINT)
                    return

else:
    import msvcrt   #  Windows 专用的控制台 I/O 模块

    class EscListener:  # type: ignore[no-redef]
        def __init__(self, on_cancel: Callable[[], None] | None = None):
            self.pressed = False
            self._on_cancel = on_cancel
            self._stop = threading.Event()
            self._paused = threading.Event()
            self._thread: threading.Thread | None = None

        def __enter__(self):
            self.pressed = False
            self._stop.clear()
            self._paused.clear()
            self._thread = threading.Thread(target=self._listen, daemon=True)
            self._thread.start()
            return self

        def __exit__(self, *_exc):
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=0.5)

        def pause(self):
            self._paused.set()

        def resume(self):
            self._paused.clear()

        def _listen(self):
            while not self._stop.is_set():
                if self._paused.is_set():
                    self._stop.wait(0.05)
                    continue
                if not msvcrt.kbhit():
                    self._stop.wait(0.05)
                    continue
                if self._paused.is_set():
                    continue
                if msvcrt.getch() == b'\x1b':
                    self.pressed = True
                    if self._on_cancel:
                        self._on_cancel()
                    return
