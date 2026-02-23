from __future__ import annotations

import traceback
from typing import Callable, Optional, Any

from PyQt5.QtCore import QObject, QThread, pyqtSignal
import queue
import threading


class _Worker(QObject):
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    done = pyqtSignal(object)

    def __init__(self, fn: Callable, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            # fn signature: fn(log_cb) -> Any
            out = self._fn(lambda s: self.log.emit(str(s)))
            self.done.emit(out)
        except Exception:
            self.error.emit(traceback.format_exc())
            self.done.emit(None)


def run_in_thread(
    fn: Callable[[Callable[[str], None]], Any],
    *,
    parent: Optional[QObject] = None,
    on_log: Optional[Callable[[str], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
    on_done: Optional[Callable[[], None]] = None,
):
    """
    Run fn(log_cb) in a QThread.
    CRITICAL: keeps strong refs on parent to prevent QThread GC crashes.
    """
    th = QThread(parent)
    worker = _Worker(fn)
    worker.moveToThread(th)

    if on_log:
        worker.log.connect(on_log)
    if on_error:
        worker.error.connect(on_error)

    def _cleanup(_result=None):
        # stop thread
        th.quit()
        th.wait(2000)
        worker.deleteLater()
        th.deleteLater()

        # remove from parent list if present
        if parent is not None and hasattr(parent, "_qt_threads"):
            try:
                parent._qt_threads.remove((th, worker))
            except Exception:
                pass

        if on_done:
            on_done()

    worker.done.connect(_cleanup)

    th.started.connect(worker.run)

    # KEEP REFS
    if parent is not None:
        if not hasattr(parent, "_qt_threads"):
            parent._qt_threads = []
        parent._qt_threads.append((th, worker))

    th.start()
    return th


class JobQueueManager(QObject):
    log_signal = pyqtSignal(str)
    job_started = pyqtSignal(str)
    job_finished = pyqtSignal(str, bool)  # job_name, success
    queue_empty = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._is_running = False

    def add_job(self, name: str, fn: Callable):
        self._queue.put((name, fn))
        if not self._is_running:
            self.start_worker()

    def start_worker(self):
        self._is_running = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()

    def stop_worker(self):
        self._stop_event.set()
        self._is_running = False

    def get_pending_count(self):
        return self._queue.qsize()

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                name, fn = self._queue.get(timeout=1.0)
            except queue.Empty:
                if self._is_running:
                    self._is_running = False
                    self.queue_empty.emit()
                break

            self.job_started.emit(name)
            try:
                # Execute the job. It must accept a log callback.
                fn(lambda msg: self.log_signal.emit(str(msg)))
                self.job_finished.emit(name, True)
            except Exception as e:
                self.log_signal.emit(f"[{name}] FATAL ERROR: {traceback.format_exc()}")
                self.job_finished.emit(name, False)
            finally:
                self._queue.task_done()