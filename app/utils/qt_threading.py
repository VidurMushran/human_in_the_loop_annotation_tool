from __future__ import annotations

import traceback
from typing import Callable, Optional, Any

from PyQt5.QtCore import QObject, QThread, pyqtSignal


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
