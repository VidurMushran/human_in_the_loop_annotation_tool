import json
import os
import threading
import time
from PyQt5.QtCore import QObject, pyqtSignal

QUEUE_FILE = "queue.json"

class PersistentQueueManager(QObject):
    log_signal = pyqtSignal(str)
    job_started = pyqtSignal(str)
    job_finished = pyqtSignal(str, bool)
    queue_empty = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.stop_event = threading.Event()
        self._load_queue()

    def _load_queue(self):
        if os.path.exists(QUEUE_FILE):
            with open(QUEUE_FILE, 'r') as f:
                self.queue = json.load(f)
        else:
            self.queue = []

    def _save_queue(self):
        with open(QUEUE_FILE, 'w') as f:
            json.dump(self.queue, f, indent=2)

    def add_job(self, job_def):
        self.queue.append(job_def)
        self._save_queue()
        self.log_signal.emit(f"Added job: {job_def['job_name']}")
        if not self.running:
            self.start_worker()

    def start_worker(self):
        self.running = True
        self.stop_event.clear()
        threading.Thread(target=self._worker_loop, daemon=True).start()

    def _worker_loop(self):
        # Avoid circular imports by importing inside the thread
        from app.training.jobs import execute_job_from_def
        
        while not self.stop_event.is_set():
            if not self.queue:
                self.running = False
                self.queue_empty.emit()
                break

            job = self.queue[0] # Peek
            self.job_started.emit(job['job_name'])
            
            try:
                execute_job_from_def(job, lambda s: self.log_signal.emit(f"[{job['job_name']}] {s}"))
                self.job_finished.emit(job['job_name'], True)
                
                # Pop only on success or handled failure to avoid infinite loop
                self.queue.pop(0) 
                self._save_queue()
                
            except Exception as e:
                self.log_signal.emit(f"Job Failed: {e}")
                self.job_finished.emit(job['job_name'], False)
                # Move failed job to end or remove? Removing to prevent lock.
                self.queue.pop(0)
                self._save_queue()