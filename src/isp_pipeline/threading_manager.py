"""
ThreadingManager skeleton with lane-based admission control.

This is a simple, single-process model to demonstrate behaviour:
- submit_compute(envelope) returns accepted/rejected
- bounded queues per lane; immediate rejection when full
"""
from queue import Queue, Full
from threading import Thread
from typing import Dict, Any
import time

class SubmitResult:
    def __init__(self, accepted: bool, reason: str = ""):
        self.accepted = accepted
        self.reason = reason

class ThreadingManager:
    def __init__(self, completion_dispatcher, fast_capacity=4, bg_capacity=32):
        self.fast_queue = Queue(maxsize=fast_capacity)
        self.bg_queue = Queue(maxsize=bg_capacity)
        self.completion_dispatcher = completion_dispatcher
        # worker threads to drain queues and invoke workers (in tests we may replace these)
        self._running = True
        self._start_background_workers()

    def _start_background_workers(self):
        def worker_loop(q, lane_name):
            while self._running:
                try:
                    envelope, worker_callable = q.get(timeout=0.1)
                except Exception:
                    continue
                try:
                    event = worker_callable(envelope)
                except Exception as e:
                    # create failing CompletionEvent-like dict
                    event = {
                        "id": envelope.id,
                        "ok": False,
                        "written_handles": [],
                        "metrics": {"error": str(e)},
                        "provenance": {"workername": lane_name}
                    }
                self.completion_dispatcher.on_completion(event)
                q.task_done()
        Thread(target=worker_loop, args=(self.fast_queue, "fast"), daemon=True).start()
        Thread(target=worker_loop, args=(self.bg_queue, "background"), daemon=True).start()

    def submit_compute(self, envelope, worker_callable, qos="realtime"):
        q = self.fast_queue if qos == "realtime" else self.bg_queue
        try:
            q.put_nowait((envelope, worker_callable))
            return SubmitResult(True)
        except Full:
            return SubmitResult(False, "queue_full")

    def queryqueuedepth(self, lane: str):
        if lane == "fast":
            return self.fast_queue.qsize()
        return self.bg_queue.qsize()

    def shutdown(self):
        self._running = False
