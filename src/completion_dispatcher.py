"""
CompletionDispatcher: single-writer finalizer thread.
"""
import threading
import queue
from typing import Callable
from isp_core.types import CompletionEvent


class CompletionDispatcher:
    def __init__(self, mainloop_finalize_fn: Callable, telemetry_fn: Callable = None):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._finalize = mainloop_finalize_fn
        self._telemetry = telemetry_fn or (lambda ce: None)
        self._stop = threading.Event()
        self._thread.start()

    def post(self, completion_event: CompletionEvent):
        self._queue.put(completion_event)

    def _run(self):
        while not self._stop.is_set():
            try:
                ce = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                # append telemetry first (best-effort)
                self._telemetry(ce)
                # perform authoritative finalization via MainLoop
                # If provenance includes a payload object, forward it
                prov = ce.provenance or {}
                # Call finalize
                self._finalize(ce)
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def stop(self):
        self._stop.set()
