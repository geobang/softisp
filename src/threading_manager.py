"""
ThreadingManager: lanes and admission control with enriched worker wrapper.
"""
import threading
import queue
import time
from typing import Any, Dict
from isp_core.types import Envelope, CompletionEvent
from isp_core.resource_manager import ResourceManager


class Lane:
    def __init__(self, name: str, worker_fn, num_workers: int, qsize: int):
        self.name = name
        self.queue = queue.Queue(maxsize=qsize)
        self.workers = []
        self.worker_fn = worker_fn
        self._stop = threading.Event()
        for _ in range(num_workers):
            t = threading.Thread(target=self._run)
            t.daemon = True
            t.start()
            self.workers.append(t)

    def _run(self):
        while not self._stop.is_set():
            try:
                envelope, ctx = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.worker_fn(envelope, ctx)
            except Exception:
                pass
            finally:
                self.queue.task_done()

    def submit(self, envelope: Envelope, ctx) -> (bool, str):
        try:
            self.queue.put_nowait((envelope, ctx))
            return True, ""
        except queue.Full:
            return False, "lane_full"

    def stop(self):
        self._stop.set()


class ThreadingManager:
    def __init__(self, resource_manager: ResourceManager, completion_poster):
        self.rm = resource_manager
        self.completion_poster = completion_poster

        def make_worker_fn():
            def _worker(envelope: Envelope, ctx):
                enqueue_ts = ctx.get("enqueue_ts", time.time())
                # resolve memmap views using ResourceManager
                read_views = [self.rm.get_memmap_view(h, dtype=self._infer_dtype(envelope, h)) for h in envelope.readhandles]
                write_views = [self.rm.get_memmap_view(h, dtype=self._infer_dtype(envelope, h)) for h in envelope.writehandles]
                worker = ctx.get("worker")
                start = time.time()
                try:
                    ce = worker.process(envelope, read_views, write_views, ctx)
                except Exception as e:
                    ce = CompletionEvent(id=envelope.id, ok=False, writtenhandles=[], metrics={"error": str(e)}, provenance={"worker": "error"})
                # annotate metrics
                ce.metrics.setdefault("queuedelayms", int((start - enqueue_ts) * 1000))
                ce.metrics.setdefault("computelatencyms", int((time.time() - start) * 1000))
                # If worker returned a CanonicalPayload in provenance, ensure it's serializable
                prov = ce.provenance or {}
                payload_obj = prov.get("payload_obj") or prov.get("payload")
                if payload_obj:
                    # ensure payload is a dict for transmission
                    try:
                        if hasattr(payload_obj, "dict"):
                            prov["payload"] = payload_obj.dict()
                        else:
                            prov["payload"] = payload_obj
                        ce = ce.copy(update={"provenance": prov})
                    except Exception:
                        pass
                # post to completion dispatcher
                self.completion_poster(ce)

            return _worker

        worker_fn = make_worker_fn()
        # fast lane: small worker count, bounded
        self.lanes = {}
        self.lanes["fast"] = Lane("fast", worker_fn, num_workers=2, qsize=4)
        self.lanes["background"] = Lane("background", worker_fn, num_workers=4, qsize=64)

    def _infer_dtype(self, envelope: Envelope, handle: str):
        # best-effort: if preview handle assume uint8 RGB; else default uint16 for Bayer
        if "preview" in handle or "RGBA" in str(handle) or "preview" in envelope.meta.get("origin", ""):
            return "uint8"
        return "uint16"

    def submit_compute(self, envelope: Envelope, ctx: Dict) -> Dict[str, Any]:
        qos = envelope.meta.get("qos", "background")
        lane = self.lanes.get("fast" if qos == "realtime" else "background")
        ctx["enqueue_ts"] = time.time()
        accepted, reason = lane.submit(envelope, ctx)
        return {"accepted": accepted, "reason": reason}

    def shutdown(self):
        for l in self.lanes.values():
            l.stop()
