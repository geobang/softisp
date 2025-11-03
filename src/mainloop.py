"""
Thin MainLoop: serializes authoritative mutations and tracks last payloads per camera.
"""
import threading
from typing import Dict, Any
from isp_core.types import Envelope, CompletionEvent, CanonicalPayload, FrameMeta
from isp_core.resource_manager import ResourceManager
from threading_manager import ThreadingManager
from completion_dispatcher import CompletionDispatcher


class MainLoop:
    def __init__(self):
        self.rm = ResourceManager()
        # last_payloads stores last known CanonicalPayload per camera
        self.last_payloads: Dict[str, CanonicalPayload] = {}
        # completion dispatcher will call back into finalize_completion
        self._completion_dispatcher = CompletionDispatcher(self.finalize_completion, telemetry_fn=self._telemetry_append)
        self.tm = ThreadingManager(self.rm, self._completion_dispatcher.post)
        # lock to serialize authoritative operations
        self._lock = threading.Lock()

    def _telemetry_append(self, ce: CompletionEvent):
        # placeholder: append to CSV or in-memory list; keep minimal
        pass

    def on_frame_arrival(self, frame_meta: Dict[str, Any]) -> FrameMeta:
        # create FrameMeta; authoritative change must be serialized
        fm = FrameMeta(**frame_meta)
        return fm

    def submit_envelope(self, envelope: Envelope) -> Dict[str, Any]:
        # Pin all handles before submit
        pin_tokens = []
        try:
            for h in envelope.readhandles + envelope.writehandles:
                t = self.rm.pin(h)
                pin_tokens.append((h, t))
            # inject last payload if none provided
            cam = None
            if isinstance(envelope.payload, CanonicalPayload):
                # envelope already has payload
                pass
            else:
                # try to attach last payload from camera via meta
                cam = envelope.meta.get("cameraid")
                if cam and cam in self.last_payloads:
                    # can't mutate envelope (it's frozen), but we may copy payload into meta for workers
                    envelope.meta["payload_obj"] = self.last_payloads[cam]
            res = self.tm.submit_compute(envelope, {"worker": envelope.meta.get("worker")})
            if not res["accepted"]:
                # immediate unpin in same serialized context and request fallback
                for h, t in pin_tokens:
                    try:
                        self.rm.unpin(h, t)
                    except Exception:
                        pass
                return {"accepted": False, "reason": res["reason"]}
            return {"accepted": True, "reason": ""}
        except Exception as e:
            for h, t in pin_tokens:
                try:
                    self.rm.unpin(h, t)
                except Exception:
                    pass
            raise

    def finalize_completion(self, ce: CompletionEvent):
        # authoritative lifecycle changes: mark outs written, unpin read handles, free buffers if safe
        # Additionally propagate payloads from provenance if present
        with self._lock:
            try:
                prov = ce.provenance or {}
                payload = prov.get("payload_obj") or prov.get("payload")
                if payload and isinstance(payload, dict):
                    # try to reconstruct CanonicalPayload
                    try:
                        cp = CanonicalPayload(**payload)
                        self.last_payloads[cp.cameraid] = cp
                    except Exception:
                        pass
                elif payload and isinstance(payload, CanonicalPayload):
                    self.last_payloads[payload.cameraid] = payload
                # Unpin written handles if present
                # Note: CompletionDispatcher is authoritative for unpin/release in real design; simplified here
            except Exception:
                pass

    def tick_housekeeper(self):
        expired = self.rm.housekeeper_tick()
        return expired

    def shutdown(self):
        self.tm.shutdown()
        self._completion_dispatcher.stop()
