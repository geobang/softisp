"""
ResourceManager stub.

Authoritative allocation/reservation/pin API. This implementation is a mock
that enforces the external contract: only ResourceManager and CompletionDispatcher
mutate pin counts and release handles.

Extend this with platform_allocator integration to allocate real native handles.
"""
from typing import Dict, Optional, List, Tuple
import threading
import time
import uuid

class AllocationError(Exception):
    pass

class ResourceManager:
    def __init__(self, global_quota_bytes: int = 1024 * 1024 * 1024):
        self._lock = threading.Lock()
        self._allocations: Dict[str, Dict] = {}  # handle_id -> {size, fourcc, native, pins}
        self._reservations: Dict[str, Dict] = {}  # plan_token -> {bytes, ttl, created_at}
        self._global_quota = global_quota_bytes
        self._used = 0

    def status(self):
        with self._lock:
            return {
                "allocated_bytes": self._used,
                "num_handles": len(self._allocations),
                "reservations": dict(self._reservations),
            }

    def reserveforplan(self, plan_token: str, bytes_: int, ttl_s: int = 60) -> bool:
        with self._lock:
            # simple quota enforcement
            reserved_total = sum(r["bytes"] for r in self._reservations.values())
            if reserved_total + bytes_ + self._used > self._global_quota:
                return False
            self._reservations[plan_token] = {"bytes": bytes_, "ttl": ttl_s, "created_at": time.time()}
            return True

    def allocate(self, size: int, fourcc: str, hints: Optional[Dict] = None, plan_token: Optional[str] = None) -> str:
        with self._lock:
            if plan_token:
                r = self._reservations.get(plan_token)
                if not r or r["bytes"] < size:
                    raise AllocationError("Reservation missing or insufficient")
                # consume reservation bytes
                r["bytes"] -= size
                if r["bytes"] == 0:
                    del self._reservations[plan_token]
            if self._used + size > self._global_quota:
                raise AllocationError("Global quota exceeded")
            handle_id = str(uuid.uuid4())
            self._allocations[handle_id] = {
                "size": size,
                "fourcc": fourcc,
                "native": None,  # platform allocator would store native handle
                "pins": 0,
                "created_at": time.time(),
            }
            self._used += size
            return handle_id

    def pin(self, handle_id: str) -> str:
        with self._lock:
            entry = self._allocations.get(handle_id)
            if not entry:
                raise KeyError("handle not found")
            entry["pins"] += 1
            pintoken = str(uuid.uuid4())
            entry.setdefault("pin_tokens", {})[pintoken] = time.time()
            return pintoken

    def pinmany(self, handle_ids: List[str]) -> List[str]:
        return [self.pin(h) for h in handle_ids]

    def unpin(self, handle_id: str, pintoken: str) -> None:
        with self._lock:
            entry = self._allocations.get(handle_id)
            if not entry:
                # silently ignore to be lenient; Housekeeper will cleanup
                return
            if pintoken in entry.get("pin_tokens", {}):
                del entry["pin_tokens"][pintoken]
                entry["pins"] = max(0, entry["pins"] - 1)

    def unpin_many(self, handle_tokens: List[tuple]) -> None:
        for handle_id, pintoken in handle_tokens:
            self.unpin(handle_id, pintoken)

    def release(self, handle_id: str) -> None:
        with self._lock:
            entry = self._allocations.pop(handle_id, None)
            if entry:
                self._used -= entry.get("size", 0)

    def exportdmabuf(self, handle_id: str) -> int:
        # placeholder: real implementation returns an fd
        raise NotImplementedError("PlatformAllocator integration required")
