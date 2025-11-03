"""
Housekeeper stub for TTL-based eviction and cleanup of reservations and leaked handles.
"""
import threading
import time

class Housekeeper:
    def __init__(self, resource_manager, telemetry_manager, interval_s=10):
        self.rm = resource_manager
        self.telemetry = telemetry_manager
        self.interval = interval_s
        self._running = False

    def start(self):
        import threading
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            self._cleanup_reservations()
            time.sleep(self.interval)

    def _cleanup_reservations(self):
        now = time.time()
        to_delete = []
        for token, data in list(self.rm._reservations.items()):
            if data["created_at"] + data["ttl"] < now:
                to_delete.append(token)
        for t in to_delete:
            del self.rm._reservations[t]
