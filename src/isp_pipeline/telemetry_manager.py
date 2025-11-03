"""
TelemetryManager: append-only provenance store with CSV export helper.
"""
import threading
import csv
from typing import List, Dict, Any

class TelemetryManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._store: List[Dict[str, Any]] = []

    def append(self, completion_event: Dict[str, Any]) -> None:
        with self._lock:
            self._store.append(completion_event.copy())

    def query_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._store)

    def export_csv(self, path: str, fieldnames: List[str] = None) -> None:
        with self._lock:
            if not self._store:
                return
            if not fieldnames:
                # derive union of keys
                keys = set()
                for r in self._store:
                    keys.update(r.keys())
                fieldnames = list(keys)
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self._store:
                    writer.writerow({k: r.get(k, "") for k in fieldnames})
