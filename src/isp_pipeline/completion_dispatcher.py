"""
CompletionDispatcher: serializes lifecycle mutations and routes next stages.

For the skeleton, it will call ResourceManager.unpin/release-like hooks provided
via injected callbacks. It also appends telemetry to TelemetryManager.
"""
import threading
from typing import Callable, Dict, Any

class CompletionDispatcher:
    def __init__(self, resource_manager, telemetry_manager):
        self._lock = threading.Lock()
        self._rm = resource_manager
        self._telemetry = telemetry_manager

    def on_completion(self, event: Dict[str, Any]) -> None:
        """
        event: dict-like CompletionEvent (for simplicity).
        This method must be single-writer for lifecycle mutations.
        """
        with self._lock:
            # record telemetry
            try:
                self._telemetry.append(event)
            except Exception:
                pass
            # unpin/release written handles if present
            written = event.get("written_handles", [])
            for h in written:
                # In this skeleton we don't have pin tokens; assume RM will handle
                try:
                    # Attempt best-effort unpin/release behavior
                    self._rm.release(h)
                except Exception:
                    pass
            # route next stage hint (no-op in skeleton)
            # In a real system we'd enqueue next workers, IO writes, etc.
