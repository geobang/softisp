"""
RawISP worker module - minimal preview renderer for fallback.
"""
from .types import now_ms

def run(envelope, model_manager=None):
    start = now_ms()
    event = {
        "id": envelope.id,
        "ok": True,
        "written_handles": envelope.write_handles,
        "metrics": {"computelatencyms": now_ms() - start},
        "provenance": {"workername": "RawISP", "fallbackused": True}
    }
    return event
