"""
FastISP worker module - realtime preview renderer.
"""
from .types import now_ms

def run(envelope, model_manager=None):
    start = now_ms()
    # simulate rendering into preview buffer (write_handles)
    event = {
        "id": envelope.id,
        "ok": True,
        "written_handles": envelope.write_handles,
        "metrics": { "computelatencyms": now_ms() - start },
        "provenance": { "workername": "FastISP", "zero_copy": False }
    }
    return event
