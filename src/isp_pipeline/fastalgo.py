"""
FastAlgo worker module - realtime fast coefficient generator.

This module exports `run(envelope)` which returns a CompletionEvent-like dict.
"""
from .types import now_ms

def run(envelope, model_manager=None):
    start = now_ms()
    # produce a tiny canonical payload (placeholder)
    payload = {
        "frame_id": envelope.selected_frames[0].frame_id,
        "camera_id": envelope.selected_frames[0].camera_id,
        "timestamp": now_ms(),
        "wb_gains": {"r": 1.0, "g": 1.0, "b": 1.0},
        "ccm_matrix": [[1.0,0,0],[0,1.0,0],[0,0,1.0]],
        "provenance": {"producer": "fast_algo", "modelhash": None}
    }
    event = {
        "id": envelope.id,
        "ok": True,
        "written_handles": envelope.write_handles,
        "metrics": {
            "computelatencyms": now_ms() - start
        },
        "provenance": {
            "workername": "FastAlgo",
            "modelhash": None,
            "deterministicflag": bool(envelope.meta.get("deterministicseed"))
        }
    }
    return event
