"""
Canonical types: Envelope and CompletionEvent dataclasses.

These are intentionally lightweight and serializable to JSON for replay and telemetry.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import uuid
import time


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class SelectedFrame:
    camera_id: str
    frame_id: str


@dataclass(frozen=True)
class Envelope:
    id: str
    selected_frames: List[SelectedFrame]
    read_handles: List[str]
    write_handles: List[str]
    payload: Dict[str, Any]
    meta: Dict[str, Any]

    @staticmethod
    def create(selected_frames, read_handles, write_handles, payload, meta):
        return Envelope(
            id=str(uuid.uuid4()),
            selected_frames=[SelectedFrame(**sf) for sf in selected_frames],
            read_handles=list(read_handles),
            write_handles=list(write_handles),
            payload=payload or {},
            meta=meta or {},
        )


@dataclass(frozen=True)
class CompletionEvent:
    id: str
    ok: bool
    written_handles: List[str]
    metrics: Dict[str, Any]
    provenance: Dict[str, Any]
    nextstagehint: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return asdict(self)
