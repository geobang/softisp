"""
Immutable data contracts and FrameMeta for the pipeline.
"""
from typing import List, Optional, Tuple, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
import json


class FrameMeta(BaseModel):
    frameid: str
    cameraid: str
    timestamp: float
    requested_exposure: Optional[Dict[str, Any]] = None
    applied_registers: Optional[Dict[str, Any]] = None
    predicted_actuals: Optional[Dict[str, Any]] = None
    measured_stats: Optional[Dict[str, Any]] = None
    awb_gains: Optional[Dict[str, float]] = None
    motion_vector: Optional[Dict[str, Any]] = None

    class Config:
        frozen = True


class CanonicalPayload(BaseModel):
    frameid: str
    cameraid: str
    timestamp: float
    exposure: Dict[str, Any]
    wb_gains: Dict[str, float]
    lscgrid: Optional[Dict[str, Any]] = None
    ccm_matrix: Optional[List[List[float]]] = None
    tone_map: Optional[Dict[str, Any]] = None

    class Config:
        frozen = True


class MergeSpec(BaseModel):
    plan_id: str
    frames: List[Tuple[str, str]]
    transforms: Optional[List[Dict[str, Any]]] = None
    modelhash: Optional[str] = None
    seed: Optional[int] = None
    output_writehandles: List[str] = []

    class Config:
        frozen = True


class Envelope(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    selectedframes: List[Tuple[str, str]] = []
    readhandles: List[str] = []
    writehandles: List[str] = []
    payload: Any = None  # CanonicalPayload or MergeSpec
    meta: Dict[str, Any] = {}

    class Config:
        frozen = True

    def to_json(self) -> str:
        return self.json()

    @classmethod
    def from_json(cls, s: str) -> "Envelope":
        return cls.parse_raw(s)


class CompletionEvent(BaseModel):
    id: UUID
    ok: bool
    writtenhandles: List[str] = []
    metrics: Dict[str, Any] = {}
    provenance: Dict[str, Any] = {}
    nextstagehint: Optional[str] = None

    class Config:
        frozen = True

    def to_json(self) -> str:
        return self.json()

    @classmethod
    def from_json(cls, s: str) -> "CompletionEvent":
        return cls.parse_raw(s)
