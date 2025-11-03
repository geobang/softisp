"""
ModelManager stub with a simple PyTorch-dev backend path.

- loadmodel returns a model_id and modelhash (sha-like string).
- submitinference runs a model if PyTorch is available; otherwise rejects.
This is deliberately minimal; extend to implement zero-copy dmabuf import and ONNX runners.
"""
import hashlib
import json
import uuid
from typing import Dict, Any, Optional

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

class ModelLoadError(Exception):
    pass

class ModelManager:
    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}

    def _compute_hash(self, blob: bytes) -> str:
        return hashlib.sha256(blob).hexdigest()

    def loadmodel(self, path_or_blob: bytes, backendhint: str = "pytorch", options: Optional[Dict] = None):
        blob = path_or_blob if isinstance(path_or_blob, (bytes, bytearray)) else str(path_or_blob).encode("utf-8")
        modelhash = self._compute_hash(blob)[:16]
        model_id = str(uuid.uuid4())
        # For dev path we accept a "torchscript" blob; for now store the blob only
        self._models[model_id] = {"hash": modelhash, "backend": backendhint, "blob": blob, "options": options}
        return model_id, modelhash

    def submitinference(self, model_id: str, input_handles, output_handles, meta: Dict[str, Any]):
        # In real system we'd import dmabuf and run zero-copy. Here we either simulate or reject.
        model = self._models.get(model_id)
        if not model:
            return {"accepted": False, "reason": "model_not_loaded"}
        if model["backend"] == "pytorch" and HAS_TORCH:
            # Mock execution: return job id
            return {"accepted": True, "job_id": str(uuid.uuid4())}
        # fallback: emulate rejection for unsupported runtime
        return {"accepted": False, "reason": "backend_unavailable"}
