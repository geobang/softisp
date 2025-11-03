#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./create_repo.sh --owner geobang --repo softisp [--push] [--no-gh]
#
# By default this will:
# - create the local repo files under the current directory
# - initialize git and create branch main
# - if --push is provided and `gh` is installed & authenticated, create the remote repo and push
#
# Example:
#   chmod +x create_repo.sh
#   ./create_repo.sh --owner geobang --repo softisp --push

OWNER="geobang"
REPO="softisp"
PUSH=false
USE_GH=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner) OWNER="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --push) PUSH=true; shift;;
    --no-gh) USE_GH=false; shift;;
    --help) echo "Usage: $0 [--owner owner] [--repo repo] [--push] [--no-gh]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

ROOT_DIR="$(pwd)"
echo "Creating softisp skeleton in ${ROOT_DIR}"
echo "Owner: ${OWNER}  Repo: ${REPO}  Push: ${PUSH}  Use gh: ${USE_GH}"

# Create directories
mkdir -p src/isp_pipeline tests .github/workflows docs scripts

# Helper to write files
write_file() {
  local path="$1"
  local content="$2"
  printf '%s' "$content" > "$path"
  echo "Wrote $path"
}

# .gitignore
write_file ".gitignore" "# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*\$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg
pip-wheel-metadata/

# Virtualenv
venv/
.env
.env.* 
.venv

# Pytest
.cache
.pytest_cache/

# Editor
.vscode/
.idea/
"

# pyproject.toml
write_file "pyproject.toml" "[build-system]
requires = [\"setuptools>=61.0\", \"wheel\"]
build-backend = \"setuptools.build_meta\"

[project]
name = \"softisp\"
version = \"0.1.0\"
description = \"Stateless fusion-first ISP pipeline skeleton (developer prototype)\"
readme = \"README.md\"
authors = [
  { name=\"ISP Dev\", email=\"dev@example.com\" }
]
license = { text = \"MIT\" }
requires-python = \">=3.10\"
dependencies = [
    \"pytest>=7.0.0\"
]

[project.optional-dependencies]
dev = [
    \"pytest>=7.0.0\",
    \"flake8>=4.0.0\",
    \"mypy>=1.0\",
    \"setuptools>=61.0\",
]
"

# requirements-dev.txt
write_file "requirements-dev.txt" "pytest>=7.0.0
flake8>=4.0.0
mypy>=1.0
"

# Makefile
write_file "Makefile" ".PHONY: help install test lint typecheck build

help:
\t@echo \"Targets:\"
\t@echo \"  make install     - install package in editable mode with dev deps\"
\t@echo \"  make test        - run pytest\"
\t@echo \"  make lint        - run flake8\"
\t@echo \"  make typecheck   - run mypy\"
\t@echo \"  make build       - build wheel\"

install:
\tpython -m pip install --upgrade pip
\tpython -m pip install -e .[dev]

test:
\tpytest -q

lint:
\tflake8 src tests

typecheck:
\tmypy src

build:
\tpython -m build
"

# scripts/build.sh
write_file "scripts/build.sh" "#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
python -m build
echo \"Build complete. Distributions are in dist/\"
"
chmod +x scripts/build.sh

# README.md (markdown - include as-is)
cat > README.md <<'README'
# ISP Pipeline Skeleton (softisp)

This repository contains a developer skeleton for a stateless, fusion-first ISP pipeline runtime:
- Central ResourceManager authoritative buffer lifecycle
- Stateless compute workers (Envelope -> CompletionEvent)
- Two QoS lanes (realtime and background)
- ModelManager with PyTorch-dev stub
- Reservation/fusion primitives (skeleton)
- Telemetry and Housekeeper

Workers are split into dedicated modules:
- fastalgo, fastisp, slowalgo, slowisp, rawalgo, rawisp

This skeleton is intended for local experimentation and iterative development:
- run unit tests with pytest
- expand PlatformAllocator to bind to memfd/dmabuf
- implement ONNX/accelerated runtime for production

See docs/PROJECT_DOCUMENTATION.md for architecture, reproduction steps, CI details and Copilot guidance.
README

# LICENSE
cat > LICENSE <<'LICENSE'
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
LICENSE

# docs/PROJECT_DOCUMENTATION.md (markdown)
cat > docs/PROJECT_DOCUMENTATION.md <<'DOC'
# ISP Pipeline — Project Documentation & Repro Steps

This document captures the repo layout, design summary, reproduce steps, CI behavior,
and guidance for code contributors and Copilot follow-ups.

## Repository layout

- src/isp_pipeline/
  - __init__.py
  - types.py
  - resource_manager.py
  - platform_allocator.py
  - threading_manager.py
  - completion_dispatcher.py
  - telemetry_manager.py
  - housekeeper.py
  - model_manager.py
  - fastalgo.py
  - fastisp.py
  - slowalgo.py
  - slowisp.py
  - rawalgo.py
  - rawisp.py
- tests/
  - test_fast_slow_flow.py
- .github/workflows/ci.yml
- pyproject.toml
- Makefile
- scripts/build.sh
- README.md
- docs/PROJECT_DOCUMENTATION.md

## High-level design (condensed)

This repo implements a skeleton of the design:
- ResourceManager: authoritative allocator/reservation/pin lifecycle
- Stateless workers: pure functions Envelope -> CompletionEvent
- Two QoS lanes: realtime and background via ThreadingManager
- CompletionDispatcher: single-threaded lifecycle mutator and telemetry append
- Fusion/reservations: basic reserveforplan, allocate semantics in ResourceManager
- ModelManager: minimal PyTorch-dev stub (will accept models but only run when torch is present)
- PlatformAllocator: LinuxAllocatorMock placeholder for memfd/dmabuf

TelemetryManager is append-only and exportable as CSV.

## Reproduce locally

1. Clone the repo:
   git clone <repo-url>
   cd <repo>

2. Install dev dependencies (recommended in a virtualenv):
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e ".[dev]"

3. Run tests:
   make test
   or
   pytest -q

4. Run lint/typecheck (optional):
   make lint
   make typecheck

5. Build distributions:
   make build
   or
   ./scripts/build.sh

## GitHub Actions

CI workflow `.github/workflows/ci.yml`:
- triggers on push, PR, and manual dispatch
- runs on ubuntu-latest with Python 3.10 and 3.11
- installs dev dependencies (`pip install -e ".[dev]"`)
- runs flake8 (non-fatal), pytest (fatal), builds a wheel, uploads dist artifact

CI permissions:
- The workflow sets permissions for contents/checks/actions/issues/pull-requests to write.
  These permissions allow automated workflows to create artifacts, files, and interact with PRs if needed.
  Note: Organization or repo administrators control whether workflows can create tokens with these permissions.

## Extending the project

Recommended next tasks:
- Implement real linux_allocator (memfd + dmabuf export) and integrate with ResourceManager.
- Implement FusionManager: createplan/reserveforplan/selectframes/buildmergespec/scheduleplan. Ensure reservation lifecycle is respected.
- Replace ModelManager stub with:
  - PyTorch dev runner (mmap/cpu fallback)
  - ONNX/ORT or TensorRT runner with dmabuf external memory import for zero-copy
- Add IoctlApplier for V4L2 driver writes and Coordinator/Modifier chain for deterministic hardware mapping.

## Guidelines for Copilot follow-ups

When asking Copilot to help:
- Provide the exact file path you want changed (e.g., src/isp_pipeline/platform_allocator.py).
- Describe target platform (Linux kernel version, userspace libs like libdrm, Vulkan/CUDA availability).
- For memfd/dmabuf work, provide details about the export mechanism you want (libdrm, GBM, or custom ioctl).
- For ModelManager changes, indicate whether you want TorchScript/ONNX or backend acceleration (CUDA/Vulkan/NNAPI).

## CI secrets and permissions (notes)

- The workflow sets `permissions` to allow write operations. If you want workflows to create branches/push code or open PRs, you must:
  - Ensure actions are allowed to create tokens with write permissions in repo settings.
  - Provide an appropriate PAT or GitHub App if cross-repo operations are needed.
- This repo does not currently include a push/PR automation action; that can be added when a secure token/automation plan is provided.

## How to run an example demo

A future demo script can:
- Allocate a fake preview buffer -> pin
- Create an Envelope for a simulated frame
- Submit to ThreadingManager for FastAlgo and FastISP
- Wait for telemetry and print telemetry CSV

This skeleton is intentionally small to facilitate iterative development and safe experiments.

## Troubleshooting

- Flake8/mypy failures in your environment: pin dev dependency versions in requirements-dev.txt
- If tests hang: ThreadingManager uses daemon threads; ensure tests call ThreadingManager.shutdown() when done.

## Contact & contributions

- For rapid iteration, create a branch named `feature/<short-desc>` and open a PR.
- Use the ISSUE/PR templates (not included) to describe design proposals and attach telemetry artifacts exported from TelemetryManager.
DOC

# .github/workflows/ci.yml
cat > .github/workflows/ci.yml <<'CI'
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: write
  checks: write
  actions: write
  issues: write
  pull-requests: write

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: [3.10, 3.11]
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python }}

      - name: Upgrade pip
        run: python -m pip install --upgrade pip

      - name: Install dev requirements
        run: |
          python -m pip install -e ".[dev]"
      - name: Run lint
        run: |
          flake8 src tests || true
      - name: Run pytest
        run: |
          pytest -q
      - name: Build wheel
        run: |
          python -m pip install build
          python -m build
      - name: Upload artifact (dist)
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/*
CI

# src/isp_pipeline/__init__.py
cat > src/isp_pipeline/__init__.py <<'PY'
"""
softisp (isp_pipeline) package entrypoint.
Expose main modules for convenience.
"""
from .types import Envelope, CompletionEvent, SelectedFrame
from .resource_manager import ResourceManager
from .platform_allocator import PlatformAllocator, LinuxAllocatorMock
from .threading_manager import ThreadingManager, SubmitResult
from .completion_dispatcher import CompletionDispatcher
from .telemetry_manager import TelemetryManager
from .housekeeper import Housekeeper
# expose worker modules
from . import fastalgo, fastisp, slowalgo, slowisp, rawalgo, rawisp
from .model_manager import ModelManager
PY

# types.py
cat > src/isp_pipeline/types.py <<'PY'
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
PY

# resource_manager.py
cat > src/isp_pipeline/resource_manager.py <<'PY'
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
PY

# platform_allocator.py
cat > src/isp_pipeline/platform_allocator.py <<'PY'
"""
PlatformAllocator mock (Linux placeholder).

This file contains a mock stub for allocatenative/exportdmabuf. Replace with a real
memfd/dmabuf implementation in production.
"""
from typing import Any, Dict

class PlatformAllocator:
    def allocatenative(self, size: int, format: str, hints: Dict = None) -> Any:
        """
        Allocate a native handle. In a real linux_allocator this would create a memfd
        and possibly call ioctl to export dmabuf with the right format.
        Return a platform-native opaque object.
        """
        raise NotImplementedError

    def exportdmabuf(self, nativehandle: Any) -> int:
        """
        Export a dmabuf FD for zero-copy import into other runtimes.
        """
        raise NotImplementedError

    def syncfordevice(self, nativehandle: Any) -> int:
        raise NotImplementedError

    def syncforcpu(self, nativehandle: Any) -> int:
        raise NotImplementedError

    def freenative(self, nativehandle: Any) -> None:
        raise NotImplementedError


class LinuxAllocatorMock(PlatformAllocator):
    def allocatenative(self, size: int, format: str, hints: Dict = None) -> Dict:
        # Return a dict as a fake native handle
        return {"fake_memfd": True, "size": size, "format": format, "hints": hints}

    def exportdmabuf(self, nativehandle: Dict) -> int:
        # Return a fake fd integer
        return 42

    def syncfordevice(self, nativehandle: Dict) -> int:
        return -1

    def syncforcpu(self, nativehandle: Dict) -> int:
        return -1

    def freenative(self, nativehandle: Dict) -> None:
        return
PY

# threading_manager.py
cat > src/isp_pipeline/threading_manager.py <<'PY'
"""
ThreadingManager skeleton with lane-based admission control.

This is a simple, single-process model to demonstrate behaviour:
- submit_compute(envelope) returns accepted/rejected
- bounded queues per lane; immediate rejection when full
"""
from queue import Queue, Full
from threading import Thread
from typing import Dict, Any
import time

class SubmitResult:
    def __init__(self, accepted: bool, reason: str = ""):
        self.accepted = accepted
        self.reason = reason

class ThreadingManager:
    def __init__(self, completion_dispatcher, fast_capacity=4, bg_capacity=32):
        self.fast_queue = Queue(maxsize=fast_capacity)
        self.bg_queue = Queue(maxsize=bg_capacity)
        self.completion_dispatcher = completion_dispatcher
        # worker threads to drain queues and invoke workers (in tests we may replace these)
        self._running = True
        self._start_background_workers()

    def _start_background_workers(self):
        def worker_loop(q, lane_name):
            while self._running:
                try:
                    envelope, worker_callable = q.get(timeout=0.1)
                except Exception:
                    continue
                try:
                    event = worker_callable(envelope)
                except Exception as e:
                    # create failing CompletionEvent-like dict
                    event = {
                        "id": envelope.id,
                        "ok": False,
                        "written_handles": [],
                        "metrics": {"error": str(e)},
                        "provenance": {"workername": lane_name}
                    }
                self.completion_dispatcher.on_completion(event)
                q.task_done()
        Thread(target=worker_loop, args=(self.fast_queue, "fast"), daemon=True).start()
        Thread(target=worker_loop, args=(self.bg_queue, "background"), daemon=True).start()

    def submit_compute(self, envelope, worker_callable, qos="realtime"):
        q = self.fast_queue if qos == "realtime" else self.bg_queue
        try:
            q.put_nowait((envelope, worker_callable))
            return SubmitResult(True)
        except Full:
            return SubmitResult(False, "queue_full")

    def queryqueuedepth(self, lane: str):
        if lane == "fast":
            return self.fast_queue.qsize()
        return self.bg_queue.qsize()

    def shutdown(self):
        self._running = False
PY

# completion_dispatcher.py
cat > src/isp_pipeline/completion_dispatcher.py <<'PY'
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
PY

# telemetry_manager.py
cat > src/isp_pipeline/telemetry_manager.py <<'PY'
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
PY

# housekeeper.py
cat > src/isp_pipeline/housekeeper.py <<'PY'
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
PY

# model_manager.py
cat > src/isp_pipeline/model_manager.py <<'PY'
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
PY

# fastalgo.py
cat > src/isp_pipeline/fastalgo.py <<'PY'
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
PY

# fastisp.py
cat > src/isp_pipeline/fastisp.py <<'PY'
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
PY

# slowalgo.py
cat > src/isp_pipeline/slowalgo.py <<'PY'
"""
SlowAlgo worker module - high-quality coefficients & fusion hints.
"""
from .types import now_ms

def run(envelope, model_manager=None):
    start = now_ms()
    event = {
        "id": envelope.id,
        "ok": True,
        "written_handles": envelope.write_handles,
        "metrics": {
            "computelatencyms": now_ms() - start
        },
        "provenance": {
            "workername": "SlowAlgo",
            "modelhash": "mock_slowhash"
        }
    }
    return event
PY

# slowisp.py
cat > src/isp_pipeline/slowisp.py <<'PY'
"""
SlowISP worker module - high-quality renderer / fusion executor.
"""
from .types import now_ms

def run(envelope, model_manager=None):
    start = now_ms()
    event = {
        "id": envelope.id,
        "ok": True,
        "written_handles": envelope.write_handles,
        "metrics": {"computelatencyms": now_ms() - start},
        "provenance": {"workername": "SlowISP"}
    }
    return event
PY

# rawalgo.py
cat > src/isp_pipeline/rawalgo.py <<'PY'
"""
RawAlgo worker module - fallback estimator.
"""
from .types import now_ms

def run(envelope, model_manager=None):
    start = now_ms()
    event = {
        "id": envelope.id,
        "ok": True,
        "written_handles": envelope.write_handles,
        "metrics": {"computelatencyms": now_ms() - start},
        "provenance": {"workername": "RawAlgo", "fallbackused": True}
    }
    return event
PY

# rawisp.py
cat > src/isp_pipeline/rawisp.py <<'PY'
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
PY

# workers compatibility module
cat > src/isp_pipeline/workers.py <<'PY'
"""
Compatibility workers module for convenience importing.

It reuses the per-worker modules and exposes convenience functions:
- fast_algo_worker -> fastalgo.run
- fast_isp_worker  -> fastisp.run
- slow_algo_worker -> slowalgo.run
- slow_isp_worker  -> slowisp.run
- raw_algo_worker  -> rawalgo.run
- raw_isp_worker   -> rawisp.run
"""
from . import fastalgo, fastisp, slowalgo, slowisp, rawalgo, rawisp

def fast_algo_worker(envelope, model_manager=None):
    return fastalgo.run(envelope, model_manager)

def fast_isp_worker(envelope, model_manager=None):
    return fastisp.run(envelope, model_manager)

def slow_algo_worker(envelope, model_manager=None):
    return slowalgo.run(envelope, model_manager)

def slow_isp_worker(envelope, model_manager=None):
    return slowisp.run(envelope, model_manager)

def raw_algo_worker(envelope, model_manager=None):
    return rawalgo.run(envelope, model_manager)

def raw_isp_worker(envelope, model_manager=None):
    return rawisp.run(envelope, model_manager)
PY

# tests
cat > tests/test_fast_slow_flow.py <<'PY'
"""
Simple integration test exercising fast and slow submission flows.

Run with: pytest tests/test_fast_slow_flow.py
"""
import time
from isp_pipeline.types import Envelope
from isp_pipeline.resource_manager import ResourceManager
from isp_pipeline.platform_allocator import LinuxAllocatorMock
from isp_pipeline.telemetry_manager import TelemetryManager
from isp_pipeline.completion_dispatcher import CompletionDispatcher
from isp_pipeline.threading_manager import ThreadingManager
from isp_pipeline import fastalgo, fastisp, slowalgo, slowisp, rawalgo, rawisp

def test_fast_flow_accepts_and_generates_completion(tmp_path):
    rm = ResourceManager(global_quota_bytes=10*1024*1024)
    tm = TelemetryManager()
    cd = CompletionDispatcher(rm, tm)
    tmgr = ThreadingManager(cd, fast_capacity=2, bg_capacity=4)

    # Simulate allocate/pin preview buffer
    preview_hdl = rm.allocate(1024*16, "RGBA", None)
    rm.pin(preview_hdl)

    env = Envelope.create(
        selected_frames=[{"camera_id":"cam0","frame_id":"f0"}],
        read_handles=["raw-handle-1"],
        write_handles=[preview_hdl],
        payload={},
        meta={"origin":"test", "qos":"realtime"}
    )

    res = tmgr.submit_compute(env, lambda e: fastalgo.run(e), qos="realtime")
    assert res.accepted

    res2 = tmgr.submit_compute(env, lambda e: fastisp.run(e), qos="realtime")
    assert res2.accepted

    # allow workers to run
    time.sleep(0.5)
    records = tm.query_all()
    assert len(records) >= 2

    tmgr.shutdown()

def test_realtime_queue_rejects_when_full(tmp_path):
    rm = ResourceManager(global_quota_bytes=10*1024*1024)
    tm = TelemetryManager()
    cd = CompletionDispatcher(rm, tm)
    tmgr = ThreadingManager(cd, fast_capacity=1, bg_capacity=1)

    env = Envelope.create(
        selected_frames=[{"camera_id":"cam0","frame_id":"f1"}],
        read_handles=[],
        write_handles=["w1"],
        payload={},
        meta={"origin":"test", "qos":"realtime"}
    )

    # fill the fast queue
    res1 = tmgr.submit_compute(env, lambda e: (time.sleep(0.5) or fastalgo.run(e)), qos="realtime")
    assert res1.accepted
    res2 = tmgr.submit_compute(env, lambda e: fastalgo.run(e), qos="realtime")
    assert not res2.accepted
    tmgr.shutdown()
PY

# Initialize git repository and commit
if [ -d .git ]; then
  echo "Git repo already initialized"
else
  git init
  git add .
  git commit -m "Initial softisp skeleton"
  git branch -M main
  echo "Created initial commit on main"
fi

# Optionally create remote repo and push
if [ "$PUSH" = true ]; then
  if [ "$USE_GH" = true ] && command -v gh >/dev/null 2>&1; then
    # create remote repo (if it doesn't exist)
    if gh repo view "${OWNER}/${REPO}" >/dev/null 2>&1; then
      echo "Remote repo ${OWNER}/${REPO} already exists"
    else
      echo "Creating remote repo ${OWNER}/${REPO} via gh"
      gh repo create "${OWNER}/${REPO}" --public --source=. --remote=origin --push || true
    fi
    # ensure remote exists
    git remote add origin "https://github.com/${OWNER}/${REPO}.git" 2>/dev/null || true
    git push -u origin main
    echo "Pushed to https://github.com/${OWNER}/${REPO}"
  else
    echo "Skipping gh-based creation (gh not available or disabled)."
    echo "Please create the repo on GitHub and then run:"
    echo "  git remote add origin https://github.com/${OWNER}/${REPO}.git"
    echo "  git push -u origin main"
  fi
fi

echo "Bootstrap complete."
