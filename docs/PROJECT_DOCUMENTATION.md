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
