````markdown
```markdown
# softisp Phase A — Detailed Design & Rationale

This document describes the Phase‑A design in detail. It tracks invariants, data contracts, runtime architecture, buffer lifecycles, QoS assumptions, and the concrete Python prototype mounting points.

Table of contents
- Executive summary
- Goals & invariants
- Data contracts
- Runtime architecture (MainLoop, ThreadingManager, CompletionDispatcher)
- Resource lifecycle & reservations
- Workers and models
- FusionManager plan lifecycle
- Zero‑copy strategy & fallbacks
- Observability & telemetry
- Testing strategy & demo harness
- Migration path to Phase B/C

Executive summary
-----------------
Phase A aims to deliver a production-minded, PyTorch-friendly host prototype that proves:
- Deterministic provenance per frame/job (modelhash, seed, plantoken, zero_copy flag)
- Single authoritative ResourceManager for allocations, reservations, pin bookkeeping
- Stateless compute workers that never manage lifetime
- Thin MainLoop that performs only serial, idempotent authoritative mutations
- Two QoS lanes: realtime Fast lane (bounded, preallocated preview pool) and Background lane (batching allowed)
- Safe batch fusion via reservation-before-allocation semantics

Core invariants
---------------
- Only ResourceManager modifies pin counts and release operations.
- Workers do not allocate/pin/unpin/release buffers; they are pure functions Envelope -> CompletionEvent.
- MainLoop is the serial executor for authoritative FrameRecord changes.
- Fast lane admission is fail-fast; submitter must fallback on reject.
- Reservations must be obtained before allocating Fusion outputs.

Data contracts
--------------
- Envelope (immutable): id, selectedframes, readhandles, writehandles, payload (CanonicalPayload / MergeSpec), meta
- CompletionEvent (immutable): id, ok, writtenhandles, metrics, provenance, nextstagehint
- CanonicalPayload: coefficient blob with exposure, wb_gains, lscgrid, ccm_matrix, tone_map refs
- MergeSpec: deterministic batch descriptor; plan_id, frames, transforms, modelhash, seed, output write handles

Runtime architecture
--------------------
- ResourceManager: authoritative allocator and bookkeeping (implemented in src/isp_core/resource_manager.py)
- PlatformAllocator: platform-backed memmap file allocator (simulated for Phase A)
- MainLoop: thin coordinator that pins before submit, handles immediate unpin on reject, calls ThreadingManager.submit_compute
- ThreadingManager: lanes (fast, background) with bounded queue semantics and worker threads; submit_compute is non-blocking for realtime lane
- CompletionDispatcher: single-writer finalizer appending telemetry and invoking finalize logic (single threaded)
- FusionManager: plan lifecycle (create, reserve, allocate, schedule) with TTL enforced by housekeeper

Workers and models
------------------
Workers are stateless and deterministic when deterministic_seed is provided. The Phase A implementation includes:
- FastAlgo: coarse, fast coefficient estimator (tile stats, AWB heuristic, coarse LSC)
- FastISP: low-latency renderer (bilinear demosaic, LSC upsample, WB, CCM, gamma, resize)
- SlowAlgo: multi-frame refined LSC generation (integer-shift align + averaging)
- SlowISP: fusion renderer (integer-shift accumulation)
- RawAlgo/RawISP: ultra-fast fallback path for realtime safety

FusionManager
-------------
Implements plan creation, reservation-before-allocation, deterministic MergeSpec creation, and scheduling. Allocations for plans must consume reservation accounting to keep quotas bounded.

Zero‑copy & import fallbacks
----------------------------
- Phase A mocks dmabuf helpers; ModelManager will try zero-copy import but falls back to explicit copy path and records zero_copy=false.
- For FastISP prefer memmap-backed buffer views to avoid intermediate copies when writing previews.

Observability & telemetry
-------------------------
- CompletionEvent.provenance includes workername, modelhash, deterministic_seed, zero_copy, fallback_used.
- TelemetryManager should append CSV rows with full provenance for each job (Phase A demo collects basic telemetry in stdout).

Testing & demo
--------------
- src/runner.py provides a demo harness that generates synthetic Bayer frames, submits preview jobs, and exercises fast-lane rejection + fallback.
- Unit tests should validate ResourceManager invariants (pin/unpin), worker determinism, and FusionManager reservation lifecycle.

Migration path
--------------
- Phase B: Port ResourceManager and PlatformAllocator to native C++ with memfd/dmabuf and real fence semantics.
- Phase C: Android HAL integration (AHardwareBuffer/gralloc imports).
- Phase D: Optimized runners (ONNX/TensorRT) and production zero-copy support.

Notes on this prototype
-----------------------
- Phase A avoids privileged syscalls; memmap-backed files are used as stand-ins for dmabuf.
- The MainLoop remains deliberately thin and authoritative. Fusion flows must go through reserve→allocate→schedule.
- The demo runner simplifies the FastAlgo→FastISP chain by executing the fast coefficient generation synchronously to obtain a payload, then submitting the FastISP job to the threaded fast lane; this keeps the demo deterministic and easy to reason about.

Contact & contribution
----------------------
- For questions and patches, open PRs against this repository. The README and docs list development steps and runbook.
```
````