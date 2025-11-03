````markdown
```markdown
# softisp — Phase A prototype

This repository contains a Phase‑A Python host prototype of a deterministic, auditable ISP pipeline.

Quickstart (dev):
1. python -m venv .venv && source .venv/bin/activate
2. pip install -r requirements.txt
3. ./scripts/build.sh
4. ./scripts/run_demo.sh

What the demo does:
- Generates 20 synthetic Bayer10 raw frames (2k size) with small per-frame variations.
- Preallocates a small preview pool and renders previews using the FastISP path.
- Shows fallback behavior when the realtime (fast) lane rejects admission.

Repository layout highlights:
- src/isp_core/: core types, ResourceManager and PlatformAllocator
- src/workers/: stateless worker modules: fast_algo, fast_isp, slow_algo, slow_isp, raw_algo, raw_isp
- src/managers/: FusionManager and other orchestration pieces
- src/runner.py: demo runner (main)
- scripts/: convenience build/run scripts
- tools/: synthetic model generator (Phase A)

Design, invariants, and detailed architecture are in ARCHITECTURE.md.
```
````