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
