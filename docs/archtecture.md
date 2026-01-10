SoftISP Architecture

Overview

SoftISP is a modular image signal processing (ISP) pipeline built around ONNX models. It separates coefficient estimation, rule-based stabilization, and ISP rendering into distinct models, enabling flexibility, reproducibility, and performance.

Pipeline Components

Camera Thread

Produces stride-aware Bayer frames and sensor metadata.

Dispatches bundles to Algos and ISP threads.

Algos Model

Inputs: Bayer frame, sensor metadata, active geometry.

Outputs: Coefficients (WB, CCM, gamma, NR, sharpen).

Runs in its own thread, producing raw coefficients per frame.

Rule Engine Model

Inputs: Previous and next coefficients, sensor metadata.

Outputs: Stabilized coefficients.

Ensures smooth transitions and gating logic.

ISP Model

Inputs: Bayer frame, stabilized coefficients, active geometry.

Outputs: YUV/RGB planes.

Applies coefficients to render final image.

Thread Interaction

Camera Thread generates Bayer + metadata.

Algos Thread computes raw coefficients.

Coordinator Thread runs Rule Engine to stabilize coefficients.

ISP Thread applies coefficients to Bayer to produce outputs.

Housekeeping Thread logs events and metrics.

Conventions

Geometry tensors: int64.

Arithmetic constants: float32.

Opset 13 compliance: Split sizes as tensor input, Unsqueeze axes as int64, Reshape shapes as int64.

SSA naming: unique tensor names per trunk.

Validation

All models validated with onnx.checker.check_model().

Custom invariants: SSA uniqueness, dtype consistency, opset compliance.

Observability

Each thread logs timings.

Per-frame stats exported to CSV.

Aggregate summaries logged at shutdown.

Directory Layout

builders/        Python ONNX graph builders
runtime/         C++ runtime (threads, queues, ONNX sessions)
docs/            model contracts, conventions, architecture
models/          generated ONNX artifacts (versioned)
harness/         Python emulation harness for testing

This architecture document ties together the model contracts and conventions, showing how the pipeline flows across threads and models.
