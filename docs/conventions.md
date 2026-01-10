


# 📄 SoftISP Detailed README

## Overview
SoftISP is a modular, GPU‑accelerated image signal processing (ISP) pipeline built around ONNX models. It separates coefficient estimation, rule‑based stabilization, and ISP rendering into distinct models, enabling flexibility, reproducibility, and performance.

The project is licensed under GPL‑2.0 and includes:
- Python builders for ONNX graphs
- Prebuilt ONNX artifacts
- A C++ runtime scaffold
- A Python emulation harness for testing

---

## Architecture

The pipeline consists of three ONNX models:

- **Algos Model**  
  Computes coefficients (WB, CCM, gamma, NR, sharpen) from sensor metadata and active area statistics.

- **Rule Engine Model**  
  Stabilizes and gates coefficients across frames.

- **ISP Model**  
  Applies coefficients to raw Bayer input, producing YUV/RGB outputs.

### Data Flow
1. Camera thread produces stride‑aware Bayer frames and metadata.  
2. Algos model derives coefficients.  
3. Rule engine stabilizes coefficients.  
4. ISP model applies coefficients to render outputs.

---

## Conventions
- **Geometry tensors**: Always `int64` (slice starts/ends, reshape shapes, split sizes, axes).  
- **Arithmetic constants**: Always `float32`.  
- **Opset 13 compliance**:  
  - Split sizes as tensor input  
  - Unsqueeze axes as `int64`  
  - Reshape shapes as `int64`  
- **SSA naming**: Unique tensor names per trunk, suffixed per stage.

---

## Directory Layout
```
builders/        Python ONNX graph builders
  algos/
  isp/
  rule_engine/
  common/
runtime/         C++ runtime (threads, queues, ONNX sessions)
docs/            model contracts, architecture diagrams
models/          generated ONNX artifacts (versioned)
harness/         Python emulation harness for testing
```

---

## Model Contracts
See `docs/model_contracts.md` for detailed input/output specifications and example feed dictionaries.

---

## Building Models
Run builders in `builders/` to regenerate ONNX models. Validation scripts run `onnx.checker.check_model()` plus custom invariants (SSA uniqueness, dtype consistency).

Example:
```bash
python builders/algos/build_algos_model.py
python builders/isp/build_isp_model.py
python builders/rule_engine/build_rule_engine.py
```

---

## Exporting Models
Exporting ONNX models is straightforward:

1. **Within builder scripts**  
   Each builder ends with:
   ```python
   onnx.save(model, "isp_algo_coeffs_full.onnx")
   ```
   This writes the model to disk.

2. **Custom export path**  
   You can change the filename or directory:
   ```python
   out_path = "models/v1.0/isp_algo_coeffs_full.onnx"
   onnx.save(model, out_path)
   ```

3. **Programmatic export**  
   From Python REPL or another script:
   ```python
   import onnx
   from builders.algos.build_algos_model import build_algos_model

   model = build_algos_model(H=1080, W=1920)
   onnx.save(model, "models/isp_algo_coeffs_full.onnx")
   ```

---

## Running the Harness
The Python harness (`harness/test_full_pipeline_emulate.py`) emulates camera frames, feeds algos, rule engine, and ISP models in threads, and logs per‑stage timings. It must feed `height_active` and `width_active` to algos and ISP sessions.

---

## CI/CD
- CI regenerates models and stores them under `models/vX.Y.Z/` with metadata.json (opset, IR version, builder commit SHA).  
- Validation ensures SSA uniqueness, dtype consistency, and opset compliance.

---

## License
GPL‑2.0

---

✨ This README now gives contributors a clear picture of the project, its architecture, conventions, and how to build and export models.  

Would you like me to also draft a **docs/conventions.md** file that spells out naming rules (SSA suffixes, trunk numbering) and dtype defaults, so builders and runtime contributors have a single reference?
