

# SoftISP Conventions

## Geometry and Shapes
- All geometry tensors (slice starts/ends, reshape shapes, split sizes, axes) must be `int64`.
- Active area dimensions (`height_active`, `width_active`) are always `int64` inputs.
- Stride width (`W_stride`) is symbolic but treated as `int64` when passed to Slice/Concat.

## Arithmetic Constants
- All arithmetic constants (e.g., 0.0, 1.0, 2.2) are `float32`.
- Used for pixel math: WB scaling, gamma exponent, NR/sharpen coefficients.

## Opset Compliance
- Target opset: 13.
- Split sizes must be provided as tensor input (`int64`).
- Unsqueeze axes must be `int64` tensors.
- Reshape shapes must be `int64` tensors.

## SSA Naming
- Every tensor name must be unique across the graph.
- Use suffixes per trunk/stage to avoid collisions:
  - Example: `wb_4d_stage` (WB trunk), `ccm_w_ccm` (CCM trunk).
- Do not reuse names like `wb_4d` or `ccm_w` in multiple trunks.

## Trunk Numbering
Builders are organized into trunks (modular methods). Each trunk handles a logical stage:
- Trunk 1: Imports and helpers  
- Trunk 2: Inputs and outputs  
- Trunk 3: Constants and shapes  
- Trunk 4: Active crop  
- Trunk 5: Masks and weighted sums  
- Trunk 6: WB computation  
- Trunk 7: CCM and gamma/nr/sharpen  
- Trunk 8: Model assembly  
- Trunk 9: Orchestration  
- Trunk 10: Save model  

## Validation
- Run `onnx.checker.check_model()` on every generated model.
- Custom invariants:
  - SSA uniqueness
  - Dtype consistency for Concat/Slice/Reshape/Split
  - Opset compliance

## Profiling and Observability
- Each runtime stage (Camera, Algos, Rule, ISP) must log timings.
- Export per-frame stats to CSV for reproducibility.

---

