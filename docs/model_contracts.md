
# 📑 Model Contracts

## Algos Model

**Inputs**
- `input.bayer`: `(1,1,H,W_stride)`, `float32` — stride‑aware Bayer input.
- `analog_gain`: `(1)`, `float32`.
- `exposure_time`: `(1)`, `float32`.
- `sensor_temp`: `(1)`, `float32`.
- `scene_change`: `(1)`, `float32`.
- `height_active`: `(1)`, `int64` — active height.
- `width_active`: `(1)`, `int64` — active width.

**Outputs**
- `wb`: `(3)`, `float32` — white balance gains.
- `ccm`: `(9)`, `float32` — color correction matrix.
- `gamma`: `(1)`, `float32` — gamma exponent.
- `nr_strength`: `(1)`, `float32` — noise reduction strength.
- `sharp_strength`: `(1)`, `float32` — sharpening strength.

---

## Rule Engine Model

**Inputs**
- `wb_prev`, `wb_next`: `(3)`, `float32`.
- `ccm_prev`, `ccm_next`: `(9)`, `float32`.
- `gamma_prev`, `gamma_next`: `(1)`, `float32`.
- `nr_prev`, `nr_next`: `(1)`, `float32`.
- `sharp_prev`, `sharp_next`: `(1)`, `float32`.
- Sensor metadata: `analog_gain`, `exposure_time`, `sensor_temp`, `scene_change`.

**Outputs**
- `wb`: stabilized WB `(3)`.
- `ccm`: stabilized CCM `(9)`.
- `gamma`: stabilized gamma `(1)`.
- `nr_strength`: stabilized NR `(1)`.
- `sharp_strength`: stabilized sharpen `(1)`.

---

## ISP Model

**Inputs**
- `input.bayer`: `(1,1,H,W_stride)`, `float32`.
- `bit_depth`: `(1)`, `float32`.
- `blc_offset`: `(1)`, `float32`.
- `lsc_gain_map`: `(1,1,H,W_stride)`, `float32`.
- `wb`: `(3)`, `float32`.
- `ccm`: `(9)`, `float32`.
- `gamma`: `(1)`, `float32`.
- `nr_strength`: `(1)`, `float32`.
- `sharp_strength`: `(1)`, `float32`.
- `height_active`: `(1)`, `int64`.
- `width_active`: `(1)`, `int64`.

**Outputs**
- `Y`: `(1,1,H,W_active)`, `float32`.
- `U`: `(1,1,H,W_active)`, `float32`.
- `V`: `(1,1,H,W_active)`, `float32`.

---

## Conventions
- Geometry tensors (slice starts/ends, reshape shapes, split sizes, axes) are always `int64`.
- Arithmetic constants are always `float32`.
- Opset 13 compliance: Split sizes as tensor input, Unsqueeze axes as `int64`, Reshape shapes as `int64`.
- SSA naming: unique tensor names per trunk, suffixed per stage.

---

## Example Feed Dictionaries

**Algos model feed**
```python
feed = {
  "input.bayer": bayer_nchw,
  "analog_gain": np.array([2.0], dtype=np.float32),
  "exposure_time": np.array([0.01], dtype=np.float32),
  "sensor_temp": np.array([35.0], dtype=np.float32),
  "scene_change": np.array([0.0], dtype=np.float32),
  "height_active": np.array([1080], dtype=np.int64),
  "width_active": np.array([1920], dtype=np.int64),
}
```

**ISP model feed**
```python
feed = {
  "input.bayer": bayer_nchw,
  "bit_depth": np.array([10], dtype=np.float32),
  "blc_offset": np.array([64], dtype=np.float32),
  "lsc_gain_map": np.ones_like(bayer_nchw, dtype=np.float32),
  "wb": coeffs["wb"].astype(np.float32),
  "ccm": coeffs["ccm"].astype(np.float32),
  "gamma": coeffs["gamma"].astype(np.float32),
  "nr_strength": coeffs["nr_strength"].astype(np.float32),
  "sharp_strength": coeffs["sharp_strength"].astype(np.float32),
  "height_active": np.array([1080], dtype=np.int64),
  "width_active": np.array([1920], dtype=np.int64),
}
```
