import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
import tempfile

def ensure_ir_compat(model_path, max_ir=11):
    """
    Ensure model IR version is <= max_ir.
    If it's higher, re-save with downgraded ir_version to a temp file and return that path.
    """
    model = onnx.load(model_path)
    if hasattr(model, "ir_version") and model.ir_version > max_ir:
        model.ir_version = max_ir
        fd, tmp_path = tempfile.mkstemp(suffix=".onnx", prefix="ruleengine_irfix_")
        os.close(fd)
        onnx.save(model, tmp_path)
        return tmp_path
    return model_path

def pick_provider():
    """
    Choose a valid provider available in current ORT install.
    Prefer CPU; fall back gracefully.
    """
    avail = ort.get_available_providers()
    if "CPUExecutionProvider" in avail:
        return ["CPUExecutionProvider"]
    # Fall back to first available
    return [avail[0]] if avail else None

def make_default_input(name, elem_type, shape):
    """
    Create a sensible default input based on name and shape.
    Uses dtype float32 for all numeric inputs (to match builder),
    except when a boolean scalar is detected.
    """
    # Map ONNX types to numpy dtypes (simplified; builder uses FLOAT)
    dtype = np.float32

    # Build values based on semantic name
    if name == "raw_wb":
        return np.array([1.2, 0.9, 1.1], dtype=dtype)
    if name == "prev_wb":
        return np.array([1.0, 1.0, 1.0], dtype=dtype)
    if name == "raw_ccm":
        return np.eye(3, dtype=dtype).flatten()
    if name == "prev_ccm":
        return np.eye(3, dtype=dtype).flatten()
    if name == "raw_gamma":
        n = shape[0] if shape and shape[0] is not None else 256
        return np.linspace(0, 1, n, dtype=dtype)
    if name == "prev_gamma":
        n = shape[0] if shape and shape[0] is not None else 256
        return np.linspace(0, 1, n, dtype=dtype)
    if name == "raw_sharpness":
        return np.array([1.0], dtype=dtype)
    if name == "prev_sharpness":
        return np.array([1.0], dtype=dtype)
    if name == "raw_nr":
        return np.array([0.5], dtype=dtype)
    if name == "prev_nr":
        return np.array([0.5], dtype=dtype)

    # Sensor meta
    if name == "analog_gain":
        return np.array([2.0], dtype=dtype)
    if name == "exposure_time":
        return np.array([0.01], dtype=dtype)
    if name == "sensor_temp":
        return np.array([40.0], dtype=dtype)
    # Note: the graph defines scene_change as FLOAT [1], so keep float 0/1
    if name == "scene_change":
        return np.array([0.0], dtype=dtype)

    # Rule params
    if name in ("alpha_wb", "alpha_ccm", "alpha_gamma", "alpha_sharp", "alpha_nr"):
        return np.array([0.2], dtype=dtype)
    if name == "alpha_fast":
        return np.array([0.5], dtype=dtype)

    # Steps
    if name == "wb_step":
        return np.array([0.05], dtype=dtype)
    if name == "sharp_step":
        return np.array([0.05], dtype=dtype)

    # Clamps
    if name == "wb_min":
        return np.array([0.5], dtype=dtype)
    if name == "wb_max":
        return np.array([2.0], dtype=dtype)
    if name == "gamma_min":
        return np.array([0.0], dtype=dtype)
    if name == "gamma_max":
        return np.array([1.0], dtype=dtype)
    if name == "ccm_min":
        return np.array([-2.0], dtype=dtype)
    if name == "ccm_max":
        return np.array([2.0], dtype=dtype)
    if name == "nr_min":
        return np.array([0.0], dtype=dtype)
    if name == "nr_max":
        return np.array([5.0], dtype=dtype)

    # Fallback: zeros matching shape
    # For scalars [1], vector [N], or known shapes
    if not shape:
        return np.array([0.0], dtype=dtype)
    if len(shape) == 1 and shape[0] is not None:
        return np.zeros(shape[0], dtype=dtype)
    return np.array([0.0], dtype=dtype)

def run_ruleengine(model_path="ruleengine_full.onnx"):
    # Ensure IR compatibility
    compat_model_path = ensure_ir_compat(model_path, max_ir=11)

    # Pick provider
    providers = pick_provider()
    if not providers:
        raise RuntimeError("No ONNX Runtime providers available.")

    # Create session
    sess = ort.InferenceSession(compat_model_path, providers=providers)

    # Build inputs from model metadata to stay compatible with latest graph
    inputs = {}
    for inp in sess.get_inputs():
        name = inp.name
        elem_type = inp.type
        # Shape may include None for dynamic dims; handle gracefully
        shape = [d if isinstance(d, int) else (d if d is not None else None) for d in inp.shape]
        inputs[name] = make_default_input(name, elem_type, shape)

    # Run inference
    outputs = sess.run(None, inputs)

    # Name map for readability
    output_names = [o.name for o in sess.get_outputs()]
    outputs_dict = {name: val for name, val in zip(output_names, outputs)}

    # Extract and validate
    wb_stab = outputs_dict.get("wb_stab")
    ccm_stab = outputs_dict.get("ccm_stab")
    gamma_stab = outputs_dict.get("gamma_stab")
    sharpness_stab = outputs_dict.get("sharpness_stab")
    nr_stab = outputs_dict.get("nr_stab")

    # Basic sanity checks with actual shapes
    assert wb_stab is not None and wb_stab.shape == (3,)
    assert ccm_stab is not None and ccm_stab.shape == (9,)
    assert gamma_stab is not None and gamma_stab.shape[0] >= 16  # allow variable gammaN
    assert sharpness_stab is not None and sharpness_stab.shape == (1,)
    assert nr_stab is not None and nr_stab.shape == (1,)

    # Range checks
    assert np.all(wb_stab >= 0.5) and np.all(wb_stab <= 2.0)
    assert np.all(gamma_stab >= 0.0) and np.all(gamma_stab <= 1.0)
    assert np.all(ccm_stab >= -2.0) and np.all(ccm_stab <= 2.0)
    assert 0.0 <= float(sharpness_stab[0]) <= 1.0
    assert 0.0 <= float(nr_stab[0]) <= 5.0

    print("RuleEngine ONNX test passed ✅")
    print("wb_stab:", wb_stab)
    print("sharpness_stab:", sharpness_stab)
    print("nr_stab:", nr_stab)

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "ruleengine_full.onnx"
    run_ruleengine(model_path)
