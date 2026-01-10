import onnx
from onnx import helper, TensorProto
import subprocess, shutil

# ---- Helpers ----
def const_scalar(name, value):
    return helper.make_node(
        "Constant", [], [name],
        value=helper.make_tensor(name + "_t", TensorProto.FLOAT, [1], [float(value)])
    )

def const_axes(name, axes):
    return helper.make_node(
        "Constant", [], [name],
        value=helper.make_tensor(name + "_t", TensorProto.INT64, [len(axes)], axes)
    )

def neg_scalar(input_name, output_name):
    return helper.make_node("Neg", [input_name], [output_name])

def make_input(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

def ema_vector(raw, prev, alpha, out):
    one = const_scalar(out + "_one", 1.0)
    alpha_comp = helper.make_node("Sub", [one.output[0], alpha], [out + "_alpha_comp"])
    mul_raw = helper.make_node("Mul", [raw, alpha], [out + "_mul_raw"])
    mul_prev = helper.make_node("Mul", [prev, alpha_comp.output[0]], [out + "_mul_prev"])
    add = helper.make_node("Add", [mul_raw.output[0], mul_prev.output[0]], [out])
    return [one, alpha_comp, mul_raw, mul_prev, add]

def rate_limit_vector(raw, prev, step, out):
    delta = helper.make_node("Sub", [raw, prev], [out + "_delta"])
    neg_step = neg_scalar(step, out + "_neg_step")
    clip = helper.make_node("Clip", [delta.output[0], neg_step.output[0], step], [out + "_clip"])
    add = helper.make_node("Add", [prev, clip.output[0]], [out])
    return [delta, neg_step, clip, add]

def reduce_mean_vec(x, out):
    return helper.make_node("ReduceMean", [x], [out], keepdims=0)

def div_vector(a, b, out):
    return helper.make_node("Div", [a, b], [out])

def max_scalar(a, b, out):
    return helper.make_node("Max", [a, b], [out])

# ---- Arithmetic blend helper ----
def blend_alpha(base_alpha, fast_alpha, scene_change, out_name):
    alpha_eff_min = out_name + "_eff_min"
    maxn = helper.make_node("Max", [base_alpha, fast_alpha], [alpha_eff_min])

    one = const_scalar(out_name + "_one", 1.0)
    one_minus = helper.make_node("Sub", [one.output[0], scene_change], [out_name + "_one_minus"])

    term_fast = helper.make_node("Mul", [scene_change, alpha_eff_min], [out_name + "_term_fast"])
    term_base = helper.make_node("Mul", [one_minus.output[0], base_alpha], [out_name + "_term_base"])
    add = helper.make_node("Add", [term_fast.output[0], term_base.output[0]], [out_name])

    return [maxn, one, one_minus, term_fast, term_base, add]

# ---- Builder ----
def build_ruleengine(gammaN=256, model_name="ruleengine_full.onnx"):
    nodes = []

    # Inputs
    raw_wb = make_input("raw_wb", [3])
    raw_ccm = make_input("raw_ccm", [9])
    raw_gamma = make_input("raw_gamma", [gammaN])
    raw_sharpness = make_input("raw_sharpness", [1])
    raw_nr = make_input("raw_nr", [1])

    prev_wb = make_input("prev_wb", [3])
    prev_ccm = make_input("prev_ccm", [9])
    prev_gamma = make_input("prev_gamma", [gammaN])
    prev_sharpness = make_input("prev_sharpness", [1])
    prev_nr = make_input("prev_nr", [1])

    analog_gain = make_input("analog_gain", [1])
    exposure_time = make_input("exposure_time", [1])
    sensor_temp = make_input("sensor_temp", [1])
    scene_change = make_input("scene_change", [1])

    alpha_wb = make_input("alpha_wb", [1])
    alpha_ccm = make_input("alpha_ccm", [1])
    alpha_gamma = make_input("alpha_gamma", [1])
    alpha_sharp = make_input("alpha_sharp", [1])
    alpha_nr = make_input("alpha_nr", [1])
    alpha_fast = make_input("alpha_fast", [1])

    wb_step = make_input("wb_step", [1])
    sharp_step = make_input("sharp_step", [1])

    wb_min = make_input("wb_min", [1])
    wb_max = make_input("wb_max", [1])
    gamma_min = make_input("gamma_min", [1])
    gamma_max = make_input("gamma_max", [1])
    ccm_min = make_input("ccm_min", [1])
    ccm_max = make_input("ccm_max", [1])
    nr_min = make_input("nr_min", [1])
    nr_max = make_input("nr_max", [1])

    # WB normalize
    wb_mean = "wb_mean"
    nodes.append(reduce_mean_vec("raw_wb", wb_mean))
    axes_const = const_axes("wb_mean_axes", [0]); nodes.append(axes_const)
    wb_mean_expand = helper.make_node("Unsqueeze", [wb_mean, "wb_mean_axes"], ["wb_mean_u"]); nodes.append(wb_mean_expand)
    wb_norm = "wb_norm"
    nodes.append(div_vector("raw_wb", "wb_mean_u", wb_norm))

    # WB alpha blend
    nodes += blend_alpha("alpha_wb", "alpha_fast", "scene_change", "alpha_wb_eff")
    nodes += ema_vector(wb_norm, "prev_wb", "alpha_wb_eff", "wb_ema")
    nodes += rate_limit_vector(wb_norm, "prev_wb", "wb_step", "wb_rl")
    half = const_scalar("half", 0.5); nodes.append(half)
    nodes.append(helper.make_node("Add", ["wb_ema", "wb_rl"], ["wb_sum"]))
    nodes.append(helper.make_node("Mul", ["wb_sum", half.output[0]], ["wb_blend"]))
    nodes.append(helper.make_node("Clip", ["wb_blend", "wb_min", "wb_max"], ["wb_stab"]))

    # CCM alpha blend
    nodes += blend_alpha("alpha_ccm", "alpha_fast", "scene_change", "alpha_ccm_eff")
    nodes += ema_vector("raw_ccm", "prev_ccm", "alpha_ccm_eff", "ccm_ema")
    nodes.append(helper.make_node("Clip", ["ccm_ema", "ccm_min", "ccm_max"], ["ccm_stab"]))

    # Gamma alpha blend
    nodes += blend_alpha("alpha_gamma", "alpha_fast", "scene_change", "alpha_gamma_eff")
    nodes += ema_vector("raw_gamma", "prev_gamma", "alpha_gamma_eff", "gamma_ema")
    nodes.append(helper.make_node("Clip", ["gamma_ema", "gamma_min", "gamma_max"], ["gamma_stab"]))

    # Sharpness alpha blend
    gain_min = const_scalar("gain_min", 1.0); nodes.append(gain_min)
    gain_max = const_scalar("gain_max", 16.0); nodes.append(gain_max)
    nodes.append(helper.make_node("Clip", ["analog_gain", gain_min.output[0], gain_max.output[0]], ["noise_factor"]))
    nodes.append(helper.make_node("Div", ["raw_sharpness", "noise_factor"], ["target_sharp"]))
    nodes += blend_alpha("alpha_sharp", "alpha_fast", "scene_change", "alpha_sharp_eff")
    nodes += ema_vector("target_sharp", "prev_sharpness", "alpha_sharp_eff", "sharp_ema")
    nodes += rate_limit_vector("target_sharp", "prev_sharpness", "sharp_step", "sharp_rl")
    nodes.append(helper.make_node("Add", ["sharp_ema", "sharp_rl"], ["sharp_sum"]))
    nodes.append(helper.make_node("Mul", ["sharp_sum", half.output[0]], ["sharp_blend"]))
    nodes.append(helper.make_node("Clip", ["sharp_blend", "gamma_min", "gamma_max"], ["sharpness_stab"]))

    # NR alpha blend
    # -------- NR alpha blend --------
    nodes.append(helper.make_node("Mul", ["raw_nr", "noise_factor"], ["target_nr"]))
    nodes += blend_alpha("alpha_nr", "alpha_fast", "scene_change", "alpha_nr_eff")
    nodes += ema_vector("target_nr", "prev_nr", "alpha_nr_eff", "nr_ema")
    nodes.append(helper.make_node("Clip", ["nr_ema", "nr_min", "nr_max"], ["nr_stab"]))

    # -------- Outputs --------
    out_wb = helper.make_tensor_value_info("wb_stab", TensorProto.FLOAT, [3])
    out_ccm = helper.make_tensor_value_info("ccm_stab", TensorProto.FLOAT, [9])
    out_gamma = helper.make_tensor_value_info("gamma_stab", TensorProto.FLOAT, [gammaN])
    out_sharp = helper.make_tensor_value_info("sharpness_stab", TensorProto.FLOAT, [1])
    out_nr = helper.make_tensor_value_info("nr_stab", TensorProto.FLOAT, [1])

    # Assemble graph
    graph = helper.make_graph(
        nodes,
        "RuleEngineFull",
        [
            raw_wb, raw_ccm, raw_gamma, raw_sharpness, raw_nr,
            prev_wb, prev_ccm, prev_gamma, prev_sharpness, prev_nr,
            analog_gain, exposure_time, sensor_temp, scene_change,
            alpha_wb, alpha_ccm, alpha_gamma, alpha_sharp, alpha_nr, alpha_fast,
            wb_step, sharp_step,
            wb_min, wb_max, gamma_min, gamma_max, ccm_min, ccm_max, nr_min, nr_max,
        ],
        [out_wb, out_ccm, out_gamma, out_sharp, out_nr],
    )

    # Build model with opset 23 for runtime compatibility
    model = helper.make_model(
        graph,
        producer_name="ruleengine_builder",
        opset_imports=[helper.make_operatorsetid("", 23)]
    )

    onnx.checker.check_model(model)
    onnx.save(model, model_name)
    return model

# ---- Export functions ----
def export_ncnn(onnx_file, param_file, bin_file):
    exe = shutil.which("pnnx")
    if exe is None:
        print("pnnx not found. Install with: pip install pnnx")
        return None
    subprocess.run([exe, onnx_file], check=True)

def export_mnn(onnx_file, mnn_file):
    exe = shutil.which("MNNConvert")
    if exe is None:
        print("MNNConvert not found.")
        return None
    subprocess.run(["MNNConvert", "-f", "ONNX", "--modelFile", onnx_file, "--MNNModel", mnn_file], check=True)

# ---- Main ----
if __name__ == "__main__":
    model = build_ruleengine(gammaN=256, model_name="ruleengine_full.onnx")
    print("Saved ruleengine_full.onnx")

    # Generate NCNN and MNN files
    export_ncnn("ruleengine_full.onnx", "ruleengine_full.param", "ruleengine_full.bin")
    export_mnn("ruleengine_full.onnx", "ruleengine_full.mnn")
    print("Exported ruleengine_full.param/bin and ruleengine_full.mnn")
