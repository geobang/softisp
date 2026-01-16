from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from .tone_base import ToneMapBase


class ToneFilmicV1(ToneMapBase):
    """
    ToneFilmicV1 (v1)
    ------------------
    Self-contained, parameter-light filmic tone mapping.

    Inputs (external):
        - prev_stage.applier : upstream image [n,3,h,w]
        - tonemap_filmic_v1.tone_coeffs : optional override of per-channel coeffs [3,4]

    Outputs:
        - tonemap_filmic_v1.applier     : tone-mapped image [n,3,h,w]
        - tonemap_filmic_v1.tone_coeffs : baseline coeffs [3,4]
        - tonemap_filmic_v1.ci_vec      : channel bias vector [3]
        - tonemap_filmic_v1.ci_scalar   : collapsed bias scalar [1]
    """
    name = "tonemap_filmic_v1"
    version = "v1"
    deps = ["resize_base"]
    needs = ["input_image"]  # upstream image only; coeffs are internal but overrideable
    provides = ["applier", "tone_coeffs", "ci_vec", "ci_scalar"]

    # -------------------------------
    # Trunk 1 — Provide static coeffs
    # -------------------------------
    def _provide_static_coeffs(self, stage, inits, vis):
        """
        Baseline per-channel parametric coeffs:
        [knee, shoulder, gamma, gain] for R,G,B.
        """
        coeffs_name = f"{stage}.tone_coeffs"
        coeffs_vals = [
            0.2, 0.85, 1.90, 1.0,   # R
            0.2, 0.85, 1.85, 1.0,   # G
            0.2, 0.85, 2.00, 1.0,   # B
        ]
        inits.append(
            oh.make_tensor(
                name=coeffs_name,
                data_type=TensorProto.FLOAT,
                dims=[3, 4],
                vals=coeffs_vals,
            )
        )
        vis.append(oh.make_tensor_value_info(coeffs_name, TensorProto.FLOAT, [3, 4]))
        return coeffs_name

    # -------------------------------
    # Trunk 2 — Gather per-channel coeffs
    # -------------------------------
    def _gather_channel_coeffs(self, stage, coeffs_name, ch_idx, nodes):
        """
        Gather [knee, shoulder, gamma, gain] for a given channel index.
        Returns (knee, shoulder, gamma, gain) tensor names.
        """
        base = f"{stage}.ch{ch_idx}"
        knee, shoulder, gamma, gain = (
            f"{base}.knee",
            f"{base}.shoulder",
            f"{base}.gamma",
            f"{base}.gain",
        )
        idx_knee, idx_shoulder, idx_gamma, idx_gain = (
            f"{base}.idx_knee",
            f"{base}.idx_shoulder",
            f"{base}.idx_gamma",
            f"{base}.idx_gain",
        )

        nodes.append(
            oh.make_node(
                "Constant",
                inputs=[],
                outputs=[idx_knee],
                value=oh.make_tensor(idx_knee, TensorProto.INT64, [2], [ch_idx, 0]),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                inputs=[],
                outputs=[idx_shoulder],
                value=oh.make_tensor(idx_shoulder, TensorProto.INT64, [2], [ch_idx, 1]),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                inputs=[],
                outputs=[idx_gamma],
                value=oh.make_tensor(idx_gamma, TensorProto.INT64, [2], [ch_idx, 2]),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                inputs=[],
                outputs=[idx_gain],
                value=oh.make_tensor(idx_gain, TensorProto.INT64, [2], [ch_idx, 3]),
            )
        )

        nodes.append(oh.make_node("GatherND", [coeffs_name, idx_knee], [knee], name=f"{base}.gather_knee"))
        nodes.append(oh.make_node("GatherND", [coeffs_name, idx_shoulder], [shoulder], name=f"{base}.gather_shoulder"))
        nodes.append(oh.make_node("GatherND", [coeffs_name, idx_gamma], [gamma], name=f"{base}.gather_gamma"))
        nodes.append(oh.make_node("GatherND", [coeffs_name, idx_gain], [gain], name=f"{base}.gather_gain"))

        return knee, shoulder, gamma, gain

    # -------------------------------
    # Trunk 3 — Apply tone to one channel
    # -------------------------------
    def _apply_tone_channel(self, stage, x, ch_idx, coeffs_name, nodes, vis):
        """
        Apply parametric filmic tone curve to a single channel tensor x.
        y = min( gain * x^gamma / (1 + knee * x), shoulder )
        """
        knee, shoulder, gamma, gain = self._gather_channel_coeffs(stage, coeffs_name, ch_idx, nodes)
        base = f"{stage}.ch{ch_idx}"

        x_pow   = f"{base}.x_pow"
        knee_x  = f"{base}.knee_x"
        one     = f"{base}.one"
        denom   = f"{base}.denom"
        num     = f"{base}.num"
        y       = f"{base}.y"
        y_clamp = f"{base}.y_clamped"

        nodes.append(oh.make_node("Pow", [x, gamma], [x_pow], name=f"{base}.pow"))
        nodes.append(oh.make_node("Mul", [knee, x], [knee_x], name=f"{base}.mul_knee_x"))
        nodes.append(oh.make_node("Constant", [], [one], value=oh.make_tensor(one, TensorProto.FLOAT, [], [1.0])))
        nodes.append(oh.make_node("Add", [one, knee_x], [denom], name=f"{base}.add_denom"))
        nodes.append(oh.make_node("Mul", [gain, x_pow], [num], name=f"{base}.mul_num"))
        nodes.append(oh.make_node("Div", [num, denom], [y], name=f"{base}.div"))
        nodes.append(oh.make_node("Min", [y, shoulder], [y_clamp], name=f"{base}.min_shoulder"))

        vis.append(oh.make_tensor_value_info(y_clamp, TensorProto.FLOAT, ["n", 1, "h", "w"]))
        return y_clamp

    # -------------------------------
    # Trunk 4 — Apply tone to RGB
    # -------------------------------
    def _apply_tone_rgb(self, stage, input_image, coeffs_name, nodes, vis):
        """
        Split RGB, apply per-channel tone, concat back to applier.
        """
        r, g, b = f"{stage}.r", f"{stage}.g", f"{stage}.b"
        nodes.append(oh.make_node("Split", [input_image], [r, g, b], name=f"{stage}.split_rgb", axis=1))
        vis += [
            oh.make_tensor_value_info(r, TensorProto.FLOAT, ["n", 1, "h", "w"]),
            oh.make_tensor_value_info(g, TensorProto.FLOAT, ["n", 1, "h", "w"]),
            oh.make_tensor_value_info(b, TensorProto.FLOAT, ["n", 1, "h", "w"]),
        ]

        r_t = self._apply_tone_channel(stage, r, 0, coeffs_name, nodes, vis)
        g_t = self._apply_tone_channel(stage, g, 1, coeffs_name, nodes, vis)
        b_t = self._apply_tone_channel(stage, b, 2, coeffs_name, nodes, vis)

        applier = f"{stage}.applier"
        nodes.append(oh.make_node("Concat", [r_t, g_t, b_t], [applier], name=f"{stage}.concat_rgb", axis=1))
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h", "w"]))
        return applier

    # -------------------------------
    # Trunk 5 — CI metrics (vector + scalar)
    # -------------------------------
    def _mean_2d_then_n(self, stage, x, tag, nodes):
        """
        ReduceMean over H,W (keepdims=0), then over N (keepdims=0).
        Returns scalar tensor name.
        """
        m_hw = f"{stage}.{tag}.m_hw"
        m    = f"{stage}.{tag}.m"
        nodes.append(oh.make_node("ReduceMean", [x], [m_hw], name=f"{stage}.{tag}.mean_hw", axes=[2, 3], keepdims=0))
        nodes.append(oh.make_node("ReduceMean", [m_hw], [m], name=f"{stage}.{tag}.mean_n", axes=[0], keepdims=0))
        return m

    def _calc_ci(self, stage, applier, nodes, vis):
        """
        Compute CI vector and scalar:
        - Split channels
        - Mean per channel (over H,W then N)
        - White mean = average of channel means
        - CI vector = abs(mean_rgb - white_mean)
        - CI scalar = mean(ci_vec)
        """
        r, g, b = f"{stage}.ci_r", f"{stage}.ci_g", f"{stage}.ci_b"
        nodes.append(oh.make_node("Split", [applier], [r, g, b], name=f"{stage}.ci_split", axis=1))

        mr = self._mean_2d_then_n(stage, r, "mr", nodes)
        mg = self._mean_2d_then_n(stage, g, "mg", nodes)
        mb = self._mean_2d_then_n(stage, b, "mb", nodes)

        sum_rg, sum_rgb = f"{stage}.sum_rg", f"{stage}.sum_rgb"
        inv3, white_mean = f"{stage}.inv3", f"{stage}.white_mean"

        nodes.append(oh.make_node("Add", [mr, mg], [sum_rg], name=f"{stage}.sum_rg"))
        nodes.append(oh.make_node("Add", [sum_rg, mb], [sum_rgb], name=f"{stage}.sum_rgb"))
        nodes.append(oh.make_node("Constant", [], [inv3], value=oh.make_tensor(inv3, TensorProto.FLOAT, [], [1.0 / 3.0])))
        nodes.append(oh.make_node("Mul", [sum_rgb, inv3], [white_mean], name=f"{stage}.white_mean"))

        ci_r, ci_g, ci_b = f"{stage}.ci_r_vec", f"{stage}.ci_g_vec", f"{stage}.ci_b_vec"
        ci_r_abs, ci_g_abs, ci_b_abs = f"{stage}.ci_r_abs", f"{stage}.ci_g_abs", f"{stage}.ci_b_abs"

        nodes.append(oh.make_node("Sub", [mr, white_mean], [ci_r], name=f"{stage}.sub_r"))
        nodes.append(oh.make_node("Sub", [mg, white_mean], [ci_g], name=f"{stage}.sub_g"))
        nodes.append(oh.make_node("Sub", [mb, white_mean], [ci_b], name=f"{stage}.sub_b"))

        nodes.append(oh.make_node("Abs", [ci_r], [ci_r_abs], name=f"{stage}.abs_r"))
        nodes.append(oh.make_node("Abs", [ci_g], [ci_g_abs], name=f"{stage}.abs_g"))
        nodes.append(oh.make_node("Abs", [ci_b], [ci_b_abs], name=f"{stage}.abs_b"))

        ci_vec = f"{stage}.ci_vec"
        nodes.append(oh.make_node("Concat", [ci_r_abs, ci_g_abs, ci_b_abs], [ci_vec], name=f"{stage}.ci_vec", axis=0))
        vis.append(oh.make_tensor_value_info(ci_vec, TensorProto.FLOAT, [3]))

        ci_scalar = f"{stage}.ci_scalar"
        nodes.append(oh.make_node("ReduceMean", [ci_vec], [ci_scalar], name=f"{stage}.ci_scalar", axes=[0], keepdims=0))
        vis.append(oh.make_tensor_value_info(ci_scalar, TensorProto.FLOAT, [1]))

        return ci_vec, ci_scalar

    # -------------------------------
    # Stitch trunks — build_algo
    # -------------------------------
    def build_algo(self, stage: str, prev_stages=None):
        """
        Orchestrate v1 filmic tone mapping:
        - Pull upstream image
        - Provide static coeffs (overrideable)
        - Apply tone per channel
        - Compute CI metrics
        """
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # Trunk 1: static coeffs
        coeffs_name = self._provide_static_coeffs(stage, inits, vis)

        # Trunk 4: tone RGB
        applier = self._apply_tone_rgb(stage, input_image, coeffs_name, nodes, vis)

        # Trunk 5: CI metrics
        ci_vec, ci_scalar = self._calc_ci(stage, applier, nodes, vis)

        # Outputs
        outputs = {
            "applier":     {"name": applier},
            "tone_coeffs": {"name": coeffs_name},
            "ci_vec":      {"name": ci_vec},
            "ci_scalar":   {"name": ci_scalar},
        }

        # Explicit external inputs:
        # - input_image: dependent (from upstream)
        # - coeffs_name: independent (promote to allow runtime override)
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image)
        result.appendInput(coeffs_name)
        return result
