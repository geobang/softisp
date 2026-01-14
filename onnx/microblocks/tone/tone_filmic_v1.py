# onnx/microblocks/tone/tone_filmic_v1.py

import onnx.helper as oh
from onnx import TensorProto
from .tone_base import ToneMapBase


class ToneFilmicV1(ToneMapBase):
    """
    ToneFilmicV1 (self-contained, parameter-free)
    ------------------------------------------------
    - Needs: input_image only (from upstream)
    - Provides: applier (tone-mapped image), tone_coeffs (baseline curve),
                ci_vec (per-channel bias), ci_scalar (collapsed bias)
    - Static parametric curve baked in; coordinator refines later without source RGB
    - Phases:
        0) Provide static coeffs
        1) Apply tone (parametric per channel)
        2) Compute CI (bias indices)
        3) Build orchestration
    """

    name = "tonemap_filmic_v1"
    version = "v1"
    deps = ["resize_base"]
    needs = ["input_image"]
    provides = ["applier", "tone_coeffs", "ci_vec", "ci_scalar"]

    # -----------------------------
    # Phase 0: static curve coeffs
    # -----------------------------
    def _provide_static_coeffs(self, stage, inits, vis):
        """
        Provide baseline per-channel parametric coeffs:
        [knee, shoulder, gamma, gain] for R,G,B.
        Coordinator can refine later; v1 keeps conservative defaults.
        """
        coeffs_name = f"{stage}.tone_coeffs"
        coeffs_vals = [
            # R
            0.20, 0.85, 1.90, 1.00,
            # G
            0.20, 0.85, 1.85, 1.00,
            # B
            0.20, 0.85, 2.00, 1.00,
        ]
        inits.append(oh.make_tensor(
            name=coeffs_name,
            data_type=TensorProto.FLOAT,
            dims=[3, 4],
            vals=coeffs_vals
        ))
        vis.append(oh.make_tensor_value_info(coeffs_name, TensorProto.FLOAT, [3, 4]))
        return coeffs_name

    # -----------------------------
    # Helpers: per-channel coeff gather
    # -----------------------------
    def _gather_channel_coeffs(self, stage, coeffs_name, ch_idx, nodes):
        """
        Gather [knee, shoulder, gamma, gain] for a given channel index.
        Returns names for knee, shoulder, gamma, gain tensors.
        """
        base = f"{stage}.ch{ch_idx}"
        knee = f"{base}.knee"; shoulder = f"{base}.shoulder"
        gamma = f"{base}.gamma"; gain = f"{base}.gain"

        # Indices for [knee, shoulder, gamma, gain]
        idx_knee = f"{base}.idx_knee"; idx_shoulder = f"{base}.idx_shoulder"
        idx_gamma = f"{base}.idx_gamma"; idx_gain = f"{base}.idx_gain"

        nodes.append(oh.make_node("Constant", inputs=[], outputs=[idx_knee],
                                  value=oh.make_tensor(idx_knee, TensorProto.INT64, [2], [ch_idx, 0])))
        nodes.append(oh.make_node("Constant", inputs=[], outputs=[idx_shoulder],
                                  value=oh.make_tensor(idx_shoulder, TensorProto.INT64, [2], [ch_idx, 1])))
        nodes.append(oh.make_node("Constant", inputs=[], outputs=[idx_gamma],
                                  value=oh.make_tensor(idx_gamma, TensorProto.INT64, [2], [ch_idx, 2])))
        nodes.append(oh.make_node("Constant", inputs=[], outputs=[idx_gain],
                                  value=oh.make_tensor(idx_gain, TensorProto.INT64, [2], [ch_idx, 3])))

        nodes.append(oh.make_node("GatherND", inputs=[coeffs_name, idx_knee], outputs=[knee],
                                  name=f"{base}.gather_knee"))
        nodes.append(oh.make_node("GatherND", inputs=[coeffs_name, idx_shoulder], outputs=[shoulder],
                                  name=f"{base}.gather_shoulder"))
        nodes.append(oh.make_node("GatherND", inputs=[coeffs_name, idx_gamma], outputs=[gamma],
                                  name=f"{base}.gather_gamma"))
        nodes.append(oh.make_node("GatherND", inputs=[coeffs_name, idx_gain], outputs=[gain],
                                  name=f"{base}.gather_gain"))

        return knee, shoulder, gamma, gain

    # -----------------------------
    # Helpers: per-channel tone op
    # y = gain * x^gamma / (1 + knee * x), clamped by shoulder
    # -----------------------------
    def _apply_tone_channel(self, stage, x, ch_idx, coeffs_name, nodes, vis):
        """
        Apply parametric tone curve to a single channel tensor x.
        Returns the clamped output tensor name.
        """
        knee, shoulder, gamma, gain = self._gather_channel_coeffs(stage, coeffs_name, ch_idx, nodes)

        base = f"{stage}.ch{ch_idx}"
        x_pow = f"{base}.x_pow"; knee_x = f"{base}.knee_x"
        one = f"{base}.one"; denom = f"{base}.denom"; num = f"{base}.num"; y = f"{base}.y"
        y_clamped = f"{base}.y_clamped"

        nodes.append(oh.make_node("Pow", inputs=[x, gamma], outputs=[x_pow], name=f"{base}.pow"))
        nodes.append(oh.make_node("Mul", inputs=[knee, x], outputs=[knee_x], name=f"{base}.mul_knee_x"))
        nodes.append(oh.make_node("Constant", inputs=[], outputs=[one],
                                  value=oh.make_tensor(one, TensorProto.FLOAT, [], [1.0])))
        nodes.append(oh.make_node("Add", inputs=[one, knee_x], outputs=[denom], name=f"{base}.add_denom"))
        nodes.append(oh.make_node("Mul", inputs=[gain, x_pow], outputs=[num], name=f"{base}.mul_num"))
        nodes.append(oh.make_node("Div", inputs=[num, denom], outputs=[y], name=f"{base}.div"))
        nodes.append(oh.make_node("Min", inputs=[y, shoulder], outputs=[y_clamped], name=f"{base}.min_shoulder"))

        # Optional: declare intermediate shapes if helpful for debugging
        vis.append(oh.make_tensor_value_info(y_clamped, TensorProto.FLOAT, ["n", 1, "h", "w"]))
        return y_clamped

    # -----------------------------
    # Phase 1: apply tone to RGB
    # -----------------------------
    def _apply_tone_rgb(self, stage, input_image, coeffs_name, nodes, vis):
        """
        Split RGB, apply per-channel tone, concat back to applier.
        """
        r = f"{stage}.r"; g = f"{stage}.g"; b = f"{stage}.b"
        nodes.append(oh.make_node("Split", inputs=[input_image], outputs=[r, g, b],
                                  name=f"{stage}.split_rgb", axis=1))
        vis += [
            oh.make_tensor_value_info(r, TensorProto.FLOAT, ["n", 1, "h", "w"]),
            oh.make_tensor_value_info(g, TensorProto.FLOAT, ["n", 1, "h", "w"]),
            oh.make_tensor_value_info(b, TensorProto.FLOAT, ["n", 1, "h", "w"]),
        ]

        r_t = self._apply_tone_channel(stage, r, 0, coeffs_name, nodes, vis)
        g_t = self._apply_tone_channel(stage, g, 1, coeffs_name, nodes, vis)
        b_t = self._apply_tone_channel(stage, b, 2, coeffs_name, nodes, vis)

        applier = f"{stage}.applier"
        nodes.append(oh.make_node("Concat", inputs=[r_t, g_t, b_t], outputs=[applier],
                                  name=f"{stage}.concat_rgb", axis=1))
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h", "w"]))
        return applier

    # -----------------------------
    # Helpers: mean over H,W then N
    # -----------------------------
    def _mean_2d_then_n(self, stage, x, tag, nodes):
        """
        ReduceMean over H,W (keepdims=0), then over N (keepdims=0).
        Returns scalar tensor name.
        """
        m_hw = f"{stage}.{tag}.m_hw"; m = f"{stage}.{tag}.m"
        nodes.append(oh.make_node("ReduceMean", inputs=[x], outputs=[m_hw],
                                  name=f"{stage}.{tag}.mean_hw", axes=[2, 3], keepdims=0))
        nodes.append(oh.make_node("ReduceMean", inputs=[m_hw], outputs=[m],
                                  name=f"{stage}.{tag}.mean_n", axes=[0], keepdims=0))
        return m

    # -----------------------------
    # Phase 2: CI (bias) calculation
    # -----------------------------
    def _calc_ci(self, stage, applier, nodes, vis):
        """
        Compute CI vector and scalar:
        - Split channels
        - Mean per channel (over H,W then N)
        - White mean = average of channel means
        - CI vector = abs(mean_rgb - white_mean)
        - CI scalar = mean(ci_vec)
        """
        r = f"{stage}.ci_r"; g = f"{stage}.ci_g"; b = f"{stage}.ci_b"
        nodes.append(oh.make_node("Split", inputs=[applier], outputs=[r, g, b],
                                  name=f"{stage}.ci_split", axis=1))

        mr = self._mean_2d_then_n(stage, r, "mr", nodes)
        mg = self._mean_2d_then_n(stage, g, "mg", nodes)
        mb = self._mean_2d_then_n(stage, b, "mb", nodes)

        sum_rg = f"{stage}.sum_rg"; sum_rgb = f"{stage}.sum_rgb"; inv3 = f"{stage}.inv3"
        white_mean = f"{stage}.white_mean"
        nodes.append(oh.make_node("Add", inputs=[mr, mg], outputs=[sum_rg], name=f"{stage}.sum_rg"))
        nodes.append(oh.make_node("Add", inputs=[sum_rg, mb], outputs=[sum_rgb], name=f"{stage}.sum_rgb"))
        nodes.append(oh.make_node("Constant", inputs=[], outputs=[inv3],
                                  value=oh.make_tensor(inv3, TensorProto.FLOAT, [], [1.0/3.0])))
        nodes.append(oh.make_node("Mul", inputs=[sum_rgb, inv3], outputs=[white_mean], name=f"{stage}.white_mean"))

        ci_r = f"{stage}.ci_r_vec"; ci_g = f"{stage}.ci_g_vec"; ci_b = f"{stage}.ci_b_vec"
        ci_r_abs = f"{stage}.ci_r_abs"; ci_g_abs = f"{stage}.ci_g_abs"; ci_b_abs = f"{stage}.ci_b_abs"
        nodes.append(oh.make_node("Sub", inputs=[mr, white_mean], outputs=[ci_r], name=f"{stage}.sub_r"))
        nodes.append(oh.make_node("Sub", inputs=[mg, white_mean], outputs=[ci_g], name=f"{stage}.sub_g"))
        nodes.append(oh.make_node("Sub", inputs=[mb, white_mean], outputs=[ci_b], name=f"{stage}.sub_b"))
        nodes.append(oh.make_node("Abs", inputs=[ci_r], outputs=[ci_r_abs], name=f"{stage}.abs_r"))
        nodes.append(oh.make_node("Abs", inputs=[ci_g], outputs=[ci_g_abs], name=f"{stage}.abs_g"))
        nodes.append(oh.make_node("Abs", inputs=[ci_b], outputs=[ci_b_abs], name=f"{stage}.abs_b"))

        ci_vec = f"{stage}.ci_vec"
        nodes.append(oh.make_node("Concat", inputs=[ci_r_abs, ci_g_abs, ci_b_abs], outputs=[ci_vec],
                                  name=f"{stage}.ci_vec", axis=0))
        vis.append(oh.make_tensor_value_info(ci_vec, TensorProto.FLOAT, [3]))

        ci_scalar = f"{stage}.ci_scalar"
        nodes.append(oh.make_node("ReduceMean", inputs=[ci_vec], outputs=[ci_scalar],
                                  name=f"{stage}.ci_scalar", axes=[0], keepdims=0))
        vis.append(oh.make_tensor_value_info(ci_scalar, TensorProto.FLOAT, [1]))

        return ci_vec, ci_scalar

    # -----------------------------
    # Phase 3: build orchestration
    # -----------------------------
    def build_algo(self, stage: str, prev_stages=None):
        """
        Orchestrate v1 self-contained tone mapping:
        - Pull upstream image (consistent with base naming)
        - Provide static coeffs
        - Apply tone per channel
        - Compute CI indices
        - Return outputs, nodes, inits, vis
        """
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"  # upstream output naming convention

        coeffs_name = self._provide_static_coeffs(stage, inits, vis)
        applier = self._apply_tone_rgb(stage, input_image, coeffs_name, nodes, vis)
        ci_vec, ci_scalar = self._calc_ci(stage, applier, nodes, vis)

        outputs = {
            "applier": {"name": applier},
            "tone_coeffs": {"name": coeffs_name},
            "ci_vec": {"name": ci_vec},
            "ci_scalar": {"name": ci_scalar},
        }
        return outputs, nodes, inits, vis
