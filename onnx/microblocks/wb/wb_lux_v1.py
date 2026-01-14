# onnx/microblocks/wb/wb_lux_v1.py

import onnx.helper as oh
from onnx import TensorProto
from .wb_base import AWBBase


class AWBLuxV1(AWBBase):
    """
    AWBLuxV1
    --------
    - Needs: input_image [n,3,h,w], lux_scalar []
    - Provides: wb_gains [3], applier [n,3,h,w]
    - Behavior:
        * Computes per-channel means (gray-world proxy) from pseudo-demosaic RGB
        * Derives gains using white_mean normalization, modulated by lux
        * Applies gains to input_image to produce AWB applier
    - Notes:
        * Keeps applier contract and channel-wise Mul consistent with AWBBase
        * Gains generation is internal; coordinator can refine later
    """

    name = "awb_lux_v1"
    version = "v1"
    deps = ["demosaic_avg_lux_v1"]
    needs = ["input_image", "lux_scalar"]
    provides = ["wb_gains", "applier"]

    # -----------------------------
    # Phase 1: split RGB channels
    # -----------------------------
    def _split_rgb(self, stage, input_image, nodes, vis):
        r = f"{stage}.r"
        g = f"{stage}.g"
        b = f"{stage}.b"
        nodes.append(
            oh.make_node(
                "Split",
                inputs=[input_image],
                outputs=[r, g, b],
                name=f"{stage}.split_rgb",
                axis=1,
            )
        )
        vis += [
            oh.make_tensor_value_info(r, TensorProto.FLOAT, ["n", 1, "h", "w"]),
            oh.make_tensor_value_info(g, TensorProto.FLOAT, ["n", 1, "h", "w"]),
            oh.make_tensor_value_info(b, TensorProto.FLOAT, ["n", 1, "h", "w"]),
        ]
        return r, g, b

    # -----------------------------
    # Phase 2: per-channel means
    # -----------------------------
    def _mean_2d_then_n(self, stage, x, tag, nodes):
        m_hw = f"{stage}.{tag}.m_hw"
        m = f"{stage}.{tag}.m"
        nodes.append(
            oh.make_node(
                "ReduceMean",
                inputs=[x],
                outputs=[m_hw],
                name=f"{stage}.{tag}.mean_hw",
                axes=[2, 3],
                keepdims=0,
            )
        )
        nodes.append(
            oh.make_node(
                "ReduceMean",
                inputs=[m_hw],
                outputs=[m],
                name=f"{stage}.{tag}.mean_n",
                axes=[0],
                keepdims=0,
            )
        )
        return m

    def _compute_channel_means(self, stage, r, g, b, nodes, vis):
        mr = self._mean_2d_then_n(stage, r, "mr", nodes)
        mg = self._mean_2d_then_n(stage, g, "mg", nodes)
        mb = self._mean_2d_then_n(stage, b, "mb", nodes)
        vis += [
            oh.make_tensor_value_info(mr, TensorProto.FLOAT, []),
            oh.make_tensor_value_info(mg, TensorProto.FLOAT, []),
            oh.make_tensor_value_info(mb, TensorProto.FLOAT, []),
        ]
        return mr, mg, mb

    # -----------------------------
    # Phase 3: derive gains from means + lux
    # -----------------------------
    def _derive_gains(self, stage, mr, mg, mb, lux_scalar, nodes, vis, inits):
        # White mean = average of channel means
        sum_rg = f"{stage}.sum_rg"
        sum_rgb = f"{stage}.sum_rgb"
        inv3 = f"{stage}.inv3"
        white_mean = f"{stage}.white_mean"

        inits.append(oh.make_tensor(inv3, TensorProto.FLOAT, [], [1.0 / 3.0]))
        nodes.append(oh.make_node("Add", inputs=[mr, mg], outputs=[sum_rg], name=f"{stage}.sum_rg"))
        nodes.append(oh.make_node("Add", inputs=[sum_rg, mb], outputs=[sum_rgb], name=f"{stage}.sum_rgb"))
        nodes.append(oh.make_node("Mul", inputs=[sum_rgb, inv3], outputs=[white_mean], name=f"{stage}.white_mean"))
        vis.append(oh.make_tensor_value_info(white_mean, TensorProto.FLOAT, []))

        # Base gains = white_mean / mean_channel
        eps = f"{stage}.eps"
        inits.append(oh.make_tensor(eps, TensorProto.FLOAT, [], [1e-6]))

        mr_eps = f"{stage}.mr_eps"
        mg_eps = f"{stage}.mg_eps"
        mb_eps = f"{stage}.mb_eps"
        nodes.append(oh.make_node("Add", inputs=[mr, eps], outputs=[mr_eps], name=f"{stage}.add_mr_eps"))
        nodes.append(oh.make_node("Add", inputs=[mg, eps], outputs=[mg_eps], name=f"{stage}.add_mg_eps"))
        nodes.append(oh.make_node("Add", inputs=[mb, eps], outputs=[mb_eps], name=f"{stage}.add_mb_eps"))

        gr = f"{stage}.gain_r_base"
        gg = f"{stage}.gain_g_base"
        gb = f"{stage}.gain_b_base"
        nodes.append(oh.make_node("Div", inputs=[white_mean, mr_eps], outputs=[gr], name=f"{stage}.div_gr"))
        nodes.append(oh.make_node("Div", inputs=[white_mean, mg_eps], outputs=[gg], name=f"{stage}.div_gg"))
        nodes.append(oh.make_node("Div", inputs=[white_mean, mb_eps], outputs=[gb], name=f"{stage}.div_gb"))

        # Lux modulation: blend toward unity at high lux, allow stronger correction at low lux
        # gains = mix(unity, base_gains, alpha), alpha = clamp(1 - k * lux, 0, 1)
        k = f"{stage}.lux_k"
        one = f"{stage}.one"
        zero = f"{stage}.zero"
        inits += [
            oh.make_tensor(k, TensorProto.FLOAT, [], [0.7]),   # sensitivity
            oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]),
            oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]),
        ]

        k_lux = f"{stage}.k_lux"
        one_minus_k_lux = f"{stage}.one_minus_k_lux"
        alpha_raw = f"{stage}.alpha_raw"
        alpha = f"{stage}.alpha"

        nodes.append(oh.make_node("Mul", inputs=[k, lux_scalar], outputs=[k_lux], name=f"{stage}.mul_k_lux"))
        nodes.append(oh.make_node("Sub", inputs=[one, k_lux], outputs=[one_minus_k_lux], name=f"{stage}.sub_one_k_lux"))
        # Clamp alpha to [0,1]
        nodes.append(oh.make_node("Max", inputs=[one_minus_k_lux, zero], outputs=[alpha_raw], name=f"{stage}.max_alpha"))
        nodes.append(oh.make_node("Min", inputs=[alpha_raw, one], outputs=[alpha], name=f"{stage}.min_alpha"))
        vis.append(oh.make_tensor_value_info(alpha, TensorProto.FLOAT, []))

        # Mix: gains = alpha * base + (1 - alpha) * 1
        one_r = f"{stage}.one_r"
        one_g = f"{stage}.one_g"
        one_b = f"{stage}.one_b"
        inits += [
            oh.make_tensor(one_r, TensorProto.FLOAT, [], [1.0]),
            oh.make_tensor(one_g, TensorProto.FLOAT, [], [1.0]),
            oh.make_tensor(one_b, TensorProto.FLOAT, [], [1.0]),
        ]

        alpha_gr = f"{stage}.alpha_gr"
        alpha_gg = f"{stage}.alpha_gg"
        alpha_gb = f"{stage}.alpha_gb"
        nodes.append(oh.make_node("Mul", inputs=[alpha, gr], outputs=[alpha_gr], name=f"{stage}.mul_alpha_gr"))
        nodes.append(oh.make_node("Mul", inputs=[alpha, gg], outputs=[alpha_gg], name=f"{stage}.mul_alpha_gg"))
        nodes.append(oh.make_node("Mul", inputs=[alpha, gb], outputs=[alpha_gb], name=f"{stage}.mul_alpha_gb"))

        inv_alpha = f"{stage}.inv_alpha"
        nodes.append(oh.make_node("Sub", inputs=[one, alpha], outputs=[inv_alpha], name=f"{stage}.sub_inv_alpha"))

        inv_alpha_r = f"{stage}.inv_alpha_r"
        inv_alpha_g = f"{stage}.inv_alpha_g"
        inv_alpha_b = f"{stage}.inv_alpha_b"
        nodes.append(oh.make_node("Mul", inputs=[inv_alpha, one_r], outputs=[inv_alpha_r], name=f"{stage}.mul_inv_alpha_r"))
        nodes.append(oh.make_node("Mul", inputs=[inv_alpha, one_g], outputs=[inv_alpha_g], name=f"{stage}.mul_inv_alpha_g"))
        nodes.append(oh.make_node("Mul", inputs=[inv_alpha, one_b], outputs=[inv_alpha_b], name=f"{stage}.mul_inv_alpha_b"))

        gain_r = f"{stage}.gain_r"
        gain_g = f"{stage}.gain_g"
        gain_b = f"{stage}.gain_b"
        nodes.append(oh.make_node("Add", inputs=[alpha_gr, inv_alpha_r], outputs=[gain_r], name=f"{stage}.add_gain_r"))
        nodes.append(oh.make_node("Add", inputs=[alpha_gg, inv_alpha_g], outputs=[gain_g], name=f"{stage}.add_gain_g"))
        nodes.append(oh.make_node("Add", inputs=[alpha_gb, inv_alpha_b], outputs=[gain_b], name=f"{stage}.add_gain_b"))

        # Pack gains into [3] for application
        wb_gains = f"{stage}.wb_gains"
        nodes.append(
            oh.make_node(
                "Concat",
                inputs=[gain_r, gain_g, gain_b],
                outputs=[wb_gains],
                name=f"{stage}.concat_wb_gains",
                axis=0,
            )
        )
        vis.append(oh.make_tensor_value_info(wb_gains, TensorProto.FLOAT, [3]))
        return wb_gains

    # -----------------------------
    # Phase 4: apply gains (AWBBase-compatible)
    # -----------------------------
    def _apply_gains(self, stage, input_image, wb_gains, nodes, vis):
        applier = f"{stage}.applier"
        nodes.append(
            oh.make_node(
                "Mul",
                inputs=[input_image, wb_gains],
                outputs=[applier],
                name=f"{stage}_awb_mul",
            )
        )
        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", 3, "h", "w"]),
            oh.make_tensor_value_info(wb_gains, TensorProto.FLOAT, [3]),
            oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h", "w"]),
        ]
        return applier

    # -----------------------------
    # Main build orchestration
    # -----------------------------
    def build_algo(self, stage: str, prev_stages=None):
        """
        Orchestrate AWB with lux:
        - Split RGB
        - Compute per-channel means
        - Derive lux-modulated gains
        - Apply gains to produce applier
        """
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        lux_scalar = f"{upstream}.lux_scalar"

        # Phase 1: split
        r, g, b = self._split_rgb(stage, input_image, nodes, vis)

        # Phase 2: means
        mr, mg, mb = self._compute_channel_means(stage, r, g, b, nodes, vis)

        # Phase 3: gains
        wb_gains = self._derive_gains(stage, mr, mg, mb, lux_scalar, nodes, vis, inits)

        # Phase 4: apply
        applier = self._apply_gains(stage, input_image, wb_gains, nodes, vis)

        outputs = {
            "wb_gains": {"name": wb_gains},
            "applier": {"name": applier},
        }
        return outputs, nodes, inits, vis
