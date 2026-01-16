from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.wb.wb_base import AWBBase


class WBAvgV1(AWBBase):
    """
    WBAvgV1 (v1)
    -------------
    Direct RGGB averaging white balance block.

    Inputs (external):
        - prev_stage.applier : upstream image tensor [n,3,h,w]
        - wb_avg_v1.ccm      : 3x3 color correction matrix

    Outputs:
        - wb_avg_v1.applier  : CCM-applied image [n,3,h,w]
        - wb_avg_v1.wb_gains : estimated WB gains [3]
        - wb_avg_v1.cct      : correlated color temperature [1]
    """
    name = "wb_avg_v1"
    version = "v1"

    # -------------------------------
    # Trunk 1 — Channel means (R, G_avg, B)
    # -------------------------------
    def _build_channel_means(self, stage, prev_stage, nodes, inits, vis):
        """
        Compute mean R,G,B from upstream stage image.
        """
        in_image = f"{prev_stage}.applier"
        mean_name = f"{stage}.mean_channels"

        # ReduceMean over spatial dims
        nodes.append(
            oh.make_node(
                "ReduceMean",
                inputs=[in_image],
                outputs=[mean_name],
                name=f"{stage}.reduce_mean",
                keepdims=0,
                axes=[2, 3],
            )
        )

        # Split into RGGB channels
        split_const = f"{stage}.split_sizes"
        inits.append(oh.make_tensor(split_const, TensorProto.INT64, [4], [1, 1, 1, 1]))
        r_mean, g1_mean, g2_mean, b_mean = (
            f"{stage}.r_mean",
            f"{stage}.g1_mean",
            f"{stage}.g2_mean",
            f"{stage}.b_mean",
        )
        nodes.append(
            oh.make_node(
                "Split",
                inputs=[mean_name, split_const],
                outputs=[r_mean, g1_mean, g2_mean, b_mean],
                name=f"{stage}.split_rggb",
                axis=0,
            )
        )

        # Average the two greens
        g_mean = f"{stage}.g_mean"
        nodes.append(oh.make_node("Add", [g1_mean, g2_mean], [g_mean], name=f"{stage}.add_g1_g2"))
        half_const = f"{stage}.half_const"
        inits.append(oh.make_tensor(half_const, TensorProto.FLOAT, [], [0.5]))
        g_mean_avg = f"{stage}.g_mean_avg"
        nodes.append(oh.make_node("Mul", [g_mean, half_const], [g_mean_avg], name=f"{stage}.avg_green"))

        # Optional metadata
        vis += [
            oh.make_tensor_value_info(in_image, TensorProto.FLOAT, ["n", 3, "h", "w"]),
            oh.make_tensor_value_info(mean_name, TensorProto.FLOAT, [4]),
            oh.make_tensor_value_info(r_mean, TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(g_mean_avg, TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(b_mean, TensorProto.FLOAT, [1]),
        ]

        return r_mean, g_mean_avg, b_mean

    # -------------------------------
    # Trunk 2 — WB gains + CCT
    # -------------------------------
    def _build_wb_gains(self, stage, nodes, r_mean, g_mean_avg, b_mean):
        """
        Compute WB gains from channel means.
        """
        gain_r, gain_g, gain_b = f"{stage}.gain_r", f"{stage}.gain_g", f"{stage}.gain_b"
        out_wb = f"{stage}.wb_gains"

        nodes.append(oh.make_node("Div", [g_mean_avg, r_mean], [gain_r], name=f"{stage}.gain_r"))
        nodes.append(oh.make_node("Div", [g_mean_avg, g_mean_avg], [gain_g], name=f"{stage}.gain_g"))
        nodes.append(oh.make_node("Div", [g_mean_avg, b_mean], [gain_b], name=f"{stage}.gain_b"))
        nodes.append(
            oh.make_node("Concat", [gain_r, gain_g, gain_b], [out_wb], name=f"{stage}.concat_gains", axis=0)
        )

        return out_wb

    def _build_cct(self, stage, nodes, inits, r_mean, g_mean_avg, b_mean):
        """
        Estimate CCT from chromaticities using McCamy’s formula.
        """
        out_cct = f"{stage}.cct"

        # Chromaticity sums
        sum_rg, sum_rgb = f"{stage}.sum_rg", f"{stage}.sum_rgb"
        nodes.append(oh.make_node("Add", [r_mean, g_mean_avg], [sum_rg], name=f"{stage}.sum_rg"))
        nodes.append(oh.make_node("Add", [sum_rg, b_mean], [sum_rgb], name=f"{stage}.sum_rgb"))

        # Chromaticities
        r_chroma, b_chroma = f"{stage}.r_chroma", f"{stage}.b_chroma"
        nodes.append(oh.make_node("Div", [r_mean, sum_rgb], [r_chroma], name=f"{stage}.r_chroma"))
        nodes.append(oh.make_node("Div", [b_mean, sum_rgb], [b_chroma], name=f"{stage}.b_chroma"))

        # Constants
        const_r, const_b = f"{stage}.const_r", f"{stage}.const_b"
        inits.append(oh.make_tensor(const_r, TensorProto.FLOAT, [], [0.332]))
        inits.append(oh.make_tensor(const_b, TensorProto.FLOAT, [], [0.1858]))

        # Shifts
        r_shift, b_shift = f"{stage}.r_shift", f"{stage}.b_shift"
        nodes.append(oh.make_node("Sub", [r_chroma, const_r], [r_shift], name=f"{stage}.r_shift"))
        nodes.append(oh.make_node("Sub", [b_chroma, const_b], [b_shift], name=f"{stage}.b_shift"))

        # Polynomial terms
        n_ratio, n2, n3 = f"{stage}.n_ratio", f"{stage}.n2", f"{stage}.n3"
        nodes.append(oh.make_node("Div", [r_shift, b_shift], [n_ratio], name=f"{stage}.n_ratio"))
        nodes.append(oh.make_node("Mul", [n_ratio, n_ratio], [n2], name=f"{stage}.n2"))
        nodes.append(oh.make_node("Mul", [n2, n_ratio], [n3], name=f"{stage}.n3"))

        # McCamy coefficients
        coeffs = {"c3": -449.0, "c2": 3525.0, "c1": -6823.3, "c0": 5520.33}
        for cname, val in coeffs.items():
            const = f"{stage}.{cname}_const"
            inits.append(oh.make_tensor(const, TensorProto.FLOAT, [], [val]))

        # Polynomial evaluation
        term3, term2, term1 = f"{stage}.term3", f"{stage}.term2", f"{stage}.term1"
        nodes.append(oh.make_node("Mul", [n3, f"{stage}.c3_const"], [term3], name=f"{stage}.term3"))
        nodes.append(oh.make_node("Mul", [n2, f"{stage}.c2_const"], [term2], name=f"{stage}.term2"))
        nodes.append(oh.make_node("Mul", [n_ratio, f"{stage}.c1_const"], [term1], name=f"{stage}.term1"))

        tmp_sum1, tmp_sum2 = f"{stage}.tmp_sum1", f"{stage}.tmp_sum2"
        nodes.append(oh.make_node("Add", [term3, term2], [tmp_sum1], name=f"{stage}.sum_terms1"))
        nodes.append(oh.make_node("Add", [tmp_sum1, term1], [tmp_sum2], name=f"{stage}.sum_terms2"))
        nodes.append(oh.make_node("Add", [tmp_sum2, f"{stage}.c0_const"], [out_cct], name=f"{stage}.add_c0"))

        return out_cct

    # -------------------------------
    # Trunk 3 — CCM application
    # -------------------------------
    def _build_ccm_apply(self, stage, prev_stage, nodes, vis):
        """
        Apply CCM to upstream image.
        """
        input_image = f"{prev_stage}.applier"
        ccm = f"{stage}.ccm"
        applier = f"{stage}.applier"

        nodes.append(oh.make_node("MatMul", inputs=[input_image, ccm], outputs=[applier], name=f"{stage}_ccm"))

        vis += [
            oh.make_tensor_value_info(ccm, TensorProto.FLOAT, [3, 3]),
            oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h", "w"]),
        ]

        return applier, ccm, input_image

    # -------------------------------
    # Stitch trunks — build_algo
    # -------------------------------
    def build_algo(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []
        upstream = prev_stages[0] if prev_stages else stage

        # Trunk 1: channel means
        r_mean, g_mean_avg, b_mean = self._build_channel_means(stage, upstream, nodes, inits, vis)

        # Trunk 2: WB gains + CCT
        out_wb = self._build_wb_gains(stage, nodes, r_mean, g_mean_avg, b_mean)
        out_cct = self._build_cct(stage, nodes, inits, r_mean, g_mean_avg, b_mean)
        vis += [
            oh.make_tensor_value_info(out_wb, TensorProto.FLOAT, [3]),
            oh.make_tensor_value_info(out_cct, TensorProto.FLOAT, [1]),
        ]

        # Trunk 3: CCM application
        applier, ccm, input_image = self._build_ccm_apply(stage, upstream, nodes, vis)

        # Outputs
        outputs = {
            "applier": {"name": applier},
            "wb_gains": {"name": out_wb},
            "cct": {"name": out_cct},
        }

        # Explicit external inputs for checker-safe wiring
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image)  # dependent input from upstream
        result.appendInput(ccm)          # independent CCM matrix
        return result
