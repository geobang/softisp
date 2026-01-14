# onnx/microblocks/wb/wb_avg_v1.py
# -----------------------------------------------------------------------------
# Child class of WBBase that implements shortcut AWB:
#   - Direct RGGB averaging
#   - WB gains from channel means
#   - Optional CCT estimation
# -----------------------------------------------------------------------------

# in wb_avg_v1.py
import onnx.helper as oh
from onnx import TensorProto
from microblocks.wb.wb_base import AWBBase

class WBAvgV1(AWBBase):
    """
    WBAvgV1
    -------
    Direct RGGB averaging white balance block.
    """

    name = "wb_avg_v1"
    version = "v1"
    deps = []
    needs = ["applier"]
    provides = ["wb_gains", "cct"]

    # -------------------------------------------------------------------------
    # Sub‑methods
    # -------------------------------------------------------------------------

    def _build_channel_means(self, stage, nodes, inits, vis):
        """Compute mean R,G,B from RGGB image."""
        in_image = f"{stage}.applier"
        mean_name = f"{stage}.mean_channels"

        nodes.append(oh.make_node(
            "ReduceMean", [in_image], [mean_name],
            name=f"{stage}.reduce_mean", axes=[2, 3], keepdims=0
        ))
        vis.append(oh.make_tensor_value_info(mean_name, TensorProto.FLOAT, [4]))

        # Split RGGB → R,G,B
        r_mean, g1_mean, g2_mean, b_mean = (
            f"{stage}.r_mean", f"{stage}.g1_mean", f"{stage}.g2_mean", f"{stage}.b_mean"
        )
        nodes.append(oh.make_node(
            "Split", [mean_name],
            [r_mean, g1_mean, g2_mean, b_mean],
            name=f"{stage}.split_rggb", axis=0, split=[1,1,1,1]
        ))

        # Average greens
        g_mean = f"{stage}.g_mean"
        nodes.append(oh.make_node("Add", [g1_mean, g2_mean], [g_mean], name=f"{stage}.add_g1_g2"))
        half_const = f"{stage}.half_const"
        inits.append(oh.make_tensor(half_const, TensorProto.FLOAT, [], [0.5]))
        g_mean_avg = f"{stage}.g_mean_avg"
        nodes.append(oh.make_node("Mul", [g_mean, half_const], [g_mean_avg], name=f"{stage}.avg_green"))

        return r_mean, g_mean_avg, b_mean

    def _build_wb_gains(self, stage, nodes, r_mean, g_mean_avg, b_mean):
        """Compute WB gains from channel means."""
        gain_r, gain_g, gain_b = (
            f"{stage}.gain_r", f"{stage}.gain_g", f"{stage}.gain_b"
        )
        out_wb = f"{stage}.wb_gains"

        nodes.append(oh.make_node("Div", [g_mean_avg, r_mean], [gain_r], name=f"{stage}.gain_r"))
        nodes.append(oh.make_node("Div", [g_mean_avg, g_mean_avg], [gain_g], name=f"{stage}.gain_g"))
        nodes.append(oh.make_node("Div", [g_mean_avg, b_mean], [gain_b], name=f"{stage}.gain_b"))

        nodes.append(oh.make_node(
            "Concat", [gain_r, gain_g, gain_b], [out_wb],
            name=f"{stage}.concat_gains", axis=0
        ))
        return out_wb

    def _build_cct(self, stage, nodes, inits, r_mean, g_mean_avg, b_mean):
        """Estimate CCT from chromaticities using McCamy’s formula."""
        out_cct = f"{stage}.cct"

        # Sum channels
        sum_rgb = f"{stage}.sum_rgb"
        nodes.append(oh.make_node("Add", [r_mean, g_mean_avg], [sum_rgb], name=f"{stage}.sum_rg"))
        nodes.append(oh.make_node("Add", [sum_rgb, b_mean], [sum_rgb], name=f"{stage}.sum_rgb"))

        # Chromaticities
        r_chroma, b_chroma = f"{stage}.r_chroma", f"{stage}.b_chroma"
        nodes.append(oh.make_node("Div", [r_mean, sum_rgb], [r_chroma], name=f"{stage}.r_chroma"))
        nodes.append(oh.make_node("Div", [b_mean, sum_rgb], [b_chroma], name=f"{stage}.b_chroma"))

        # n = (r-0.3320)/(b-0.1858)
        const_r, const_b = f"{stage}.const_r", f"{stage}.const_b"
        inits.append(oh.make_tensor(const_r, TensorProto.FLOAT, [], [0.3320]))
        inits.append(oh.make_tensor(const_b, TensorProto.FLOAT, [], [0.1858]))
        r_shift, b_shift = f"{stage}.r_shift", f"{stage}.b_shift"
        nodes.append(oh.make_node("Sub", [r_chroma, const_r], [r_shift], name=f"{stage}.r_shift"))
        nodes.append(oh.make_node("Sub", [b_chroma, const_b], [b_shift], name=f"{stage}.b_shift"))
        n_ratio = f"{stage}.n_ratio"
        nodes.append(oh.make_node("Div", [r_shift, b_shift], [n_ratio], name=f"{stage}.n_ratio"))

        # Polynomial fit
        n2, n3 = f"{stage}.n2", f"{stage}.n3"
        nodes.append(oh.make_node("Mul", [n_ratio, n_ratio], [n2], name=f"{stage}.n2"))
        nodes.append(oh.make_node("Mul", [n2, n_ratio], [n3], name=f"{stage}.n3"))

        coeffs = {"c3": -449.0, "c2": 3525.0, "c1": -6823.3, "c0": 5520.33}
        for cname, val in coeffs.items():
            const = f"{stage}.{cname}_const"
            inits.append(oh.make_tensor(const, TensorProto.FLOAT, [], [val]))

        term3, term2, term1 = f"{stage}.term3", f"{stage}.term2", f"{stage}.term1"
        nodes.append(oh.make_node("Mul", [n3, f"{stage}.c3_const"], [term3], name=f"{stage}.term3"))
        nodes.append(oh.make_node("Mul", [n2, f"{stage}.c2_const"], [term2], name=f"{stage}.term2"))
        nodes.append(oh.make_node("Mul", [n_ratio, f"{stage}.c1_const"], [term1], name=f"{stage}.term1"))

        tmp_sum1, tmp_sum2 = f"{stage}.tmp_sum1", f"{stage}.tmp_sum2"
        nodes.append(oh.make_node("Add", [term3, term2], [tmp_sum1], name=f"{stage}.sum_terms1"))
        nodes.append(oh.make_node("Add", [tmp_sum1, term1], [tmp_sum2], name=f"{stage}.sum_terms2"))
        nodes.append(oh.make_node("Add", [tmp_sum2, f"{stage}.c0_const"], [out_cct], name=f"{stage}.add_c0"))

        return out_cct

    # -------------------------------------------------------------------------
    # Main build_algo
    # -------------------------------------------------------------------------

    def build_algo(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []

        # Value info for outputs
        vis.append(oh.make_tensor_value_info(f"{stage}.wb_gains", TensorProto.FLOAT, [3]))
        vis.append(oh.make_tensor_value_info(f"{stage}.cct", TensorProto.FLOAT, [1]))

        # Sub‑methods
        r_mean, g_mean_avg, b_mean = self._build_channel_means(stage, nodes, inits, vis)
        out_wb = self._build_wb_gains(stage, nodes, r_mean, g_mean_avg, b_mean)
        out_cct = self._build_cct(stage, nodes, inits, r_mean, g_mean_avg, b_mean)

        outputs = {"wb_gains": {"name": out_wb}, "cct": {"name": out_cct}}
        return outputs, nodes, inits, vis
