# onnx/microblocks/ccm/ccm_quadratic_v1.py

import numpy as np
import onnx.helper as oh
from onnx import TensorProto
from .ccm_base import CCMBase


class CCMQuadraticV1(CCMBase):
    """
    CCMQuadraticV1 (v1)
    -------------------
    Generates a 3x3 CCM from input CCT (Kelvin) using quadratic fits per
    coefficient, then applies it to the upstream image (`applier`).
    """

    name = "ccm_quadratic_v1"
    version = "v1"
    deps = ["wb_avg_v1"]
    needs = ["cct", "applier"]
    provides = ["ccm", "applier"]

    _COEFFS = np.array([
        [ 2.87052802921548e-08, -3.63344500781886e-04,  2.92253209267719],
        [-1.92068768211296e-08,  3.22622617167509e-04, -2.18213227828730],
        [-1.04368749046855e-08,  5.07679632871548e-05,  2.34421184318986e-01],

        [-8.93013372263768e-09,  1.27851433852538e-04, -5.44051094414940e-01],
        [-7.25370218484244e-09,  6.16464800677276e-05,  1.10201120343782],
        [ 1.52453644738198e-08, -1.79451834247488e-04,  4.16860885712545e-01],

        [-2.92219872000960e-08,  3.21012425876420e-04, -7.36574309127542e-01],
        [ 1.11868324612324e-08,  3.95401185937005e-05, -1.54545345239585],
        [ 1.89740762424375e-08, -3.70598624142898e-04,  3.30720676281451],
    ], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Sub‑methods
    # -------------------------------------------------------------------------

    def _init_coeffs(self, stage, inits, vis):
        coeffs = f"{stage}.coeffs"
        inits.append(oh.make_tensor(
            name=coeffs,
            data_type=TensorProto.FLOAT,
            dims=list(self._COEFFS.shape),
            vals=self._COEFFS.flatten().tolist()
        ))
        return coeffs

    def _split_coeffs(self, stage, coeffs, inits, nodes, vis):
        split_sizes = f"{stage}.split_sizes"
        inits.append(oh.make_tensor(split_sizes, TensorProto.INT64, [3], [1, 1, 1]))
        coeff_a2, coeff_a1, coeff_a0 = (
            f"{stage}.coeff_a2", f"{stage}.coeff_a1", f"{stage}.coeff_a0"
        )
        nodes.append(oh.make_node(
            "Split",
            inputs=[coeffs, split_sizes],
            outputs=[coeff_a2, coeff_a1, coeff_a0],
            name=f"{stage}.split_coeffs",
            axis=1
        ))
        return coeff_a2, coeff_a1, coeff_a0

    def _compute_terms(self, stage, in_cct, coeff_a2, coeff_a1, coeff_a0, nodes):
        cct_pow2 = f"{stage}.cct_pow2"
        nodes.append(oh.make_node("Mul", [in_cct, in_cct], [cct_pow2], name=f"{stage}.mul_cct_pow2"))

        term2, term1, tmp_sum, ccm_flat = (
            f"{stage}.term2", f"{stage}.term1", f"{stage}.tmp_sum", f"{stage}.ccm_flat"
        )
        nodes.append(oh.make_node("Mul", [coeff_a2, cct_pow2], [term2], name=f"{stage}.mul_term2"))
        nodes.append(oh.make_node("Mul", [coeff_a1, in_cct], [term1], name=f"{stage}.mul_term1"))
        nodes.append(oh.make_node("Add", [term2, term1], [tmp_sum], name=f"{stage}.add_terms"))
        nodes.append(oh.make_node("Add", [tmp_sum, coeff_a0], [ccm_flat], name=f"{stage}.add_plus_a0"))
        return ccm_flat

    def _reshape_and_normalize(self, stage, ccm_flat, inits, nodes):
        shape_33 = f"{stage}.shape_33"
        inits.append(oh.make_tensor(shape_33, TensorProto.INT64, [2], [3, 3]))

        ccm_raw = f"{stage}.ccm_raw"
        row_sum = f"{stage}.row_sum"
        out_ccm = f"{stage}.ccm"

        # Reshape flat [9,1] into [3,3]
        nodes.append(oh.make_node(
            "Reshape",
            inputs=[ccm_flat, shape_33],
            outputs=[ccm_raw],
            name=f"{stage}.reshape_33"
        ))

        # Axes tensor for ReduceSum (axis=1)
        axes_tensor = f"{stage}.axes_row"
        inits.append(oh.make_tensor(
            name=axes_tensor,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]
        ))

        # Row sums
        nodes.append(oh.make_node(
            "ReduceSum",
            inputs=[ccm_raw, axes_tensor],
            outputs=[row_sum],
            name=f"{stage}.reduce_row_sum",
            keepdims=1
        ))

        # Normalize rows
        nodes.append(oh.make_node(
            "Div",
            inputs=[ccm_raw, row_sum],
            outputs=[out_ccm],
            name=f"{stage}.row_normalize"
        ))

        return out_ccm

    def _apply_ccm(self, stage, in_image, out_ccm, nodes):
        out_applier = f"{stage}.applier"
        # MatMul expects [n,3,h,w] reshaped to [n,h*w,3] or similar; here we assume CCMBase handles shape.
        nodes.append(oh.make_node(
            "MatMul",
            inputs=[out_ccm, in_image],
            outputs=[out_applier],
            name=f"{stage}.apply_ccm"
        ))
        return out_applier

    # -------------------------------------------------------------------------
    # Main build_algo
    # -------------------------------------------------------------------------

    def build_algo(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []
        in_cct = f"{stage}.cct"
        upstream = prev_stages[0] if prev_stages else stage
        in_image = f"{upstream}.applier"


        vis.append(oh.make_tensor_value_info(in_cct, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(in_image, TensorProto.FLOAT, ["n", 3, "h", "w"]))

        coeffs = self._init_coeffs(stage, inits, vis)
        coeff_a2, coeff_a1, coeff_a0 = self._split_coeffs(stage, coeffs, inits, nodes, vis)
        ccm_flat = self._compute_terms(stage, in_cct, coeff_a2, coeff_a1, coeff_a0, nodes)
        out_ccm = self._reshape_and_normalize(stage, ccm_flat, inits, nodes)
        out_applier = self._apply_ccm(stage, in_image, out_ccm, nodes)

        vis.append(oh.make_tensor_value_info(out_ccm, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(out_applier, TensorProto.FLOAT, ["n", 3, "h", "w"]))

        outputs = {
            "ccm": {"name": out_ccm},
            "applier": {"name": out_applier}
        }
        return outputs, nodes, inits, vis
