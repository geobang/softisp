# onnx/microblocks/ccm/ccm_quadratic_v1.py
# -----------------------------------------------------------------------------
# Quadratic CCM generator that inherits from CCMBase.
# Produces a [3,3] CCM tensor `{stage}.ccm` from input CCT using embedded
# quadratic coefficients per matrix entry. Designed to be paired with
# CCMBase.build_applier(), which applies the CCM via MatMul to the image.
#
# Contract:
#   - deps: ['awb_base'] (same as CCMBase)
#   - needs: ['cct']     (this block consumes CCT and provides 'ccm')
#   - provides: ['ccm']  (3x3 matrix for CCMBase.applier)
#
# Design choices:
#   - Quadratic fits per coefficient: c_i(T) = a2_i*T^2 + a1_i*T + a0_i
#     Smooth, interpolation-friendly, and ONNX-embeddable without splines.
#   - Row normalization: preserves luminance (row sums ≈ 1.0).
#   - Fully embedded: coefficients are ONNX initializers; no JSON or external IO.
#   - Audit-friendly: explicit names, value_info, and minimal ops.
# -----------------------------------------------------------------------------

import numpy as np
import onnx.helper as oh
from onnx import TensorProto

from .ccm_base import CCMBase  # relative import alongside ccm_base.py


class CCMQuadraticV1(CCMBase):
    """
    CCMQuadraticV1
    ---------------
    Generates a 3x3 CCM from input CCT (Kelvin) using quadratic fits per
    coefficient. Output tensor is `{stage}.ccm` with shape [3,3], ready for
    CCMBase.build_applier() to consume via MatMul.

    Algorithm:
        For each coefficient c_i:
            c_i(T) = a2_i * T^2 + a1_i * T + a0_i
        Evaluate all 9 coefficients, reshape to [3,3], then normalize rows.

    Inputs:
        - `{stage}.cct` : [1] or [n,1] (Kelvin). Scalar per frame is fine.

    Outputs:
        - `{stage}.ccm` : [3,3] (row-normalized CCM).

    Notes:
        - If upstream provides batched CCT, you can adapt shapes to [n,1] and
          emit [n,3,3]; CCMBase.applier expects [3,3], so we keep it static
          per stage by default.
        - Coefficients are fitted for a smooth Public baseline across 2300–7000K.
        - To add Qcom deltas later, compose a second block and sum matrices
          before normalization.
    """

    name = "ccm_quadratic_v1"
    version = "v1"
    deps = ["wb_avg_v1"]      # mirrors CCMBase
    needs = ["cct"]          # consumes CCT
    provides = ["ccm"]       # provides CCM for CCMBase.applier

    # ---- Embedded quadratic coefficients: [9,3] rows of [a2, a1, a0] ----
    _COEFFS = np.array([
        [ 2.87052802921548e-08, -3.63344500781886e-04,  2.92253209267719],   # c0 (R→R)
        [-1.92068768211296e-08,  3.22622617167509e-04, -2.18213227828730],   # c1 (G→R)
        [-1.04368749046855e-08,  5.07679632871548e-05,  2.34421184318986e-01],# c2 (B→R)

        [-8.93013372263768e-09,  1.27851433852538e-04, -5.44051094414940e-01],# c3 (R→G)
        [-7.25370218484244e-09,  6.16464800677276e-05,  1.10201120343782],   # c4 (G→G)
        [ 1.52453644738198e-08, -1.79451834247488e-04,  4.16860885712545e-01],# c5 (B→G)

        [-2.92219872000960e-08,  3.21012425876420e-04, -7.36574309127542e-01],# c6 (R→B)
        [ 1.11868324612324e-08,  3.95401185937005e-05, -1.54545345239585],   # c7 (G→B)
        [ 1.89740762424375e-08, -3.70598624142898e-04,  3.30720676281451],   # c8 (B→B)
    ], dtype=np.float32)

    def build_algo(self, stage: str, prev_stages=None):
        """
        Build the ONNX nodes that compute `{stage}.ccm` from `{stage}.cct`.

        Returns:
            outputs: dict with {'ccm': {'name': f"{stage}.ccm"}}
            nodes:   list of onnx.helper.make_node(...)
            inits:   list of onnx.helper.make_tensor(...)
            vis:     list of onnx.helper.make_tensor_value_info(...)
        """
        # Names
        out_ccm = f"{stage}.ccm"     # [3,3] CCM consumed by CCMBase.applier
        in_cct = f"{stage}.cct"      # scalar or [1] CCT (Kelvin)

        # Intermediate names
        cct_pow2 = f"{stage}.cct_pow2"       # T^2
        coeffs = f"{stage}.coeffs"           # [9,3] initializer
        a2, a1, a0 = f"{stage}.a2", f"{stage}.a1", f"{stage}.a0"  # [9,1]
        # We’ll evaluate coefficients into a flat [9] vector, then reshape to [3,3]
        ccm_flat = f"{stage}.ccm_flat"       # [9]
        ccm_raw = f"{stage}.ccm_raw"         # [3,3]
        row_sum = f"{stage}.row_sum"         # [3,1]

        nodes, inits, vis = [], [], []

        # ---- Value info (audit-friendly) ----
        # Input CCT: allow scalar or [1]; coordinator can feed a single value per stage.
        vis.append(oh.make_tensor_value_info(in_cct, TensorProto.FLOAT, [1]))
        # Output CCM: [3,3] matrix
        vis.append(oh.make_tensor_value_info(out_ccm, TensorProto.FLOAT, [3, 3]))

        # ---- Initializer: coefficients [9,3] ----
        inits.append(oh.make_tensor(
            name=coeffs,
            data_type=TensorProto.FLOAT,
            dims=list(self._COEFFS.shape),
            vals=self._COEFFS.flatten().tolist()
        ))

        # ---- Split coeffs into a2, a1, a0 (axis=1) ----
        nodes.append(oh.make_node(
            "Split",
            inputs=[coeffs],
            outputs=[a2, a1, a0],
            name=f"{stage}.split_coeffs",
            axis=1,
            split=[1, 1, 1]
        ))
        # Value info for split outputs (optional but helpful)
        vis.append(oh.make_tensor_value_info(a2, TensorProto.FLOAT, [9, 1]))
        vis.append(oh.make_tensor_value_info(a1, TensorProto.FLOAT, [9, 1]))
        vis.append(oh.make_tensor_value_info(a0, TensorProto.FLOAT, [9, 1]))

        # ---- Compute T^2 ----
        nodes.append(oh.make_node(
            "Mul",
            inputs=[in_cct, in_cct],
            outputs=[cct_pow2],
            name=f"{stage}.mul_cct_pow2"
        ))
        vis.append(oh.make_tensor_value_info(cct_pow2, TensorProto.FLOAT, [1]))

        # ---- Evaluate quadratic per coefficient ----
        # We want: c_i = a2_i * T^2 + a1_i * T + a0_i
        # Broadcast rules:
        #   - a2/a1/a0 are [9,1]
        #   - T and T^2 are [1]
        #   ONNX will broadcast [1] across [9,1] → result [9,1]
        term2 = f"{stage}.term2"  # a2 * T^2
        term1 = f"{stage}.term1"  # a1 * T
        tmp_sum = f"{stage}.tmp_sum"

        nodes.append(oh.make_node(
            "Mul",
            inputs=[a2, cct_pow2],
            outputs=[term2],
            name=f"{stage}.mul_term2"
        ))
        nodes.append(oh.make_node(
            "Mul",
            inputs=[a1, in_cct],
            outputs=[term1],
            name=f"{stage}.mul_term1"
        ))
        nodes.append(oh.make_node(
            "Add",
            inputs=[term2, term1],
            outputs=[tmp_sum],
            name=f"{stage}.add_terms"
        ))
        nodes.append(oh.make_node(
            "Add",
            inputs=[tmp_sum, a0],
            outputs=[ccm_flat],
            name=f"{stage}.add_plus_a0"
        ))
        vis.append(oh.make_tensor_value_info(term2, TensorProto.FLOAT, [9, 1]))
        vis.append(oh.make_tensor_value_info(term1, TensorProto.FLOAT, [9, 1]))
        vis.append(oh.make_tensor_value_info(tmp_sum, TensorProto.FLOAT, [9, 1]))
        vis.append(oh.make_tensor_value_info(ccm_flat, TensorProto.FLOAT, [9, 1]))

        # ---- Reshape to [3,3] ----
        # Provide a shape initializer [3,3] for deterministic reshape.
        shape_33 = f"{stage}.shape_33"
        inits.append(oh.make_tensor(
            name=shape_33,
            data_type=TensorProto.INT64,
            dims=[2],
            vals=[3, 3]
        ))
        nodes.append(oh.make_node(
            "Reshape",
            inputs=[ccm_flat, shape_33],
            outputs=[ccm_raw],
            name=f"{stage}.reshape_33"
        ))
        vis.append(oh.make_tensor_value_info(ccm_raw, TensorProto.FLOAT, [3, 3]))

        # ---- Row normalization: divide each row by its sum (axis=1, keepdims=1) ----
        nodes.append(oh.make_node(
            "ReduceSum",
            inputs=[ccm_raw],
            outputs=[row_sum],
            name=f"{stage}.reduce_row_sum",
            axes=[1],
            keepdims=1
        ))
        nodes.append(oh.make_node(
            "Div",
            inputs=[ccm_raw, row_sum],
            outputs=[out_ccm],
            name=f"{stage}.row_normalize"
        ))
        vis.append(oh.make_tensor_value_info(row_sum, TensorProto.FLOAT, [3, 1]))

        # ---- Return contract ----
        outputs = {"ccm": {"name": out_ccm}}
        return outputs, nodes, inits, vis
