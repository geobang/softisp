from microblocks.base import BuildResult
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

    Inputs (external):
        - prev_stage.applier : upstream image tensor [n,3,h,w]
        - ccm_quadratic_v1.cct : scalar CCT [1]

    Outputs:
        - ccm_quadratic_v1.ccm     : normalized 3x3 CCM
        - ccm_quadratic_v1.applier : CCM-applied image [n,3,h,w]
    """
    name = "ccm_quadratic_v1"
    version = "v1"

    # Pre-fit quadratic coefficients for each CCM element (flattened 9x3: a2,a1,a0)
    _COEFFS = np.array(
        [
            [2.87052802921548e-08, -0.000363344500781886, 2.92253209267719],
            [-1.92068768211296e-08, 0.000322622617167509, -2.1821322782873],
            [-1.04368749046855e-08, 5.07679632871548e-05, 0.234421184318986],
            [-8.93013372263768e-09, 0.000127851433852538, -0.54405109441494],
            [-7.25370218484244e-09, 6.16464800677276e-05, 1.10201120343782],
            [1.52453644738198e-08, -0.000179451834247488, 0.416860885712545],
            [-2.9221987200096e-08, 0.00032101242587642, -0.736574309127542],
            [1.11868324612324e-08, 3.95401185937005e-05, -1.54545345239585],
            [1.89740762424375e-08, -0.000370598624142898, 3.30720676281451],
        ],
        dtype=np.float32,
    )

    # -------------------------------
    # Trunk 1 — Initialize and split quadratic coefficients
    # -------------------------------
    def _init_coeffs(self, stage, inits, vis):
        """
        Initialize the 9x3 coefficient table [a2, a1, a0] and expose as initializer.
        """
        coeffs = f"{stage}.coeffs"
        inits.append(
            oh.make_tensor(
                name=coeffs,
                data_type=TensorProto.FLOAT,
                dims=list(self._COEFFS.shape),
                vals=self._COEFFS.flatten().tolist(),
            )
        )
        vis.append(oh.make_tensor_value_info(coeffs, TensorProto.FLOAT, [9, 3]))
        return coeffs

    def _split_coeffs(self, stage, coeffs, inits, nodes, vis):
        """
        Split the 9x3 table into three 9x1 vectors: a2, a1, a0.
        """
        split_sizes = f"{stage}.split_sizes"
        inits.append(oh.make_tensor(split_sizes, TensorProto.INT64, [3], [1, 1, 1]))
        coeff_a2, coeff_a1, coeff_a0 = (
            f"{stage}.coeff_a2",
            f"{stage}.coeff_a1",
            f"{stage}.coeff_a0",
        )
        nodes.append(
            oh.make_node(
                "Split",
                inputs=[coeffs, split_sizes],
                outputs=[coeff_a2, coeff_a1, coeff_a0],
                name=f"{stage}.split_coeffs",
                axis=1,
            )
        )
        vis += [
            oh.make_tensor_value_info(coeff_a2, TensorProto.FLOAT, [9, 1]),
            oh.make_tensor_value_info(coeff_a1, TensorProto.FLOAT, [9, 1]),
            oh.make_tensor_value_info(coeff_a0, TensorProto.FLOAT, [9, 1]),
        ]
        return coeff_a2, coeff_a1, coeff_a0

    # -------------------------------
    # Trunk 2 — Evaluate quadratic per element and normalize to 3x3 CCM
    # -------------------------------
    def _compute_terms(self, stage, in_cct, coeff_a2, coeff_a1, coeff_a0, nodes, vis):
        """
        Evaluate a2*CCT^2 + a1*CCT + a0 → 9x1 flattened CCM.
        """
        cct_pow2 = f"{stage}.cct_pow2"
        nodes.append(oh.make_node("Mul", [in_cct, in_cct], [cct_pow2], name=f"{stage}.mul_cct_pow2"))

        term2, term1, tmp_sum, ccm_flat = (
            f"{stage}.term2",
            f"{stage}.term1",
            f"{stage}.tmp_sum",
            f"{stage}.ccm_flat",
        )
        nodes.append(oh.make_node("Mul", [coeff_a2, cct_pow2], [term2], name=f"{stage}.mul_term2"))
        nodes.append(oh.make_node("Mul", [coeff_a1, in_cct], [term1], name=f"{stage}.mul_term1"))
        nodes.append(oh.make_node("Add", [term2, term1], [tmp_sum], name=f"{stage}.add_terms"))
        nodes.append(oh.make_node("Add", [tmp_sum, coeff_a0], [ccm_flat], name=f"{stage}.add_plus_a0"))

        vis += [
            oh.make_tensor_value_info(in_cct, TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(ccm_flat, TensorProto.FLOAT, [9, 1]),
        ]
        return ccm_flat

    def _reshape_and_normalize(self, stage, ccm_flat, inits, nodes, vis):
        """
        Reshape 9x1 → 3x3 and row-normalize CCM.
        """
        shape_33 = f"{stage}.shape_33"
        inits.append(oh.make_tensor(shape_33, TensorProto.INT64, [2], [3, 3]))

        ccm_raw = f"{stage}.ccm_raw"
        row_sum = f"{stage}.row_sum"
        out_ccm = f"{stage}.ccm"

        nodes.append(oh.make_node("Reshape", [ccm_flat, shape_33], [ccm_raw], name=f"{stage}.reshape_33"))

        axes_tensor = f"{stage}.axes_row"
        inits.append(oh.make_tensor(name=axes_tensor, data_type=TensorProto.INT64, dims=[1], vals=[1]))
        nodes.append(
            oh.make_node(
                "ReduceSum",
                inputs=[ccm_raw, axes_tensor],
                outputs=[row_sum],
                name=f"{stage}.reduce_row_sum",
                keepdims=1,
            )
        )
        nodes.append(oh.make_node("Div", [ccm_raw, row_sum], [out_ccm], name=f"{stage}.row_normalize"))

        vis += [
            oh.make_tensor_value_info(ccm_raw, TensorProto.FLOAT, [3, 3]),
            oh.make_tensor_value_info(out_ccm, TensorProto.FLOAT, [3, 3]),
        ]
        return out_ccm

    # -------------------------------
    # Trunk 3 — Apply CCM to upstream image
    # -------------------------------
    def _apply_ccm(self, stage, in_image, out_ccm, nodes, vis):
        """
        Apply CCM to upstream image: applier = MatMul(image, CCM).
        """
        out_applier = f"{stage}.applier"
        nodes.append(
            oh.make_node("MatMul", inputs=[in_image, out_ccm], outputs=[out_applier], name=f"{stage}.apply_ccm")
        )
        vis.append(oh.make_tensor_value_info(out_applier, TensorProto.FLOAT, ["n", 3, "h", "w"]))
        return out_applier

    # -------------------------------
    # Stitch trunks — build_algo
    # -------------------------------
    def build_algo(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []
        upstream = prev_stages[0] if prev_stages else stage

        # External inputs
        in_cct = f"{stage}.cct"
        in_image = f"{upstream}.applier"
        vis += [
            oh.make_tensor_value_info(in_cct, TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(in_image, TensorProto.FLOAT, ["n", 3, "h", "w"]),
        ]

        # Trunk 1: init & split coefficients
        coeffs = self._init_coeffs(stage, inits, vis)
        coeff_a2, coeff_a1, coeff_a0 = self._split_coeffs(stage, coeffs, inits, nodes, vis)

        # Trunk 2: evaluate quadratic and normalize CCM
        ccm_flat = self._compute_terms(stage, in_cct, coeff_a2, coeff_a1, coeff_a0, nodes, vis)
        out_ccm = self._reshape_and_normalize(stage, ccm_flat, inits, nodes, vis)

        # Trunk 3: apply CCM
        out_applier = self._apply_ccm(stage, in_image, out_ccm, nodes, vis)

        # Outputs
        outputs = {"ccm": {"name": out_ccm}, "applier": {"name": out_applier}}

        # Explicit external inputs for checker-safe wiring
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(in_image)  # dependent upstream image
        result.appendInput(in_cct)    # independent CCT scalar
        return result
