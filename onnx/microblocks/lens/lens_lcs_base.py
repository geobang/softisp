# onnx/microblocks/lens/lens_lcs_base.py

import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase

class LensLCSBase(MicroblockBase):
    """
    LensLCSBase (v0)
    ----------------
    Canonical base microblock for Lens Correction & Shading (LCS).

    Needs:
        - applier [n,3,h,w] : image tensor from upstream
        - lcs_coeffs [h,w]  : per-pixel correction coefficients

    Provides:
        - applier [n,3,h,w] : corrected image tensor

    Behavior:
        - build_algo: declares lcs_coeffs as an external need (no generation here)
        - build_applier: multiplies applier × lcs_coeffs (broadcasted across channels)
    """

    name = "lens_lcs_base"
    version = "v0"
    needs = ["applier", "lcs_coeffs"]
    provides = ["applier"]

    def build_algo(self, stage: str, prev_stages=None):
        """
        Base version does not generate lcs_coeffs.
        Assumes lcs_coeffs are provided externally (from coordinator or calibration).
        """
        vis, nodes, inits = [], [], []
        coeffs_name = f"{stage}.lcs_coeffs"

        vis.append(
            oh.make_tensor_value_info(coeffs_name, TensorProto.FLOAT, ["h", "w"])
        )
        outputs = {"lcs_coeffs": {"name": coeffs_name}}
        return outputs, nodes, inits, vis

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply lens correction coefficients to applier (image).
        """
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        lcs_coeffs = f"{upstream}.lcs_coeffs"

        applier = f"{stage}.applier"
        nodes.append(
            oh.make_node(
                "Mul",
                inputs=[input_image, lcs_coeffs],
                outputs=[applier],
                name=f"{stage}.mul_apply"
            )
        )

        vis.append(
            oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h", "w"])
        )
        outputs = {"applier": {"name": applier}}
        return outputs, nodes, inits, vis
