# onnx/microblocks/lens/lens_lcs_v1.py

import onnx.helper as oh
from onnx import TensorProto
from .lens_lcs_base import LensLCSBase


class LensLCSV1(LensLCSBase):
    """
    LensLCSV1
    ---------
    Inherits LensLCSBase and extends it with resize support.

    Needs:
        - input_image [n,3,h,w]
        - lcs_coeffs [H,W]
        - resize_factor []

    Provides:
        - applier [n,3,h*resize_factor,w*resize_factor]
        - lcs_coeffs_resized [h*resize_factor,w*resize_factor]

    Behavior:
        - build_algo: resizes lcs_coeffs according to resize_factor
        - build_applier: reuses LensLCSBase multiplication logic, but applies resized coeffs
    """

    name = "lens_lcs_v1"
    version = "v1"
    needs = ["input_image", "lcs_coeffs", "resize_factor"]
    provides = ["applier", "lcs_coeffs_resized"]

    def build_algo(self, stage: str, prev_stages=None):
        """
        Resize lcs_coeffs according to resize_factor.
        """
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        lcs_coeffs = f"{upstream}.lcs_coeffs"
        resize_factor = f"{upstream}.resize_factor"

        # scales = [1.0, 1.0, resize_factor, resize_factor]
        one_n = f"{stage}.one_n"
        one_c = f"{stage}.one_c"
        inits += [
            oh.make_tensor(one_n, TensorProto.FLOAT, [], [1.0]),
            oh.make_tensor(one_c, TensorProto.FLOAT, [], [1.0]),
        ]

        scales = f"{stage}.scales"
        nodes.append(
            oh.make_node(
                "Concat",
                inputs=[one_n, one_c, resize_factor, resize_factor],
                outputs=[scales],
                name=f"{stage}.concat_scales",
                axis=0,
            )
        )

        lcs_resized = f"{stage}.lcs_coeffs_resized"
        nodes.append(
            oh.make_node(
                "Resize",
                inputs=[lcs_coeffs, "", scales],
                outputs=[lcs_resized],
                name=f"{stage}.resize_lcs",
                mode="linear",
            )
        )

        vis.append(
            oh.make_tensor_value_info(
                lcs_resized, TensorProto.FLOAT, ["h*", "w*"]
            )
        )
        outputs = {"lcs_coeffs_resized": {"name": lcs_resized}}
        return outputs, nodes, inits, vis

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply resized lcs_coeffs to input_image.
        Reuses LensLCSBase logic but swaps in lcs_coeffs_resized.
        """
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.input_image"
        lcs_resized = f"{upstream}.lcs_coeffs_resized"

        applier = f"{stage}.applier"
        nodes.append(
            oh.make_node(
                "Mul",
                inputs=[input_image, lcs_resized],
                outputs=[applier],
                name=f"{stage}.mul_apply",
            )
        )
        vis.append(
            oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h*", "w*"])
        )
        outputs = {"applier": {"name": applier}}
        return outputs, nodes, inits, vis
