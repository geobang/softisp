from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from .lens_lcs_base import LensLCSBase

class LensLCSV1(LensLCSBase):
    """
    LensLCSV1 (v1)
    --------------
    Adaptive lens shading correction block.

    Inputs (external):
        - prev_stage.applier : upstream image tensor [n,3,h,w]
        - lens_lcs_v1.lcs_coeffs : full-resolution correction coefficients [H,W]
        - prev_stage.resize_factor : scalar resize factor []

    Outputs:
        - lens_lcs_v1.applier : corrected image tensor [n,3,h*,w*]
        - lens_lcs_v1.lcs_coeffs_resized : resized coefficient map [h*,w*]
        - lens_lcs_v1.lcs_coeffs_out : identity copy of original coeffs [H,W]
    """
    name = "lens_lcs_v1"
    version = "v1"

    def build_algo(self, stage: str, prev_stages=None):
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage

        # External names
        input_image   = f"{upstream}.applier"
        resize_factor = f"{upstream}.resize_factor"
        lcs_coeffs    = f"{stage}.lcs_coeffs"

        # Constants for batch/channel scales
        one_n, one_c = f"{stage}.one_n", f"{stage}.one_c"
        inits += [
            oh.make_tensor(one_n, TensorProto.FLOAT, [], [1.0]),
            oh.make_tensor(one_c, TensorProto.FLOAT, [], [1.0]),
        ]

        # Build scales vector [1,1,resize_factor,resize_factor]
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
        vis.append(oh.make_tensor_value_info(scales, TensorProto.FLOAT, [4]))

        # Resize lcs_coeffs
        lcs_resized = f"{stage}.lcs_coeffs_resized"
        nodes.append(
            oh.make_node(
                "Resize",
                inputs=[lcs_coeffs, "", scales],  # roi left empty
                outputs=[lcs_resized],
                name=f"{stage}.resize_lcs",
                mode="linear",
            )
        )
        vis.append(oh.make_tensor_value_info(lcs_resized, TensorProto.FLOAT, ["h*", "w*"]))

        # Identity node to expose original coeffs
        lcs_coeffs_out = f"{stage}.lcs_coeffs_out"
        nodes.append(
            oh.make_node("Identity", inputs=[lcs_coeffs], outputs=[lcs_coeffs_out], name=f"{stage}.identity_lcs")
        )
        vis.append(oh.make_tensor_value_info(lcs_coeffs_out, TensorProto.FLOAT, ["H", "W"]))

        # Apply correction
        applier = f"{stage}.applier"
        nodes.append(
            oh.make_node("Mul", inputs=[input_image, lcs_resized], outputs=[applier], name=f"{stage}.algo_mul_apply")
        )
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h*", "w*"]))

        # Outputs
        outputs = {
            "applier": {"name": applier},
            "lcs_coeffs_resized": {"name": lcs_resized},
            "lcs_coeffs": {"name": lcs_coeffs_out},
        }

        # BuildResult with explicit inputs
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image)
        result.appendInput(lcs_coeffs)
        result.appendInput(resize_factor)
        return result

    def build_applier(self, stage: str, prev_stages=None):
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage

        input_image = f"{upstream}.applier"
        lcs_coeffs  = f"{stage}.lcs_coeffs"
        applier     = f"{stage}.applier"

        # Apply correction directly
        nodes.append(
            oh.make_node("Mul", inputs=[input_image, lcs_coeffs], outputs=[applier], name=f"{stage}.mul_apply")
        )

        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", 3, "h*", "w*"]),
            oh.make_tensor_value_info(lcs_coeffs, TensorProto.FLOAT, ["h*", "w*"]),
            oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h*", "w*"]),
        ]

        outputs = {"applier": {"name": applier}}

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image)
        result.appendInput(lcs_coeffs)
        return result
