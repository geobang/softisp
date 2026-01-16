from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto


class ToneMapBase(MicroblockBase):
    """
    ToneMapBase (v0)
    ----------------
    Minimal tone mapping block.

    Inputs:
        - prev_stage.applier : upstream image [n,3,h,w]
        - tonemap_base.tonemap_curve : scalar or LUT [1] (overrideable)

    Outputs:
        - tonemap_base.applier       : tone-mapped image [n,3,h,w]
        - tonemap_base.tonemap_curve : visible curve output [1]
    """
    name = "tonemap_base"
    version = "v0"
    deps = ["resize_base"]
    needs = ["tonemap_curve"]
    provides = ["applier", "tonemap_curve"]

    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        curve = f"{stage}.tonemap_curve"
        out_name = f"{stage}.applier"

        # Simplified tone map: multiply by curve scalar
        node = oh.make_node("Mul", [input_image, curve], [out_name], name=f"{stage}_tonemap")

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", 3, "h", "w"]),
            oh.make_tensor_value_info(curve, TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(out_name, TensorProto.FLOAT, ["n", 3, "h", "w"]),
        ]

        outputs = {"applier": {"name": out_name}}

        result = BuildResult(outputs, [node], [], vis)
        result.appendInput(input_image)  # upstream image
        result.appendInput(curve)        # independent curve
        return result

    def build_algo(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # Internal curve parameter
        curve = f"{stage}.tonemap_curve"
        inits.append(oh.make_tensor(curve, TensorProto.FLOAT, [1], [0.8]))
        vis.append(oh.make_tensor_value_info(curve, TensorProto.FLOAT, [1]))

        # Identity to expose curve as distinct output
        curve_out = f"{stage}.tonemap_curve_out"
        nodes.append(oh.make_node("Identity", [curve], [curve_out], name=f"{stage}.curve_identity"))
        vis.append(oh.make_tensor_value_info(curve_out, TensorProto.FLOAT, [1]))

        # Pass-through image
        out_name = f"{stage}.applier"
        nodes.append(oh.make_node("Identity", [input_image], [out_name], name=f"{stage}.identity"))
        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", 3, "h", "w"]),
            oh.make_tensor_value_info(out_name, TensorProto.FLOAT, ["n", 3, "h", "w"]),
        ]

        outputs = {
            "applier": {"name": out_name},
            "tonemap_curve": {"name": curve_out},  # visible output
        }

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image)  # upstream image
#        result.appendInput(curve)        # independent curve (not curve_out!)
        return result
