from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto


class GammaBase(MicroblockBase):
    """
    Gamma correction microblock.
    Input:  [n,3,target_h,target_w] (RGB)
    Output: [n,3,target_h,target_w] (gamma corrected RGB)
    Optional: gamma_value (scalar float, e.g. 2.2)
    """
    name = "gamma_base"
    version = "v0"
    deps = ["tonemap_base"]
    needs = []  # no mandatory needs
    provides = ["applier", "gamma_value"]

    # -------------------------------
    # Applier (runtime gamma correction)
    # -------------------------------
    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        gamma = f"{stage}.gamma_value"
        out_name = f"{stage}.applier"

        # Apply gamma correction: Pow(input_image, gamma)
        node = oh.make_node("Pow", [input_image, gamma], [out_name], name=f"{stage}_gamma")

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", "3", "target_h", "target_w"]),
            oh.make_tensor_value_info(gamma, TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(out_name, TensorProto.FLOAT, ["n", "3", "target_h", "target_w"]),
        ]

        outputs = {"applier": {"name": out_name}}

        result = BuildResult(outputs, [node], [], vis)
        result.appendInput(input_image)  # upstream image only
        result.appendInput(gamma)
        return result

    # -------------------------------
    # Algo (declare gamma + pass-through image)
    # -------------------------------
    def build_algo(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # Internal gamma parameter with default
        gamma = f"{stage}.gamma_value"
        inits.append(oh.make_tensor(gamma, TensorProto.FLOAT, [1], [2.2]))
        vis.append(oh.make_tensor_value_info(gamma, TensorProto.FLOAT, [1]))

        # Identity to expose gamma as visible output
        gamma_out = f"{stage}.gamma_value_out"
        nodes.append(oh.make_node("Identity", [gamma], [gamma_out], name=f"{stage}.gamma_identity"))
        vis.append(oh.make_tensor_value_info(gamma_out, TensorProto.FLOAT, [1]))

        # Pass-through image
        out_name = f"{stage}.applier"
        nodes.append(oh.make_node("Identity", [input_image], [out_name], name=f"{stage}.identity"))
        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", "3", "target_h", "target_w"]),
            oh.make_tensor_value_info(out_name,   TensorProto.FLOAT, ["n", "3", "target_h", "target_w"]),
        ]

        outputs = {
            "applier":     {"name": out_name},
            "gamma_value": {"name": gamma_out},  # visible output
        }

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image)  # upstream image only
        # gamma is optional: do not appendInput here
        return result
