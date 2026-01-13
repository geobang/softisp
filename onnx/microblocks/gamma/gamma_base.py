import onnx.helper as oh
from microblocks.base import MicroblockBase

class GammaBase(MicroblockBase):
    """
    Gamma correction microblock.
    Input:  [n,3,target_h,target_w] (RGB)
    Output: [n,3,target_h,target_w] (gamma corrected RGB)
    Needs:  gamma_value (scalar float, e.g. 2.2)
    """

    name = "gamma_base"
    version = "v0"
    deps = ["tonemap_base"]
    needs = ["gamma_value"]

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f"{stage}.applier"
        gamma    = f"{stage}.gamma_value"

        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # Gamma correction: output = input^(1/gamma)
        node = oh.make_node(
            "Pow",
            inputs=[input_image, gamma],
            outputs=[out_name],
            name=f"{stage}_gamma"
        )

        vis = [
            oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
            oh.make_tensor_value_info(gamma, oh.TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
        ]

        outputs = {"applier": {"name": out_name}}
        return outputs, [node], [], vis
