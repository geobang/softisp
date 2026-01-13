import onnx.helper as oh
from microblocks.base import MicroblockBase

class ToneMapBase(MicroblockBase):
    """
    Tone Mapping microblock.
    Input:  [n,3,target_h,target_w] (RGB)
    Output: [n,3,target_h,target_w] (tone-mapped RGB)
    Needs:  tonemap_curve (lookup table or curve parameters)
    """

    name = "tonemap_base"
    version = "v0"
    deps = ["resize_base"]
    needs = ["tonemap_curve"]

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f"{stage}.applier"
        curve    = f"{stage}.tonemap_curve"

        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        node = oh.make_node(
            "Mul",  # placeholder: real tone mapping often uses LUT or nonlinear ops
            inputs=[input_image, curve],
            outputs=[out_name],
            name=f"{stage}_tonemap"
        )

        vis = [
            oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
            oh.make_tensor_value_info(curve, oh.TensorProto.FLOAT, [1]),  # placeholder for LUT/curve
            oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
        ]

        outputs = {"applier": {"name": out_name}}
        return outputs, [node], [], vis
