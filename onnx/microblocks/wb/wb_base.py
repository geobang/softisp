import onnx.helper as oh
from microblocks.base import MicroblockBase

class AWBBase(MicroblockBase):
    """
    Auto White Balance microblock.
    Input:  [n,3,h,width] (RGB)
    Output: [n,3,h,width] (RGB after channel gains)
    Needs:  wb_gains [3] (R,G,B multipliers)
    """

    name = "awb_base"
    version = "v0"
    deps = ["demosaic_base"]
    needs = ["wb_gains"]

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f"{stage}.applier"
        gains    = f"{stage}.wb_gains"

        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        node = oh.make_node(
            "Mul",
            inputs=[input_image, gains],
            outputs=[out_name],
            name=f"{stage}_awb"
        )

        vis = [
            oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ["n","3","h","width"]),
            oh.make_tensor_value_info(gains, oh.TensorProto.FLOAT, [3]),
            oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ["n","3","h","width"]),
        ]

        outputs = {"applier": {"name": out_name}}
        return outputs, [node], [], vis
