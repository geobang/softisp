import onnx
import onnx.helper as oh
from microblocks.base import MicroblockBase

class LcsBase(MicroblockBase):
    """
    Lens correction (LCS) microblock.
    Input:  [n,3,h,width] (RGB from CCM)
    Needs:  lcs_gain_map [1,3,h,width] (per-channel gain factors)
    Output: [n,3,h,width] (lens-corrected RGB)
    """

    name = "lcs_base"
    version = "v0"
    deps = ["ccm_base"]
    needs = ["lcs_gain_map"]

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f"{stage}.applier"
        gain_map = f"{stage}.lcs_gain_map"

        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # Elementwise lens correction: RGB * gain_map
        node = oh.make_node(
            "Mul",
            inputs=[input_image, gain_map],
            outputs=[out_name],
            name=f"{stage}_lcs"
        )

        vis = [
            oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ["n", "3", "h", "width"]),
            oh.make_tensor_value_info(gain_map,   oh.TensorProto.FLOAT, [1, "3", "h", "width"]),
            oh.make_tensor_value_info(out_name,   oh.TensorProto.FLOAT, ["n", "3", "h", "width"]),
        ]

        outputs = {"applier": {"name": out_name}}
        return outputs, [node], [], vis
