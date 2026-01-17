# bayer_to_float.py
from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class BayerToFloat(MicroblockBase):
    """
    BayerToFloat Microblock
    Input:  [n,1,h,w] raw Bayer mosaic (uint)
    Output: [n,1,h,w] float32 tensor (values unchanged)
    Purpose: Convert integer sensor values to float for downstream math ops.
    """

    name = "bayer2float32_base"
    version = "v0"
    provides = ["applier"]

    def build_algo(self, stage, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        out_name    = f"{stage}.applier"

        # Cast node: int → float
        nodes = [
            oh.make_node("Cast", [input_image], [out_name],
                         name=f"{stage}_cast_to_float",
                         to=TensorProto.FLOAT)
        ]

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.UINT16, ["n",1,"h","w"]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT,  ["n",1,"h","w"]),
        ]

        outputs = {"applier": {"name": out_name}}
        result = BuildResult(outputs, nodes, [], vis)
        result.appendInput(input_image)
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_applier(self, stage, prev_stages=None):
        return self.build_algo(stage, prev_stages)
