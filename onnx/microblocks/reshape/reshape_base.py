# reshape.py
from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class ReshapeBase(MicroblockBase):
    """
    Reshape Base Class
    Input:  [h,w] raw Bayer array
    Output: [n,c,h,w] tensor (default [1,1,h,w])
    """

    name = "reshape"
    version = "v0"
    provides = ["applier"]

    # Default target shape (subclasses override if needed)
    target_shape = [1,1,"h","w"]

    def build_algo(self, stage, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        out_name    = f"{stage}.applier"

        shape_const = f"{stage}.shape"

        nodes = [
            oh.make_node("Reshape", [input_image, shape_const],
                         [out_name], name=f"{stage}_reshape")
        ]

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["h","w"]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ["n","c","h","w"]),
        ]

        # Initializer for target shape
        init = [
            oh.make_tensor(shape_const, TensorProto.INT64, [len(self.target_shape)],
                           [1,1,-1,-1])  # -1 lets ONNX infer h,w
        ]

        outputs = {"applier": {"name": out_name}}
        result = BuildResult(outputs, nodes, init, vis)
        result.appendInput(input_image)
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_applier(self, stage, prev_stages=None):
        return self.build_algo(stage, prev_stages)
