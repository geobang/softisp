import onnx.helper as oh
from onnx import TensorProto
from .resize_base import ResizeBase

class ResizeV1(ResizeBase):
    name = "resize_v1"
    version = "v1"
    needs = [""]
    provides = ["resize_factor"]

    def build_algo(self, stage: str, prev_stages=None):
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        rgb = f"{upstream}.applier"

        applier = f"{stage}.applier"
        scales = f"{stage}.scales"
        inits.append(oh.make_tensor(scales, TensorProto.FLOAT, [4], [1.0, 1.0, 0.5, 0.5]))

        resize_factor = f"{stage}.resize_factor"
        inits.append(oh.make_tensor(resize_factor, TensorProto.FLOAT, [], [0.5]))
        vis.append(oh.make_tensor_value_info(resize_factor, TensorProto.FLOAT, []))

        nodes.append(
            oh.make_node("Resize", inputs=[rgb, scales], outputs=[applier],
                         name=f"{stage}.resize", mode="linear")
        )
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h/2", "w/2"]))

        outputs = {
            "applier": {"name": applier},
            "resize_factor": {"name": resize_factor},
        }
        return outputs, nodes, inits, vis

    def build_applier(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages=prev_stages)
