import onnx.helper as oh
from microblocks.base import MicroblockBase
import onnx.helper as oh

class ImageDescBase(MicroblockBase):
    name = "image_desc_base"
    version = "v0"
    deps = []
    needs = []

    def build_applier(self, stage: str, prev_stages=None):
        outputs = {
            "image": {"name": f"{stage}.image"},
            "h": {"name": f"{stage}.h"},
            "w": {"name": f"{stage}.w"},
            "c": {"name": f"{stage}.c"},
            "stride": {"name": f"{stage}.stride"},
        }

        node = oh.make_node(
            "Identity",
            inputs=["input_image"],
            outputs=[f"{stage}.image"],
            name=f"{stage}_identity"
        )

        vis = [
            oh.make_tensor_value_info("input_image", oh.TensorProto.FLOAT, ["N","C","H","W"]),
            oh.make_tensor_value_info(f"{stage}.image", oh.TensorProto.FLOAT, ["N","C","H","W"]),
            oh.make_tensor_value_info(f"{stage}.h", oh.TensorProto.INT64, [1]),
            oh.make_tensor_value_info(f"{stage}.w", oh.TensorProto.INT64, [1]),
            oh.make_tensor_value_info(f"{stage}.c", oh.TensorProto.INT64, [1]),
            oh.make_tensor_value_info(f"{stage}.stride", oh.TensorProto.INT64, [1]),
        ]

        return outputs, [node], [], vis
