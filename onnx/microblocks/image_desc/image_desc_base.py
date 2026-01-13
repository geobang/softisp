import onnx.helper as oh
from microblocks.base import MicroblockBase

class ImageDescBase(MicroblockBase):
    """
    Pseudo microblock that anchors image metadata.
    Declares image [n,c,h,stride] and width as graph inputs.
    Each input is re-emitted by an Identity node so outputs are node-backed.
    """

    name = "image_desc_base"
    version = "v0"
    deps = []
    needs = ["image", "width"]

    def build_applier(self, stage: str, prev_stages=None):
        # Graph input names
        image_in = f"{stage}.image_in"
        width_in = f"{stage}.width_in"

        # Node-backed outputs
        image_out = f"{stage}.applier"
#        width_out = f"{stage}.width"

        # Identity nodes to anchor inputs
        nodes = [
            oh.make_node("Identity", inputs=[image_in], outputs=[image_out], name=f"{stage}_image_id"),
#            oh.make_node("Identity", inputs=[width_in], outputs=[width_out], name=f"{stage}_width_id"),
        ]

        # Value infos (declared graph inputs)
        vis = [
            # Image tensor: [n,c,h,stride]
            oh.make_tensor_value_info(image_in, oh.TensorProto.FLOAT, ["n","c","h","stride"]),
            # Active width metadata
#            oh.make_tensor_value_info(width_in, oh.TensorProto.INT64, [1]),
            # Outputs (node-backed)
            oh.make_tensor_value_info(image_out, oh.TensorProto.FLOAT, ["n","c","h","stride"]),
#            oh.make_tensor_value_info(width_out, oh.TensorProto.INT64, [1]),
        ]

        outputs = {
            "image": {"name": image_out},
#            "width": {"name": width_out},
        }

        return outputs, nodes, [], vis
