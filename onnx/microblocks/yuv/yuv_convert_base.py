import onnx.helper as oh
from microblocks.base import MicroblockBase

class YUVConvertBase(MicroblockBase):
    """
    RGB → YUV conversion microblock.
    Input:  [n,3,target_h,target_w] (RGB)
    Output: [n,3,target_h,target_w] (YUV)
    Needs:  rgb2yuv_matrix [3,3]
    """

    name = "yuvconvert_base"
    version = "v0"
    deps = ["tonemap_base"]
    needs = ["rgb2yuv_matrix"]

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f"{stage}.applier"
        matrix   = f"{stage}.rgb2yuv_matrix"

        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        node = oh.make_node(
            "MatMul",
            inputs=[input_image, matrix],
            outputs=[out_name],
            name=f"{stage}_yuvconvert"
        )

        vis = [
            oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
            oh.make_tensor_value_info(matrix, oh.TensorProto.FLOAT, [3,3]),
            oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
        ]

        outputs = {"applier": {"name": out_name}}
        return outputs, [node], [], vis

    def build_algo(self, stage: str, prev_stages=None):
        """Declare or initialize rgb2yuv_matrix for this stage."""
        nodes, inits, vis = [], [], []

        matrix = f"{stage}.rgb2yuv_matrix"

        # Value info: 3x3 conversion matrix
        vis.append(oh.make_tensor_value_info(matrix, oh.TensorProto.FLOAT, [3,3]))

        # Optionally initialize with a standard RGB→YUV conversion matrix
        # BT.601 full‑range example:
        default_matrix = [
            0.299,  0.587,  0.114,
           -0.147, -0.289,  0.436,
            0.615, -0.515, -0.100
        ]
        inits.append(
            oh.make_tensor(matrix, oh.TensorProto.FLOAT, [3,3], default_matrix)
        )

        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        out_name    = f"{stage}.applier"

        # Identity node to forward input → output
        nodes.append(oh.make_node(
            "Identity",
            inputs=[input_image],
            outputs=[out_name],
            name=f"{stage}.identity"
        ))

        vis += [
            oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
            oh.make_tensor_value_info(out_name,   oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
        ]

        # Explicitly add the applier item to outputs
        outputs = {
            "applier": {"name": out_name},
            "rgb2yuv_matrix": {"name": matrix},
        }
        return outputs, nodes, inits, vis
