from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto


class YUVConvertBase(MicroblockBase):
    """
    RGB → YUV conversion microblock.
    Input:  [n,3,target_h,target_w] (RGB)
    Output: [n,3,target_h,target_w] (YUV)
    Optional: rgb2yuv_matrix [3,3]
    """
    name = "yuvconvert_base"
    version = "v0"

    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        matrix = f"{stage}.rgb2yuv_matrix"
        out_name = f"{stage}.applier"

        # Apply RGB→YUV conversion
        node = oh.make_node("MatMul", [input_image, matrix], [out_name], name=f"{stage}_yuvconvert")

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", "3", "target_h", "target_w"]),
            oh.make_tensor_value_info(matrix, TensorProto.FLOAT, [3, 3]),
            oh.make_tensor_value_info(out_name, TensorProto.FLOAT, ["n", "3", "target_h", "target_w"]),
        ]

        outputs = {"applier": {"name": out_name}}

        result = BuildResult(outputs, [node], [], vis)
        result.appendInput(input_image)  # upstream image only
        # matrix is optional: do not appendInput here
        result.appendInput(matrix)
        return result

    def build_algo(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # Internal matrix parameter with default
        matrix = f"{stage}.rgb2yuv_matrix"
        default_matrix = [
            0.299, 0.587, 0.114,
            -0.147, -0.289, 0.436,
            0.615, -0.515, -0.100,
        ]
        inits.append(oh.make_tensor(matrix, TensorProto.FLOAT, [3, 3], default_matrix))
        vis.append(oh.make_tensor_value_info(matrix, TensorProto.FLOAT, [3, 3]))

        # Identity to expose matrix as visible output
        matrix_out = f"{stage}.rgb2yuv_matrix_out"
        nodes.append(oh.make_node("Identity", [matrix], [matrix_out], name=f"{stage}.matrix_identity"))
        vis.append(oh.make_tensor_value_info(matrix_out, TensorProto.FLOAT, [3, 3]))

        # Pass-through image
        out_name = f"{stage}.applier"
        nodes.append(oh.make_node("Identity", [input_image], [out_name], name=f"{stage}.identity"))
        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", "3", "target_h", "target_w"]),
            oh.make_tensor_value_info(out_name,   TensorProto.FLOAT, ["n", "3", "target_h", "target_w"]),
        ]

        outputs = {
            "applier":        {"name": out_name},
            "rgb2yuv_matrix": {"name": matrix_out},  # visible output
        }

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image)  # upstream image only
        # matrix is optional: do not appendInput here
        return result
