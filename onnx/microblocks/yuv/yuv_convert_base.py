from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class YUVConvertBase(MicroblockBase):
    """
    RGB → YUV conversion microblock.
    Input:  [n,3,target_h,target_w] (RGB)
    Output: [n,3,target_h,target_w] (YUV)
    Needs:  rgb2yuv_matrix [3,3]
    """
    name = 'yuvconvert_base'
    version = 'v0'
    deps = ['tonemap_base']
    needs = ['rgb2yuv_matrix']
    provides = ['applier', 'rgb2yuv_matrix']

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f'{stage}.applier'
        matrix = f'{stage}.rgb2yuv_matrix'
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'

        # Apply RGB→YUV conversion
        node = oh.make_node(
            'MatMul',
            inputs=[input_image, matrix],
            outputs=[out_name],
            name=f'{stage}_yuvconvert'
        )

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']),
            oh.make_tensor_value_info(matrix, TensorProto.FLOAT, [3, 3]),
            oh.make_tensor_value_info(out_name, TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w'])
        ]

        outputs = {'applier': {'name': out_name}}
        return BuildResult(outputs, [node], [], vis).appendInput(f'{upstream}.applier')

    def build_algo(self, stage: str, prev_stages=None):
        """Declare rgb2yuv_matrix with default, override, and SSA-safe output."""
        nodes, inits, vis = ([], [], [])

        # Stage-scoped matrix name
        matrix = f'{stage}.rgb2yuv_matrix'

        # 1) Default matrix as initializer
        default_matrix = [
            0.299, 0.587, 0.114,
            -0.147, -0.289, 0.436,
            0.615, -0.515, -0.100
        ]
        inits.append(oh.make_tensor(matrix, TensorProto.FLOAT, [3, 3], default_matrix))

        # 2) Promote to graph input (so runtime can override)
        vis.append(oh.make_tensor_value_info(matrix, TensorProto.FLOAT, [3, 3]))

        # 3) Identity node to expose rgb2yuv_matrix as visible output (distinct name)
        matrix_out = f'{stage}.rgb2yuv_matrix_out'
        nodes.append(
            oh.make_node("Identity", inputs=[matrix], outputs=[matrix_out], name=f'{stage}.matrix_identity')
        )
        vis.append(oh.make_tensor_value_info(matrix_out, TensorProto.FLOAT, [3, 3]))

        # 4) Pass-through image (algo stage doesn’t apply conversion)
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        out_name = f'{stage}.applier'
        nodes.append(
            oh.make_node('Identity', inputs=[input_image], outputs=[out_name], name=f'{stage}.identity')
        )
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']))
        vis.append(oh.make_tensor_value_info(out_name,   TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']))

        # 5) Outputs: image + replicated matrix_out
        outputs = {
            'applier':        {'name': out_name},
            'rgb2yuv_matrix': {'name': matrix_out},
        }

        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{upstream}.applier')
