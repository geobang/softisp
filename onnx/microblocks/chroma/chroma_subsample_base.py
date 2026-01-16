from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class ChromaSubsampleBase(MicroblockBase):
    """
    Chroma subsampling microblock.
    Input:  [n,3,target_h,target_w] (YUV)
    Output: [n,3,target_h,target_w/2] (YUV 4:2:0)
    Needs:  subsample_scale [4] (scale factors for N,C,H,W)
    """
    name = 'chroma_subsample_base'
    version = 'v0'
    deps = ['yuvconvert_base']
    needs = ['subsample_scale']
    provides = ['applier', 'subsample_scale']

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f'{stage}.applier'
        scale = f'{stage}.subsample_scale'
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'

        # Apply chroma subsampling using Resize
        node = oh.make_node(
            'Resize',
            inputs=[input_image, '', scale],
            outputs=[out_name],
            name=f'{stage}_chroma',
            mode='nearest'
        )

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']),
            oh.make_tensor_value_info(scale, TensorProto.FLOAT, [4]),
            oh.make_tensor_value_info(out_name, TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w/2'])
        ]

        outputs = {'applier': {'name': out_name}}
        return BuildResult(outputs, [node], [], vis).appendInput(f'{upstream}.applier')

    def build_algo(self, stage: str, prev_stages=None):
        """Declare subsample_scale with default, override, and SSA-safe output."""
        nodes, inits, vis = ([], [], [])

        # Stage-scoped scale name
        scale = f'{stage}.subsample_scale'

        # 1) Default scale as initializer
        default_scale = [1.0, 1.0, 1.0, 0.5]
        inits.append(oh.make_tensor(scale, TensorProto.FLOAT, [4], default_scale))

        # 2) Promote to graph input (so runtime can override)
        vis.append(oh.make_tensor_value_info(scale, TensorProto.FLOAT, [4]))

        # 3) Identity node to expose subsample_scale as visible output (distinct name)
        scale_out = f'{stage}.subsample_scale_out'
        nodes.append(
            oh.make_node("Identity", inputs=[scale], outputs=[scale_out], name=f'{stage}.scale_identity')
        )
        vis.append(oh.make_tensor_value_info(scale_out, TensorProto.FLOAT, [4]))

        # 4) Pass-through image (algo stage doesn’t apply subsampling)
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        out_name = f'{stage}.applier'
        nodes.append(
            oh.make_node('Identity', inputs=[input_image], outputs=[out_name], name=f'{stage}.identity')
        )
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']))
        vis.append(oh.make_tensor_value_info(out_name,   TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']))

        # 5) Outputs: image + replicated scale_out
        outputs = {
            'applier':        {'name': out_name},
            'subsample_scale': {'name': scale_out},
        }

        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{upstream}.applier')
