from microblocks.base import BuildResult, MicroblockBase
import onnx
import onnx.helper as oh
from onnx import TensorProto

class BayerNormBase(MicroblockBase):
    """
    Normalizer microblock.
    Consumes desc.image [n,c,h,w], selects the first image,
    reshapes to [1,4,h,w], and normalizes raw values to [0,1].
    """
    name = 'bayernorm_base'
    version = 'v0'
    deps = ['image_desc_base']
    needs = ['norm_scale']  # plus slice/reshape params provided by coordinator

    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage

        # Names
        input_image  = f'{upstream}.applier'
        starts       = f'{stage}.starts'
        ends         = f'{stage}.ends'
        axes         = f'{stage}.axes'
        shape_tensor = f'{stage}.target_shape'
        norm_scale   = f'{stage}.norm_scale'

        slice_out    = f'{stage}.sliced'
        reshape_out  = f'{stage}.reshaped'
        out_name     = f'{stage}.applier'

        # Nodes
        slice_node = oh.make_node(
            'Slice',
            inputs=[input_image, starts, ends, axes],
            outputs=[slice_out],
            name=f'{stage}_slice_first'
        )

        reshape_node = oh.make_node(
            'Reshape',
            inputs=[slice_out, shape_tensor],
            outputs=[reshape_out],
            name=f'{stage}_reshape'
        )

        div_node = oh.make_node(
            'Div',
            inputs=[reshape_out, norm_scale],
            outputs=[out_name],
            name=f'{stage}_normalize'
        )

        # ValueInfos (metadata)
        vis = [
            oh.make_tensor_value_info(input_image,  TensorProto.FLOAT, ['n', 'c', 'h', 'w']),
            oh.make_tensor_value_info(starts,       TensorProto.INT64, [1]),
            oh.make_tensor_value_info(ends,         TensorProto.INT64, [1]),
            oh.make_tensor_value_info(axes,         TensorProto.INT64, [1]),
            oh.make_tensor_value_info(shape_tensor, TensorProto.INT64, [4]),
            oh.make_tensor_value_info(norm_scale,   TensorProto.FLOAT, []),
            oh.make_tensor_value_info(slice_out,    TensorProto.FLOAT, ['n', 'c', 'h', 'w']),  # intermediate
            oh.make_tensor_value_info(reshape_out,  TensorProto.FLOAT, ['1', '4', 'h', 'w']),  # intermediate
            oh.make_tensor_value_info(out_name,     TensorProto.FLOAT, ['1', '4', 'h', 'w']),
        ]

        outputs = {'applier': {'name': out_name}}

        # BuildResult + declare all external needs as function inputs
        result = BuildResult(outputs, [slice_node, reshape_node, div_node], [], vis)
        result.appendInput(input_image)
        result.appendInput(starts)
        result.appendInput(ends)
        result.appendInput(axes)
        result.appendInput(shape_tensor)
        result.appendInput(norm_scale)
        return result

    def build_algo(self, stage: str, prev_stages=None):
        """
        Algo path mirrors applier for consistency.
        Coordinator supplies slice/reshape params and norm_scale.
        """
        return self.build_applier(stage, prev_stages=prev_stages)
