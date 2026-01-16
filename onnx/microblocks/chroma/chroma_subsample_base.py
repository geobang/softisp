from microblocks.base import BuildResult
import onnx.helper as oh
from microblocks.base import MicroblockBase

class ChromaSubsampleBase(MicroblockBase):
    """
    Chroma subsampling microblock.
    Input:  [n,3,target_h,target_w] (YUV)
    Output: [n,3,target_h,target_w/2] (YUV 4:2:0)
    Needs:  subsample_scale [2] (scale factors for H and W)
    """
    name = 'chroma_subsample_base'
    version = 'v0'
    deps = ['yuvconvert_base']
    needs = ['subsample_scale']

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f'{stage}.applier'
        scale = f'{stage}.subsample_scale'
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        node = oh.make_node('Resize', inputs=[input_image, '', scale], outputs=[out_name], name=f'{stage}_chroma', mode='nearest')
        vis = [oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']), oh.make_tensor_value_info(scale, oh.TensorProto.FLOAT, [4]), oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w/2'])]
        outputs = {'applier': {'name': out_name}}
        return BuildResult(outputs, [node], [], vis).appendInput(f'{prev_stages[0]}.applier')

    def build_algo(self, stage: str, prev_stages=None):
        """Declare or initialize subsample_scale for this stage."""
        nodes, inits, vis = ([], [], [])
        scale = f'{stage}.subsample_scale'
        vis.append(oh.make_tensor_value_info(scale, oh.TensorProto.FLOAT, [4]))
        default_scale = [1.0, 1.0, 1.0, 0.5]
        inits.append(oh.make_tensor(scale, oh.TensorProto.FLOAT, [4], default_scale))
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        out_name = f'{stage}.applier'
        nodes.append(oh.make_node('Identity', inputs=[input_image], outputs=[out_name], name=f'{stage}.identity'))
        vis += [oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']), oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w'])]
        outputs = {'applier': {'name': out_name}, 'subsample_scale': {'name': scale}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')