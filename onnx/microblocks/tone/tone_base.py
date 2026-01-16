from microblocks.base import BuildResult
import onnx.helper as oh
from microblocks.base import MicroblockBase

class ToneMapBase(MicroblockBase):
    """
    Tone Mapping microblock.
    Input:  [n,3,target_h,target_w] (RGB)
    Output: [n,3,target_h,target_w] (tone-mapped RGB)
    Needs:  tonemap_curve (lookup table or curve parameters)
    """
    name = 'tonemap_base'
    version = 'v0'
    deps = ['resize_base']
    needs = ['tonemap_curve']

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f'{stage}.applier'
        curve = f'{stage}.tonemap_curve'
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        node = oh.make_node('Mul', inputs=[input_image, curve], outputs=[out_name], name=f'{stage}_tonemap')
        vis = [oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']), oh.make_tensor_value_info(curve, oh.TensorProto.FLOAT, [1]), oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w'])]
        outputs = {'applier': {'name': out_name}}
        return BuildResult(outputs, [node], [], vis).appendInput(f'{prev_stages[0]}.applier')

    def build_algo(self, stage: str, prev_stages=None):
        """Declare or initialize tonemap_curve for this stage."""
        nodes, inits, vis = ([], [], [])
        curve = f'{stage}.tonemap_curve'
        vis.append(oh.make_tensor_value_info(curve, oh.TensorProto.FLOAT, [1]))
        default_curve = [0.8]
        inits.append(oh.make_tensor(curve, oh.TensorProto.FLOAT, [1], default_curve))
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        out_name = f'{stage}.applier'
        nodes.append(oh.make_node('Identity', inputs=[input_image], outputs=[out_name], name=f'{stage}.identity'))
        vis = [oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']), oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w'])]
        outputs = {'applier': {'name': out_name}, 'tonemap_curve': {'name': curve}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')