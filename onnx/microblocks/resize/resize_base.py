from microblocks.base import BuildResult
import onnx.helper as oh
from microblocks.base import MicroblockBase

class ResizeBase(MicroblockBase):
    """
    Resize microblock.
    Input:  [n,3,h,width] (RGB)
    Output: [n,3,target_h,target_w]
    Needs:  target_h, target_w
    """
    name = 'resize_base'
    version = 'v0'

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f'{stage}.applier'
        target_h = f'{stage}.target_h'
        target_w = f'{stage}.target_w'
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        node = oh.make_node('Resize', inputs=[input_image, '', target_h, target_w], outputs=[out_name], name=f'{stage}_resize')
        vis = [oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ['n', '3', 'h', 'width']), oh.make_tensor_value_info(target_h, oh.TensorProto.INT64, [1]), oh.make_tensor_value_info(target_w, oh.TensorProto.INT64, [1]), oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w'])]
        outputs = {'applier': {'name': out_name}}
        return BuildResult(outputs, [node], [], vis).appendInput(f'{prev_stages[0]}.applier')