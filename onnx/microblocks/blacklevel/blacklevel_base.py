from microblocks.base import BuildResult
import onnx.helper as oh
from microblocks.base import MicroblockBase

class BlackLevelBase(MicroblockBase):
    name = 'blacklevel'
    version = 'v0'
    deps = ['stride_remove_crop']
    needs = ['offset']

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f'{stage}.applier'
        offset_name = f'{stage}.offset'
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        node = oh.make_node('Sub', inputs=[input_image, offset_name], outputs=[out_name], name=f'{stage}_sub')
        vis = [oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ['N', 'C', 'H', 'W']), oh.make_tensor_value_info(offset_name, oh.TensorProto.FLOAT, [1]), oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ['N', 'C', 'H', 'W'])]
        outputs = {'applier': {'name': out_name}}
        return BuildResult(outputs, [node], [], vis).appendInput(f'{prev_stages[0]}.applier')