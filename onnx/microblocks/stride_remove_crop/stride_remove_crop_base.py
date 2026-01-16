from microblocks.base import BuildResult
import onnx.helper as oh
from microblocks.base import MicroblockBase

class StrideRemoveCropBase(MicroblockBase):
    name = 'stride_remove_crop'
    version = 'v0'
    deps = ['image_desc_base']
    needs = [f'{name}.crop_starts', f'{name}.crop_ends']

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f'{stage}.applier'
        starts = f'{stage}.crop_starts'
        ends = f'{stage}.crop_ends'
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        node = oh.make_node('Slice', inputs=[input_image, starts, ends], outputs=[out_name], name=f'{stage}_slice')
        vis = [oh.make_tensor_value_info(starts, oh.TensorProto.INT64, [4]), oh.make_tensor_value_info(ends, oh.TensorProto.INT64, [4]), oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ['N', 'C', 'H', 'W'])]
        outputs = {'applier': {'name': out_name}}
        return BuildResult(outputs, [node], [], vis).appendInput(f'{prev_stages[0]}.applier')

    def build_algo(self, stage: str, prev_stages=None):
        """
        Wrapper for algorithmic build. By default, just calls build_applier().
        This keeps applier and algo paths consistent unless overridden.
        """
        return self.build_applier(stage, prev_stages=prev_stages)