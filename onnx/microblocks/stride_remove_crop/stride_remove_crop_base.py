from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class StrideRemoveCropBase(MicroblockBase):
    """
    Stride removal + crop microblock.
    Input:  [n,c,h,w] (image)
    Output: [N,C,H,W] (cropped image)
    Needs:  crop_starts [4], crop_ends [4]
    """
    name = 'stride_remove_crop'
    version = 'v0'
    deps = ['image_desc_base']
    needs = ['crop_starts', 'crop_ends']
    provides = ['applier']

    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage

        # Names
        input_image = f'{upstream}.applier'
        starts      = f'{stage}.crop_starts'
        ends        = f'{stage}.crop_ends'
        out_name    = f'{stage}.applier'

        # Node
        slice_node = oh.make_node(
            'Slice',
            inputs=[input_image, starts, ends],
            outputs=[out_name],
            name=f'{stage}_slice'
        )

        # ValueInfos
        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['N', 'C', 'H', 'W']),
            oh.make_tensor_value_info(starts,      TensorProto.INT64, [4]),
            oh.make_tensor_value_info(ends,        TensorProto.INT64, [4]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ['N', 'C', 'H', 'W']),
        ]

        outputs = {'applier': {'name': out_name}}

        # BuildResult + declare all external needs as inputs
        result = BuildResult(outputs, [slice_node], [], vis)
        result.appendInput(input_image)
        result.appendInput(starts)
        result.appendInput(ends)
        return result

    def build_algo(self, stage: str, prev_stages=None):
        """
        Algo path mirrors applier for consistency.
        Coordinator supplies crop_starts and crop_ends.
        """
        return self.build_applier(stage, prev_stages=prev_stages)
