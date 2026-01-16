from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class BlackLevelBase(MicroblockBase):
    """
    Black level subtraction microblock.
    Input:  [N,C,H,W] (image)
    Output: [N,C,H,W] (image with black level offset removed)
    Needs:  offset [1] (scalar)
    """
    name = 'blacklevel'
    version = 'v0'

    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage

        # Names
        input_image = f'{upstream}.applier'
        offset_name = f'{stage}.offset'
        out_name    = f'{stage}.applier'

        # Node
        sub_node = oh.make_node(
            'Sub',
            inputs=[input_image, offset_name],
            outputs=[out_name],
            name=f'{stage}_sub'
        )

        # ValueInfos
        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['N', 'C', 'H', 'W']),
            oh.make_tensor_value_info(offset_name, TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ['N', 'C', 'H', 'W']),
        ]

        outputs = {'applier': {'name': out_name}}

        # BuildResult + declare all external needs as inputs
        result = BuildResult(outputs, [sub_node], [], vis)
        result.appendInput(input_image)
        result.appendInput(offset_name)
        return result

    def build_algo(self, stage: str, prev_stages=None):
        """
        Algo path mirrors applier for consistency.
        Coordinator supplies offset.
        """
        return self.build_applier(stage, prev_stages=prev_stages)
