from microblocks.base import BuildResult
import onnx.helper as oh
from microblocks.base import MicroblockBase

class CCMBase(MicroblockBase):
    """
    Color Correction Matrix microblock.
    Input:  [n,3,h,width] (RGB)
    Output: [n,3,h,width] (RGB after CCM)
    Needs:  ccm [3,3] matrix
    """
    name = 'ccm_base'
    version = 'v0'
    deps = ['awb_base']
    needs = ['ccm']

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f'{stage}.applier'
        ccm = f'{stage}.ccm'
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        node = oh.make_node('MatMul', inputs=[input_image, ccm], outputs=[out_name], name=f'{stage}_ccm')
        vis = [oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ['n', '3', 'h', 'width']), oh.make_tensor_value_info(ccm, oh.TensorProto.FLOAT, [3, 3]), oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ['n', '3', 'h', 'width'])]
        outputs = {'applier': {'name': out_name}}
        return BuildResult(outputs, [node], [], vis).appendInput(f'{prev_stages[0]}.applier')