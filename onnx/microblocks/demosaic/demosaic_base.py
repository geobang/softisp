from microblocks.base import BuildResult
import onnx.helper as oh
from microblocks.base import MicroblockBase

class DemosaicBase(MicroblockBase):
    """
    Microblock that performs Bayer demosaicing.
    Input:  [n,4,h,width]  (RGGB mosaic channels)
    Output: [n,3,h,width]  (RGB image)
    Needs:  None (parameters are fixed for bilinear demo)
    """
    name = 'demosaic_base'
    version = 'v0'

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f'{stage}.applier'
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        conv_out = out_name
        node = oh.make_node('Conv', inputs=[input_image, f'{stage}.kernels'], outputs=[conv_out], name=f'{stage}_demosaic')
        vis = [oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ['n', '4', 'h', 'width']), oh.make_tensor_value_info(f'{stage}.kernels', oh.TensorProto.FLOAT, [3, 4, 3, 3]), oh.make_tensor_value_info(conv_out, oh.TensorProto.FLOAT, ['n', '3', 'h', 'width'])]
        outputs = {'applier': {'name': conv_out}}
        inits = []
        return BuildResult(outputs, [node], inits, vis).appendInput(f'{prev_stages[0]}.applier')