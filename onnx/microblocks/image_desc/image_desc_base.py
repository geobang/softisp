from microblocks.base import BuildResult
import onnx.helper as oh
from microblocks.base import MicroblockBase

class ImageDescBase(MicroblockBase):
    """
    Pseudo microblock that anchors image metadata.
    Declares image [n,c,h,stride] and width as graph inputs.
    Each input is re-emitted by an Identity node so outputs are node-backed.
    """
    name = 'image_desc_base'
    version = 'v0'
    deps = []
    needs = ['image', 'width']

    def build_applier(self, stage: str, prev_stages=None):
        image_in = f'{stage}.image_in'
        width_in = f'{stage}.width_in'
        image_out = f'{stage}.applier'
        nodes = [oh.make_node('Identity', inputs=[image_in], outputs=[image_out], name=f'{stage}_image_id')]
        vis = [oh.make_tensor_value_info(image_in, oh.TensorProto.FLOAT, ['n', 'c', 'h', 'stride']), oh.make_tensor_value_info(image_out, oh.TensorProto.FLOAT, ['n', 'c', 'h', 'stride'])]
        outputs = {'image': {'name': image_out}}
        return BuildResult(outputs, nodes, [], vis).appendInput(f'{stage}.applier').appendInput(f'{stage}.width_in')

    def build_algo(self, stage: str, prev_stages=None):
        """
        Wrapper for algorithmic build. By default, just calls build_applier().
        This keeps applier and algo paths consistent unless overridden.
        """
        return self.build_applier(stage, prev_stages=prev_stages)
