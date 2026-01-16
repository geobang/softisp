from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto
from .demosaic_base import DemosaicBase

class DemosaicAvgV1(DemosaicBase):
    """
    DemosaicAvgV1 (v1)
    ------------------
    Needs:
        - input_image [n,4,h,w] (R,G1,G2,B planes)

    Provides:
        - applier [n,3,h,w] : full-resolution RGB image

    Behavior:
        * Average G1 and G2 -> Gavg
        * Concat R,Gavg,B -> RGB
    """
    name = 'demosaic_avg_v1'
    version = 'v1'
    needs = ['input_image']
    provides = ['applier']

    def _split_rggb(self, stage, input_image, nodes, vis):
        r, g1, g2, b = [f'{stage}.{ch}' for ch in ('r', 'g1', 'g2', 'b')]
        nodes.append(
            oh.make_node(
                'Split',
                inputs=[input_image],
                outputs=[r, g1, g2, b],
                name=f'{stage}.split_rg1g2b',
                axis=1
            )
        )
        return r, g1, g2, b

    def _avg_green(self, stage, g1, g2, nodes, inits):
        gsum = f'{stage}.g_sum'
        gavg = f'{stage}.gavg'
        half = f'{stage}.half'
        nodes.append(oh.make_node('Add', inputs=[g1, g2], outputs=[gsum], name=f'{stage}.add_g'))
        inits.append(oh.make_tensor(half, TensorProto.FLOAT, [], [0.5]))
        nodes.append(oh.make_node('Mul', inputs=[gsum, half], outputs=[gavg], name=f'{stage}.mul_half'))
        return gavg

    def _concat_rgb(self, stage, r, gavg, b, nodes, vis):
        rgb = f'{stage}.applier'
        nodes.append(
            oh.make_node('Concat', inputs=[r, gavg, b], outputs=[rgb], name=f'{stage}.concat_rgb', axis=1)
        )
        vis.append(oh.make_tensor_value_info(rgb, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        return rgb

    def build_algo(self, stage: str, prev_stages=None):
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'

        # Build nodes
        r, g1, g2, b = self._split_rggb(stage, input_image, nodes, vis)
        gavg = self._avg_green(stage, g1, g2, nodes, inits)
        applier = self._concat_rgb(stage, r, gavg, b, nodes, vis)

        outputs = {'applier': {'name': applier}}

        # BuildResult + declare external input
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image)
        return result

    def build_applier(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)
