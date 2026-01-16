from microblocks.base import BuildResult
from .ccm_quadratic_v1 import CCMQuadraticV1
import onnx.helper as oh
from onnx import TensorProto

class CCMQuadraticV2(CCMQuadraticV1):
    """
    Extends CCMQuadraticV1:
    - Reuses quadratic fit + row normalization
    - Adds first-pass post-CCM bias calculation (CI vector/scalar)
    """
    name = 'ccm_quadratic_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        outputs, nodes, inits, vis = super().build_algo(stage, prev_stages)
        in_tiles = f'{stage}.stats_tiles'
        in_mask = f'{stage}.neutral_mask'
        in_w = f'{stage}.ci_weights'
        out_tiles = f'{stage}.post_ccm_tiles'
        out_ci_vec = f'{stage}.ci_vec'
        out_ci_scalar = f'{stage}.ci_scalar'
        ccm_T = f'{stage}.ccm_T'
        nodes.append(oh.make_node('Transpose', inputs=[outputs['ccm']['name']], outputs=[ccm_T], name=f'{stage}.transpose_ccm', perm=[1, 0]))
        nodes.append(oh.make_node('MatMul', inputs=[in_tiles, ccm_T], outputs=[out_tiles], name=f'{stage}.apply_ccm'))
        outputs.update({'post_ccm_tiles': {'name': out_tiles}, 'ci_vec': {'name': out_ci_vec}, 'ci_scalar': {'name': out_ci_scalar}})
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')