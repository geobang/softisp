from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from .demosaic_base import DemosaicBase

class DemosaicAvgLuxV1(DemosaicBase):
    """
    DemosaicAvgLuxV1
    ----------------
    - Needs: input_image [n,4,h,w] (R, G1, G2, B planes)
    - Provides: applier [n,3,h,w] (R, Gavg, B), lux_scalar []
    - Behavior:
        * No interpolation: average G1 and G2 -> Gavg
        * Concat R, Gavg, B to produce RGB applier
        * Compute lux as mean pseudo-Y (0.2126*R + 0.7152*G + 0.0722*B) over H,W,N
    - Notes:
        * Keeps applier contract intact
        * Adds lux_scalar for downstream consumers (AWB/CCM/Tone v2)
    """
    name = 'demosaic_avg_lux_v1'
    version = 'v1'
    needs = ['input_image']
    provides = ['applier', 'lux_scalar']

    def _split_rggb(self, stage, input_image, nodes, vis):
        r = f'{stage}.r'
        g1 = f'{stage}.g1'
        g2 = f'{stage}.g2'
        b = f'{stage}.b'
        nodes.append(oh.make_node('Split', inputs=[input_image], outputs=[r, g1, g2, b], name=f'{stage}.split_rg1g2b', axis=1))
        vis += [oh.make_tensor_value_info(r, TensorProto.FLOAT, ['n', 1, 'h', 'w']), oh.make_tensor_value_info(g1, TensorProto.FLOAT, ['n', 1, 'h', 'w']), oh.make_tensor_value_info(g2, TensorProto.FLOAT, ['n', 1, 'h', 'w']), oh.make_tensor_value_info(b, TensorProto.FLOAT, ['n', 1, 'h', 'w'])]
        return (r, g1, g2, b)

    def _avg_green(self, stage, g1, g2, nodes, vis, inits):
        gsum = f'{stage}.g_sum'
        gavg = f'{stage}.gavg'
        half = f'{stage}.half'
        nodes.append(oh.make_node('Add', inputs=[g1, g2], outputs=[gsum], name=f'{stage}.add_g'))
        inits.append(oh.make_tensor(half, TensorProto.FLOAT, [], [0.5]))
        nodes.append(oh.make_node('Mul', inputs=[gsum, half], outputs=[gavg], name=f'{stage}.mul_half'))
        vis.append(oh.make_tensor_value_info(gavg, TensorProto.FLOAT, ['n', 1, 'h', 'w']))
        return gavg

    def _concat_rgb(self, stage, r, gavg, b, nodes, vis):
        applier = f'{stage}.applier'
        nodes.append(oh.make_node('Concat', inputs=[r, gavg, b], outputs=[applier], name=f'{stage}.concat_rgb', axis=1))
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        return applier

    def _compute_lux(self, stage, r, g, b, nodes, vis, inits):
        wr = f'{stage}.wr'
        wg = f'{stage}.wg'
        wb = f'{stage}.wb'
        inits += [oh.make_tensor(wr, TensorProto.FLOAT, [], [0.2126]), oh.make_tensor(wg, TensorProto.FLOAT, [], [0.7152]), oh.make_tensor(wb, TensorProto.FLOAT, [], [0.0722])]
        yr = f'{stage}.yr'
        yg = f'{stage}.yg'
        yb = f'{stage}.yb'
        nodes.append(oh.make_node('Mul', inputs=[r, wr], outputs=[yr], name=f'{stage}.mul_yr'))
        nodes.append(oh.make_node('Mul', inputs=[g, wg], outputs=[yg], name=f'{stage}.mul_yg'))
        nodes.append(oh.make_node('Mul', inputs=[b, wb], outputs=[yb], name=f'{stage}.mul_yb'))
        ysum1 = f'{stage}.ysum1'
        ysum = f'{stage}.ysum'
        nodes.append(oh.make_node('Add', inputs=[yr, yg], outputs=[ysum1], name=f'{stage}.add_yrg'))
        nodes.append(oh.make_node('Add', inputs=[ysum1, yb], outputs=[ysum], name=f'{stage}.add_y'))
        lux_hw = f'{stage}.lux_hw'
        lux = f'{stage}.lux_scalar'
        nodes.append(oh.make_node('ReduceMean', inputs=[ysum], outputs=[lux_hw], name=f'{stage}.lux_mean_hw', axes=[2, 3], keepdims=0))
        nodes.append(oh.make_node('ReduceMean', inputs=[lux_hw], outputs=[lux], name=f'{stage}.lux_mean_n', axes=[0], keepdims=0))
        vis.append(oh.make_tensor_value_info(lux, TensorProto.FLOAT, []))
        return lux

    def build_algo(self, stage: str, prev_stages=None):
        """
        Orchestrate pseudo-demosaic + lux:
        - Split RGGB planes
        - Average greens
        - Concat to RGB applier
        - Compute lux from RGB
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.input_image'
        r, g1, g2, b = self._split_rggb(stage, input_image, nodes, vis)
        gavg = self._avg_green(stage, g1, g2, nodes, vis, inits)
        applier = self._concat_rgb(stage, r, gavg, b, nodes, vis)
        lux = self._compute_lux(stage, r, gavg, b, nodes, vis, inits)
        outputs = {'applier': {'name': applier}, 'lux_scalar': {'name': lux}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')