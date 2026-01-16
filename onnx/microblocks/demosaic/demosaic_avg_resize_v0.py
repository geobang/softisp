from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from .demosaic_base import DemosaicBase

class DemosaicAvgResizeV0(DemosaicBase):
    """
    DemosaicAvgResizeV1
    -------------------
    - Needs: input_image [n,4,h,w] (R,G1,G2,B planes)
    - Provides: applier [n,3,h,w], lux_scalar []
    - Behavior:
        * Average G1 and G2 -> Gavg
        * Concat R,Gavg,B -> RGB
        * Compute lux from RGB
    """
    name = 'demosaic_avg_resize_v1'
    version = 'v1'

    def _split_rggb(self, stage, input_image, nodes, vis):
        r, g1, g2, b = [f'{stage}.{ch}' for ch in ('r', 'g1', 'g2', 'b')]
        nodes.append(oh.make_node('Split', inputs=[input_image], outputs=[r, g1, g2, b], name=f'{stage}.split_rg1g2b', axis=1))
        return (r, g1, g2, b)

    def _avg_green(self, stage, g1, g2, nodes, inits):
        gsum = f'{stage}.g_sum'
        gavg = f'{stage}.gavg'
        half = f'{stage}.half'
        nodes.append(oh.make_node('Add', inputs=[g1, g2], outputs=[gsum], name=f'{stage}.add_g'))
        inits.append(oh.make_tensor(half, TensorProto.FLOAT, [], [0.5]))
        nodes.append(oh.make_node('Mul', inputs=[gsum, half], outputs=[gavg], name=f'{stage}.mul_half'))
        return gavg

    def _concat_rgb(self, stage, r, gavg, b, nodes, vis):
        local_rgb = f'{stage}.concat_rgb_output'
        nodes.append(oh.make_node('Concat', inputs=[r, gavg, b], outputs=[local_rgb], name=f'{stage}.concat_rgb', axis=1))
        vis.append(oh.make_tensor_value_info(local_rgb, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        return local_rgb

    def _resize_half(self, stage, rgb, nodes, vis, inits):
        applier = f'{stage}.applier'
        scales = f'{stage}.scales'
        inits.append(oh.make_tensor(scales, TensorProto.FLOAT, [4], [1.0, 1.0, 0.5, 0.5]))
        resize_factor = f'{stage}.resize_factor'
        inits.append(oh.make_tensor(resize_factor, TensorProto.FLOAT, [], [0.5]))
        vis.append(oh.make_tensor_value_info(resize_factor, TensorProto.FLOAT, []))
        nodes.append(oh.make_node('Resize', inputs=[rgb, scales], outputs=[applier], name=f'{stage}.resize_half', mode='linear'))
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h/2', 'w/2']))
        return (applier, resize_factor)

    def _compute_lux(self, stage, applier, nodes, vis, inits):
        r, g, b = [f'{stage}.lux_{ch}' for ch in ('r', 'g', 'b')]
        nodes.append(oh.make_node('Split', inputs=[applier], outputs=[r, g, b], name=f'{stage}.lux_split', axis=1))
        wr, wg, wb = [f'{stage}.w_{ch}' for ch in ('r', 'g', 'b')]
        inits += [oh.make_tensor(wr, TensorProto.FLOAT, [], [0.2126]), oh.make_tensor(wg, TensorProto.FLOAT, [], [0.7152]), oh.make_tensor(wb, TensorProto.FLOAT, [], [0.0722])]
        yr, yg, yb = [f'{stage}.y_{ch}' for ch in ('r', 'g', 'b')]
        nodes.append(oh.make_node('Mul', inputs=[r, wr], outputs=[yr], name=f'{stage}.mul_yr'))
        nodes.append(oh.make_node('Mul', inputs=[g, wg], outputs=[yg], name=f'{stage}.mul_yg'))
        nodes.append(oh.make_node('Mul', inputs=[b, wb], outputs=[yb], name=f'{stage}.mul_yb'))
        ysum1, ysum = (f'{stage}.ysum1', f'{stage}.ysum')
        nodes.append(oh.make_node('Add', inputs=[yr, yg], outputs=[ysum1], name=f'{stage}.add_yrg'))
        nodes.append(oh.make_node('Add', inputs=[ysum1, yb], outputs=[ysum], name=f'{stage}.add_y'))
        lux_hw, lux = (f'{stage}.lux_hw', f'{stage}.lux_scalar')
        nodes.append(oh.make_node('ReduceMean', inputs=[ysum], outputs=[lux_hw], name=f'{stage}.lux_mean_hw', axes=[2, 3], keepdims=0))
        nodes.append(oh.make_node('ReduceMean', inputs=[lux_hw], outputs=[lux], name=f'{stage}.lux_mean_n', axes=[0], keepdims=0))
        vis.append(oh.make_tensor_value_info(lux, TensorProto.FLOAT, []))
        return lux

    def build_algo(self, stage: str, prev_stages=None):
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        r, g1, g2, b = self._split_rggb(stage, input_image, nodes, vis)
        gavg = self._avg_green(stage, g1, g2, nodes, inits)
        rgb_val = self._concat_rgb(stage, r, gavg, b, nodes, vis)
        applier, resize_factor = self._resize_half(stage, rgb_val, nodes, vis, inits)
        lux = self._compute_lux(stage, applier, nodes, vis, inits)
        outputs = {'applier': {'name': applier}, 'lux_scalar': {'name': lux}, 'resize_factor': {'name': resize_factor}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')