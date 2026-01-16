from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from .gamma_base import GammaBase

class GammaV2(GammaBase):
    """
    GammaV2
    -------
    - Needs: input_image [n,3,h,w], gamma_value [], lux_scalar []
    - Provides: applier [n,3,h,w], gamma_effective []
    - Behavior:
        * Computes an effective gamma from base gamma_value and lux_scalar:
            gamma_eff = clamp(gamma_value * (1 + k * (1 - norm_lux)), min_g, max_g)
          where norm_lux = clamp(lux / lux_ref, 0, 1)
        * Applies gamma correction: output = input ^ (1 / gamma_eff)
        * Exposes gamma_effective for audit
    - Notes:
        * Fast-path compatible: single Pow op on RGB
        * Coordinator can tune k, lux_ref, min_g, max_g via initializers or future needs
    """
    name = 'gamma_v2'
    version = 'v2'

    def _normalize_lux(self, stage, lux_scalar, nodes, vis, inits):
        lux_ref = f'{stage}.lux_ref'
        inv_ref = f'{stage}.inv_ref'
        norm_raw = f'{stage}.norm_raw'
        zero = f'{stage}.zero'
        one = f'{stage}.one'
        inits += [oh.make_tensor(lux_ref, TensorProto.FLOAT, [], [100.0]), oh.make_tensor(inv_ref, TensorProto.FLOAT, [], [1.0 / 100.0]), oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]), oh.make_tensor(one, TensorProto.FLOAT, [], [1.0])]
        nodes.append(oh.make_node('Mul', inputs=[lux_scalar, inv_ref], outputs=[norm_raw], name=f'{stage}.lux_norm_mul'))
        norm0 = f'{stage}.norm0'
        norm1 = f'{stage}.norm1'
        nodes.append(oh.make_node('Max', inputs=[norm_raw, zero], outputs=[norm0], name=f'{stage}.lux_norm_max'))
        nodes.append(oh.make_node('Min', inputs=[norm0, one], outputs=[norm1], name=f'{stage}.lux_norm_min'))
        vis.append(oh.make_tensor_value_info(norm1, TensorProto.FLOAT, []))
        return norm1

    def _compute_gamma_eff(self, stage, gamma_value, norm_lux, nodes, vis, inits):
        k = f'{stage}.k'
        one = f'{stage}.one'
        min_g = f'{stage}.min_g'
        max_g = f'{stage}.max_g'
        inits += [oh.make_tensor(k, TensorProto.FLOAT, [], [0.25]), oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]), oh.make_tensor(min_g, TensorProto.FLOAT, [], [1.8]), oh.make_tensor(max_g, TensorProto.FLOAT, [], [2.6])]
        inv_norm = f'{stage}.inv_norm'
        k_inv_norm = f'{stage}.k_inv_norm'
        one_plus = f'{stage}.one_plus'
        nodes.append(oh.make_node('Sub', inputs=[one, norm_lux], outputs=[inv_norm], name=f'{stage}.lux_inv'))
        nodes.append(oh.make_node('Mul', inputs=[k, inv_norm], outputs=[k_inv_norm], name=f'{stage}.lux_k_mul'))
        nodes.append(oh.make_node('Add', inputs=[one, k_inv_norm], outputs=[one_plus], name=f'{stage}.lux_one_plus'))
        gamma_raw = f'{stage}.gamma_raw'
        nodes.append(oh.make_node('Mul', inputs=[gamma_value, one_plus], outputs=[gamma_raw], name=f'{stage}.gamma_scale'))
        gamma_clamp0 = f'{stage}.gamma_clamp0'
        gamma_eff = f'{stage}.gamma_effective'
        nodes.append(oh.make_node('Max', inputs=[gamma_raw, min_g], outputs=[gamma_clamp0], name=f'{stage}.gamma_max'))
        nodes.append(oh.make_node('Min', inputs=[gamma_clamp0, max_g], outputs=[gamma_eff], name=f'{stage}.gamma_min'))
        vis.append(oh.make_tensor_value_info(gamma_eff, TensorProto.FLOAT, []))
        return gamma_eff

    def _apply_gamma(self, stage, input_image, gamma_eff, nodes, vis):
        inv_gamma = f'{stage}.inv_gamma'
        applier = f'{stage}.applier'
        nodes.append(oh.make_node('Div', inputs=[oh.make_tensor_value_info('', TensorProto.FLOAT, []), gamma_eff], outputs=[inv_gamma], name=f'{stage}.inv_gamma_div'))
        one = f'{stage}.one_pow'
        nodes.pop()
        nodes.append(oh.make_node('Constant', inputs=[], outputs=[one], value=oh.make_tensor(one, TensorProto.FLOAT, [], [1.0])))
        nodes.append(oh.make_node('Div', inputs=[one, gamma_eff], outputs=[inv_gamma], name=f'{stage}.inv_gamma'))
        nodes.append(oh.make_node('Pow', inputs=[input_image, inv_gamma], outputs=[applier], name=f'{stage}_gamma_pow'))
        vis += [oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['n', 3, 'h', 'w']), oh.make_tensor_value_info(inv_gamma, TensorProto.FLOAT, []), oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h', 'w'])]
        return applier

    def build_algo(self, stage: str, prev_stages=None):
        """
        Orchestrate GammaV2:
        - Normalize lux to [0,1]
        - Compute gamma_effective from base gamma and lux
        - Apply gamma via Pow(input, 1/gamma_effective)
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        gamma_value = f'{upstream}.gamma_value'
        lux_scalar = f'{upstream}.lux_scalar'
        norm_lux = self._normalize_lux(stage, lux_scalar, nodes, vis, inits)
        gamma_eff = self._compute_gamma_eff(stage, gamma_value, norm_lux, nodes, vis, inits)
        applier = self._apply_gamma(stage, input_image, gamma_eff, nodes, vis)
        outputs = {'applier': {'name': applier}, 'gamma_effective': {'name': gamma_eff}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')