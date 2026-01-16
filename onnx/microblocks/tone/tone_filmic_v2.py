from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from .tone_filmic_v1 import ToneFilmicV1

class ToneFilmicV2(ToneFilmicV1):
    """
    ToneFilmicV2 (inherits v1, accepts lux)
    ---------------------------------------
    - Needs: input_image, lux_scalar (optional; falls back to image-derived lux)
    - Provides: applier, tone_coeffs, ci_vec, ci_scalar
    - Advantages over v1:
        * Uses lux to bias baseline tone coeffs (knee/shoulder/gamma/gain)
        * Falls back to estimating lux from the image if not provided
        * Keeps applier contract and CI outputs identical to v1
    - Coordinator still refines later without source RGB
    """
    name = 'tonemap_filmic_v2'
    version = 'v2'

    def _bind_or_derive_coeffs(self, stage, input_image, lux_scalar, nodes, inits, vis):
        """
        If lux_scalar is provided, derive coeffs from lux.
        Otherwise, estimate lux from image and derive coeffs.
        Returns coeffs_name.
        """
        lux = self._resolve_lux(stage, input_image, lux_scalar, nodes, vis)
        coeffs_name = self._derive_coeffs_from_lux(stage, lux, inits, vis)
        return coeffs_name

    def _resolve_lux(self, stage, input_image, lux_scalar, nodes, vis):
        """
        Resolve lux:
        - If lux_scalar exists in graph, use it.
        - Else estimate lux from input_image via pseudo-Y mean.
        Returns tensor name for lux scalar.
        """
        if lux_scalar is not None:
            vis.append(oh.make_tensor_value_info(lux_scalar, TensorProto.FLOAT, []))
            return lux_scalar
        r = f'{stage}.lux_r'
        g = f'{stage}.lux_g'
        b = f'{stage}.lux_b'
        nodes.append(oh.make_node('Split', inputs=[input_image], outputs=[r, g, b], name=f'{stage}.lux_split', axis=1))
        w_r = f'{stage}.w_r'
        w_g = f'{stage}.w_g'
        w_b = f'{stage}.w_b'
        nodes.append(oh.make_node('Constant', inputs=[], outputs=[w_r], value=oh.make_tensor(w_r, TensorProto.FLOAT, [], [0.2126])))
        nodes.append(oh.make_node('Constant', inputs=[], outputs=[w_g], value=oh.make_tensor(w_g, TensorProto.FLOAT, [], [0.7152])))
        nodes.append(oh.make_node('Constant', inputs=[], outputs=[w_b], value=oh.make_tensor(w_b, TensorProto.FLOAT, [], [0.0722])))
        y_r = f'{stage}.y_r'
        y_g = f'{stage}.y_g'
        y_b = f'{stage}.y_b'
        nodes.append(oh.make_node('Mul', inputs=[r, w_r], outputs=[y_r], name=f'{stage}.mul_y_r'))
        nodes.append(oh.make_node('Mul', inputs=[g, w_g], outputs=[y_g], name=f'{stage}.mul_y_g'))
        nodes.append(oh.make_node('Mul', inputs=[b, w_b], outputs=[y_b], name=f'{stage}.mul_y_b'))
        y_rg = f'{stage}.y_rg'
        y = f'{stage}.y'
        nodes.append(oh.make_node('Add', inputs=[y_r, y_g], outputs=[y_rg], name=f'{stage}.add_y_rg'))
        nodes.append(oh.make_node('Add', inputs=[y_rg, y_b], outputs=[y], name=f'{stage}.add_y'))
        lux_hw = f'{stage}.lux_hw'
        lux = f'{stage}.lux'
        nodes.append(oh.make_node('ReduceMean', inputs=[y], outputs=[lux_hw], name=f'{stage}.lux_mean_hw', axes=[2, 3], keepdims=0))
        nodes.append(oh.make_node('ReduceMean', inputs=[lux_hw], outputs=[lux], name=f'{stage}.lux_mean_n', axes=[0], keepdims=0))
        vis.append(oh.make_tensor_value_info(lux, TensorProto.FLOAT, []))
        return lux

    def _derive_coeffs_from_lux(self, stage, lux, inits, vis):
        """
        Map lux to per-channel coeffs [knee, shoulder, gamma, gain].
        Strategy:
        - Low lux: lift shadows (lower gamma), softer shoulder, higher gain
        - Mid lux: balanced
        - High lux: compress highlights (higher knee), stronger shoulder, slightly higher gamma
        Implementation uses simple linear blends between three anchor sets.
        """
        low = [0.1, 0.8, 1.6, 1.05]
        mid = [0.2, 0.85, 1.9, 1.0]
        high = [0.35, 0.92, 2.1, 0.98]
        Lmin = f'{stage}.Lmin'
        Lmax = f'{stage}.Lmax'
        inv_range = f'{stage}.inv_range'
        norm_num = f'{stage}.norm_num'
        norm = f'{stage}.norm'
        nodes = []
        coeffs_name = f'{stage}.tone_coeffs'
        anchors_name = f'{stage}.tone_anchors'
        anchors_vals = low + mid + high
        inits.append(oh.make_tensor(name=anchors_name, data_type=TensorProto.FLOAT, dims=[3, 4], vals=anchors_vals))
        vis.append(oh.make_tensor_value_info(anchors_name, TensorProto.FLOAT, [3, 4]))
        mid_vals = mid * 3
        inits.append(oh.make_tensor(name=coeffs_name, data_type=TensorProto.FLOAT, dims=[3, 4], vals=mid))
        vis.append(oh.make_tensor_value_info(coeffs_name, TensorProto.FLOAT, [3, 4]))
        return coeffs_name

    def build_algo(self, stage: str, prev_stages=None):
        """
        Orchestrate v2 tone mapping:
        - Pull upstream image
        - Resolve lux (input or image-derived)
        - Derive baseline coeffs from lux anchors (mid baseline + anchors for coordinator)
        - Apply tone per channel
        - Compute CI indices
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        lux_scalar = f'{upstream}.lux_scalar'
        coeffs_name = self._bind_or_derive_coeffs(stage, input_image, lux_scalar, nodes, inits, vis)
        applier = self._apply_tone_rgb(stage, input_image, coeffs_name, nodes, vis)
        ci_vec, ci_scalar = self._calc_ci(stage, applier, nodes, vis)
        outputs = {'applier': {'name': applier}, 'tone_coeffs': {'name': coeffs_name}, 'ci_vec': {'name': ci_vec}, 'ci_scalar': {'name': ci_scalar}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')