from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from .tone_filmic_v2 import ToneFilmicV2

class ToneFilmicV3(ToneFilmicV2):
    """
    ToneFilmicV3
    -------------
    Self-contained tone mapping microblock:
    - Independent of external parameters (only needs RGB image).
    - Outputs baseline tone coeffs and biased CI indices.
    - Coordinator can refine coeffs later without source image.
    """
    name = 'tonemap_filmic_v3'
    version = 'v3'
    deps = ['resize_base']
    needs = ['input_image', 'neutral_mask', 'ci_weights']
    provides = ['applier', 'tone_coeffs', 'ci_vec', 'ci_scalar']

    def _provide_lut_info(self, stage, inits, vis):
        lut_name = f'{stage}.lut_default'
        lut_vals = [self._filmic_curve(x / 255.0) for x in range(256)]
        inits.append(oh.make_tensor(name=lut_name, data_type=TensorProto.FLOAT, dims=[256], vals=lut_vals))
        vis.append(oh.make_tensor_value_info(lut_name, TensorProto.FLOAT, [256]))
        return lut_name

    def _filmic_curve(self, x):
        return x * (6.2 * x + 0.5) / (x * (6.2 * x + 1.7) + 0.06)

    def _calc_tone_coeffs(self, stage, in_img, nodes, vis):
        tone_coeffs = f'{stage}.tone_coeffs'
        out_img = f'{stage}.tone_applied'
        nodes.append(oh.make_node('Identity', inputs=[in_img], outputs=[out_img], name=f'{stage}.identity_tone'))
        vis.append(oh.make_tensor_value_info(out_img, TensorProto.FLOAT, ['n', '3', 'h', 'w']))
        vis.append(oh.make_tensor_value_info(tone_coeffs, TensorProto.FLOAT, [3, 4]))
        return (out_img, tone_coeffs)

    def _apply_tone(self, stage, out_img, nodes, vis):
        applier = f'{stage}.applier'
        nodes.append(oh.make_node('Identity', inputs=[out_img], outputs=[applier], name=f'{stage}.identity_applier'))
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', '3', 'h', 'w']))
        return applier

    def _calc_ci(self, stage, applier, in_mask, in_weights, nodes, vis):
        ci_vec = f'{stage}.ci_vec'
        ci_scalar = f'{stage}.ci_scalar'
        nodes.append(oh.make_node('Identity', inputs=[applier], outputs=[ci_vec], name=f'{stage}.identity_ci_vec'))
        nodes.append(oh.make_node('Identity', inputs=[applier], outputs=[ci_scalar], name=f'{stage}.identity_ci_scalar'))
        vis.append(oh.make_tensor_value_info(ci_vec, TensorProto.FLOAT, [3]))
        vis.append(oh.make_tensor_value_info(ci_scalar, TensorProto.FLOAT, [1]))
        return (ci_vec, ci_scalar)

    def build_algo(self, stage: str, prev_stages=None):
        vis, nodes, inits = ([], [], [])
        in_img = f'{stage}.input_image'
        in_mask = f'{stage}.neutral_mask'
        in_weights = f'{stage}.ci_weights'
        lut_name = self._provide_lut_info(stage, inits, vis)
        out_img, tone_coeffs = self._calc_tone_coeffs(stage, in_img, nodes, vis)
        applier = self._apply_tone(stage, out_img, nodes, vis)
        ci_vec, ci_scalar = self._calc_ci(stage, applier, in_mask, in_weights, nodes, vis)
        outputs = {'applier': {'name': applier}, 'tone_coeffs': {'name': tone_coeffs}, 'ci_vec': {'name': ci_vec}, 'ci_scalar': {'name': ci_scalar}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')