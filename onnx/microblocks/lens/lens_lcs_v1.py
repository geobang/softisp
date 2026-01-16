from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from .lens_lcs_base import LensLCSBase

class LensLCSV1(LensLCSBase):
    """
    LensLCSV1 (v1)
    --------------
    Adaptive lens shading correction block.

    Needs:
        - applier [n,3,h,w]       : image tensor from upstream
        - lcs_coeffs [H,W]        : full-resolution correction coefficients (graph input)
        - resize_factor []        : scalar (e.g. 0.5 for half-res)

    Provides:
        - applier [n,3,h*,w*] : corrected image tensor
        - lcs_coeffs_resized [h*,w*] : resized coefficient map
    """
    name = 'lens_lcs_v1'
    version = 'v1'
    needs = ['applier', 'lcs_coeffs', 'resize_factor']
    provides = ['applier', 'lcs_coeffs_resized']

    def build_algo(self, stage: str, prev_stages=None):
        """
        Resize lcs_coeffs according to resize_factor, and forward original coeffs.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        lcs_coeffs = f'{stage}.lcs_coeffs'
        vis.append(oh.make_tensor_value_info(lcs_coeffs, TensorProto.FLOAT, ['H', 'W']))
        resize_factor = f'{upstream}.resize_factor'
        one_n = f'{stage}.one_n'
        one_c = f'{stage}.one_c'
        inits += [oh.make_tensor(one_n, TensorProto.FLOAT, [], [1.0]), oh.make_tensor(one_c, TensorProto.FLOAT, [], [1.0])]
        scales = f'{stage}.scales'
        nodes.append(oh.make_node('Concat', inputs=[one_n, one_c, resize_factor, resize_factor], outputs=[scales], name=f'{stage}.concat_scales', axis=0))
        lcs_resized = f'{stage}.lcs_coeffs_resized'
        nodes.append(oh.make_node('Resize', inputs=[lcs_coeffs, scales], outputs=[lcs_resized], name=f'{stage}.resize_lcs', mode='linear'))
        vis.append(oh.make_tensor_value_info(lcs_resized, TensorProto.FLOAT, ['h*', 'w*']))
        lcs_coeffs_out = f'{stage}.lcs_coeffs_out'
        nodes.append(oh.make_node('Identity', inputs=[lcs_coeffs], outputs=[lcs_coeffs_out], name=f'{stage}.identity_lcs'))
        vis.append(oh.make_tensor_value_info(lcs_coeffs_out, TensorProto.FLOAT, ['H', 'W']))
        applier = f'{stage}.applier'
        nodes.append(oh.make_node('Mul', inputs=[input_image, lcs_resized], outputs=[applier], name=f'{stage}.algo_mul_apply'))
        outputs = {'applier': {'name': applier}, 'lcs_coeffs_resized': {'name': lcs_resized}, 'lcs_coeffs': {'name': lcs_coeffs_out}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply lcs_coeffs (external input) to applier (image).
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['n', 3, 'h*', 'w*']))
        lcs_coeffs = f'{stage}.lcs_coeffs'
        vis.append(oh.make_tensor_value_info(lcs_coeffs, TensorProto.FLOAT, ['h*', 'w*']))
        applier = f'{stage}.applier'
        nodes.append(oh.make_node('Mul', inputs=[input_image, lcs_coeffs], outputs=[applier], name=f'{stage}.mul_apply'))
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h*', 'w*']))
        outputs = {'applier': {'name': applier}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')