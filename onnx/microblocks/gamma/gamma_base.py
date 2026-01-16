from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class GammaBase(MicroblockBase):
    """
    Gamma correction microblock.
    Input:  [n,3,target_h,target_w] (RGB)
    Output: [n,3,target_h,target_w] (gamma corrected RGB)
    Needs:  gamma_value (scalar float, e.g. 2.2)
    """
    name = 'gamma_base'
    version = 'v0'
    deps = ['tonemap_base']
    needs = ['gamma_value']
    provides = ['applier', 'gamma_value']

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f'{stage}.applier'
        gamma = f'{stage}.gamma_value'
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'

        # Apply gamma correction: Pow(input_image, gamma)
        node = oh.make_node(
            'Pow',
            inputs=[input_image, gamma],
            outputs=[out_name],
            name=f'{stage}_gamma'
        )

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']),
            oh.make_tensor_value_info(gamma, TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(out_name, TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w'])
        ]

        outputs = {'applier': {'name': out_name}}
        return BuildResult(outputs, [node], [], vis).appendInput(f'{upstream}.applier')

    def build_algo(self, stage: str, prev_stages=None):
        """Declare gamma_value with default, override, and SSA-safe output."""
        nodes, inits, vis = ([], [], [])

        # Stage-scoped gamma name
        gamma = f'{stage}.gamma_value'

        # 1) Default gamma as initializer
        default_gamma = 2.2
        inits.append(oh.make_tensor(gamma, TensorProto.FLOAT, [1], [default_gamma]))

        # 2) Promote to graph input (so runtime can override)
        vis.append(oh.make_tensor_value_info(gamma, TensorProto.FLOAT, [1]))

        # 3) Identity node to expose gamma_value as visible output (distinct name)
        gamma_out = f'{stage}.gamma_value_out'
        nodes.append(
            oh.make_node("Identity", inputs=[gamma], outputs=[gamma_out], name=f'{stage}.gamma_identity')
        )
        vis.append(oh.make_tensor_value_info(gamma_out, TensorProto.FLOAT, [1]))

        # 4) Pass-through image (algo stage doesn’t apply gamma)
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        out_name = f'{stage}.applier'
        nodes.append(
            oh.make_node('Identity', inputs=[input_image], outputs=[out_name], name=f'{stage}.identity')
        )
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']))
        vis.append(oh.make_tensor_value_info(out_name,   TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']))

        # 5) Outputs: image + replicated gamma_value_out
        outputs = {
            'applier':     {'name': out_name},
            'gamma_value': {'name': gamma_out},
        }

        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{upstream}.applier')
