from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class ToneMapBase(MicroblockBase):
    """
    Tone Mapping microblock.
    Input:  [n,3,target_h,target_w] (RGB)
    Output: [n,3,target_h,target_w] (tone-mapped RGB)
    Needs:  tonemap_curve (lookup table or curve parameters)
    """
    name = 'tonemap_base'
    version = 'v0'
    deps = ['resize_base']
    needs = ['tonemap_curve']
    provides = ['applier', 'tonemap_curve']

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f'{stage}.applier'
        curve = f'{stage}.tonemap_curve'
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'

        # Apply tone mapping (here simplified as a Mul with curve scalar)
        node = oh.make_node(
            'Mul',
            inputs=[input_image, curve],
            outputs=[out_name],
            name=f'{stage}_tonemap'
        )

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']),
            oh.make_tensor_value_info(curve, TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(out_name, TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w'])
        ]

        outputs = {'applier': {'name': out_name}}
        return BuildResult(outputs, [node], [], vis).appendInput(f'{upstream}.applier')

    def build_algo(self, stage: str, prev_stages=None):
        """Declare or initialize tonemap_curve for this stage (default + override + visible output)."""
        nodes, inits, vis = ([], [], [])

        # Stage-scoped curve name
        curve = f'{stage}.tonemap_curve'

        # 1) Default curve as initializer (scalar gain here; replace with LUT if needed)
        default_curve = [0.8]
        inits.append(oh.make_tensor(curve, TensorProto.FLOAT, [1], default_curve))

        # 2) Promote to graph input so runtime can override the default
        vis.append(oh.make_tensor_value_info(curve, TensorProto.FLOAT, [1]))

        # 3) Expose a visible output via Identity to satisfy SSA (distinct name)
        curve_out = f'{stage}.tonemap_curve_out'
        nodes.append(
            oh.make_node("Identity", inputs=[curve], outputs=[curve_out], name=f'{stage}.curve_identity')
        )
        vis.append(oh.make_tensor_value_info(curve_out, TensorProto.FLOAT, [1]))

        # 4) Pass-through image (algo stage doesn’t apply tone map)
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        out_name = f'{stage}.applier'
        nodes.append(
            oh.make_node('Identity', inputs=[input_image], outputs=[out_name], name=f'{stage}.identity')
        )
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']))
        vis.append(oh.make_tensor_value_info(out_name,   TensorProto.FLOAT, ['n', '3', 'target_h', 'target_w']))

        # 5) Outputs: image + replicated curve (SSA-safe)
        outputs = {
            'applier':       {'name': out_name},
            'tonemap_curve': {'name': curve_out},
        }

        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{upstream}.applier')
