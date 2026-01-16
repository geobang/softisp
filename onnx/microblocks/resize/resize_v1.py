from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from .resize_base import ResizeBase

class ResizeV1(ResizeBase):
    name = 'resize_v1'
    version = 'v1'
    needs = ['']
    provides = ['resize_factor']

    def build_algo(self, stage: str, prev_stages=None):
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        rgb = f'{upstream}.applier'
        applier = f'{stage}.applier'
        scales = f'{stage}.scales'

        # Default scales initializer
        inits.append(
            oh.make_tensor(scales, TensorProto.FLOAT, [4], [1.0, 1.0, 0.5, 0.5])
        )

        # Default resize factor initializer
        resize_factor = f'{stage}.resize_factor'
        inits.append(
            oh.make_tensor(resize_factor, TensorProto.FLOAT, [], [0.5])
        )

        # Identity node to replicate resize_factor as a visible output
        resize_factor_out = f'{stage}.resize_factor_out'
        nodes.append(
            oh.make_node("Identity", inputs=[resize_factor], outputs=[resize_factor_out])
        )
        vis.append(
            oh.make_tensor_value_info(resize_factor_out, TensorProto.FLOAT, [])
        )

        # Resize node
        nodes.append(
            oh.make_node(
                'Resize',
                inputs=[rgb, scales],
                outputs=[applier],
                name=f'{stage}.resize',
                mode='linear'
            )
        )
        vis.append(
            oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h/2', 'w/2'])
        )

        # Outputs: applier and the replicated resize_factor_out
        outputs = {
            'applier': {'name': applier},
            'resize_factor': {'name': resize_factor_out}
        }

        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{upstream}.applier')

    def build_applier(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages=prev_stages)
