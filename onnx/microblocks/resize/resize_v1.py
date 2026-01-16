from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from .resize_base import ResizeBase

class ResizeV1(ResizeBase):
    """
    ResizeV1 (v1)
    -------------
    Needs:
        - resize_factor [1] (scalar factor)

    Provides:
        - applier [n,3,h/2,w/2] : resized image
        - resize_factor_out [1] : visible output of resize factor
    """
    name = 'resize_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage

        # Names
        rgb              = f'{upstream}.applier'
        applier          = f'{stage}.applier'
        resize_factor    = f'{stage}.resize_factor'
        resize_factor_out = f'{stage}.resize_factor_out'
        scales           = f'{stage}.scales'

        # Constant "1.0" for batch and channel dimensions
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))

        # Build scales vector [1, 1, resize_factor, resize_factor]
        nodes.append(
            oh.make_node(
                'Concat',
                inputs=[one, one, resize_factor, resize_factor],
                outputs=[scales],
                name=f'{stage}.make_scales',
                axis=0
            )
        )
        vis.append(oh.make_tensor_value_info(scales, TensorProto.FLOAT, [4]))

        # Identity node to expose resize_factor
        nodes.append(
            oh.make_node("Identity", inputs=[resize_factor], outputs=[resize_factor_out], name=f'{stage}.factor_identity')
        )
        vis.append(oh.make_tensor_value_info(resize_factor_out, TensorProto.FLOAT, []))

        # Resize node
        nodes.append(
            oh.make_node(
                'Resize',
                inputs=[rgb, '', scales],  # second input (roi) left empty
                outputs=[applier],
                name=f'{stage}.resize',
                mode='linear'
            )
        )
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h/2', 'w/2']))

        # Outputs
        outputs = {
            'applier': {'name': applier},
            'resize_factor': {'name': resize_factor_out},
        }

        # BuildResult + declare external inputs
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(rgb)
        result.appendInput(resize_factor)
        return result

    def build_applier(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages=prev_stages)
