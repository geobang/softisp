from microblocks.base import BuildResult
import onnx.helper as oh
from microblocks.base import MicroblockBase
from onnx import TensorProto  # optional, for consistency


class AWBBase(MicroblockBase):
    """
    Auto White Balance microblock.
    Input:  [n,3,h,width] (RGB)
    Output: [n,3,h,width] (RGB after channel gains)
    Needs:  wb_gains [3] (R,G,B multipliers)
    """
    name = 'awb_base'
    version = 'v0'

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f'{stage}.applier'
        gains = f'{stage}.wb_gains'
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'

        # Node: apply per-channel gains
        node = oh.make_node('Mul', inputs=[input_image, gains], outputs=[out_name], name=f'{stage}_awb')

        # ValueInfos (optional but helpful for audit)
        vis = [
            oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ['n', '3', 'h', 'width']),
            oh.make_tensor_value_info(gains,    oh.TensorProto.FLOAT, [3]),
            oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ['n', '3', 'h', 'width']),
        ]

        outputs = {'applier': {'name': out_name}}

        # Explicit external inputs:
        # - input_image: dependent (from upstream)
        # - gains: independent (must be promoted to graph input)
        result = BuildResult(outputs, [node], [], vis)
        result.appendInput(input_image)
        result.appendInput(gains)          # <-- minimal fix
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Coordinator: stabilizes wb_gains by clipping the delta
        relative to previous frame gains.
        """
        # ---- Wiring trunk ----
        curr_gains  = f'{stage}.wb_gains'       # algo output (external input)
        prev_gains  = f'{stage}.wb_gains_prev'  # previous stabilized gains
        delta_name  = f'{stage}.delta'
        clipped     = f'{stage}.delta_clipped'
        out_gains   = f'{stage}.wb_gains_out'

        # ---- Node trunk ----
        sub_node = oh.make_node(
            'Sub', inputs=[curr_gains, prev_gains],
            outputs=[delta_name], name=f'{stage}_awb_delta'
        )

        min_tensor = f'{stage}.min_delta'
        max_tensor = f'{stage}.max_delta'
        clip_node = oh.make_node(
            'Clip', inputs=[delta_name, min_tensor, max_tensor],
            outputs=[clipped], name=f'{stage}_awb_clip'
        )

        add_node = oh.make_node(
            'Add', inputs=[prev_gains, clipped],
            outputs=[out_gains], name=f'{stage}_awb_out'
        )

        # ---- Initializers trunk ----
        delta_init_min = oh.make_tensor(min_tensor, TensorProto.FLOAT, [1], [-0.2])
        delta_init_max = oh.make_tensor(max_tensor, TensorProto.FLOAT, [1], [0.2])

        # ---- ValueInfo trunk ----
        vis = [
            oh.make_tensor_value_info(curr_gains, TensorProto.FLOAT, [3]),
            oh.make_tensor_value_info(prev_gains, TensorProto.FLOAT, [3]),
            oh.make_tensor_value_info(out_gains,  TensorProto.FLOAT, [3]),
        ]

        # ---- Outputs trunk ----
        outputs = {'wb_gains_out': {'name': out_gains}}

        # ---- BuildResult trunk ----
        result = BuildResult(outputs, [sub_node, clip_node, add_node],
                             [delta_init_min, delta_init_max], vis)
        result.appendInput(curr_gains)   # algo output
        result.appendInput(prev_gains)   # previous gains
        return result
