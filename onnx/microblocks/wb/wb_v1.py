import onnx
import onnx.helper as oh
import numpy as np
from microblocks.base import MicroblockBase
from microblocks.wb.wb_base import AWBBase

class WhiteBalanceV1(AWBBase):
    """
    Auto White Balance (AWB) microblock v1.
    Uses gray-world assumption to compute per-channel gains
    directly inside the ONNX graph.
    """

    name = "awb_base"
    version = "v1"
    deps = ["demosaic_base"]
    needs = []  # gains are computed internally

    def build_applier(self, stage: str, prev_stages=None):
        # Input image comes from previous stage
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        wb_out = f"{stage}.applier"

        # Apply gains to image
        mul_node = oh.make_node(
            "Mul",
            [input_image, f"{stage}.wb_gains"],
            [wb_out],
            name=f"{stage}_apply_wb"
        )

        vis = [
            oh.make_tensor_value_info(input_image, onnx.TensorProto.FLOAT, ["n","c","h","w"]),
            oh.make_tensor_value_info(f"{stage}.wb_gains", onnx.TensorProto.FLOAT, [3]),
            oh.make_tensor_value_info(wb_out, onnx.TensorProto.FLOAT, ["n","c","h","w"]),
        ]

        outputs = {"applier": {"name": wb_out}}
        return outputs, [mul_node], [], vis

    def build_algo(self, stage: str, prev_stages=None):
        """
        Build ONNX subgraph to compute wb_gains [3].
        Uses ReduceMean over H,W and gray-world assumption.
        """
        upstream = prev_stages[0] if prev_stages else stage
        image_in = f"{upstream}.applier"
        reduce_out = f"{stage}.avg_channels"
        applier = f"{stage}.applier"

        reduce_node = oh.make_node(
            "ReduceMean",
            inputs=[image_in],
            outputs=[reduce_out],
            axes=[2,3],
            keepdims=0,
            name=f"{stage}_reduce_mean"
        )

        # Slice R,G,B averages
        r_out = f"{stage}.avg_r"
        g_out = f"{stage}.avg_g"
        b_out = f"{stage}.avg_b"

        slice_r = oh.make_node("Slice", [reduce_out, f"{stage}.r_start", f"{stage}.r_end"], [r_out], name=f"{stage}_slice_r")
        slice_g = oh.make_node("Slice", [reduce_out, f"{stage}.g_start", f"{stage}.g_end"], [g_out], name=f"{stage}_slice_g")
        slice_b = oh.make_node("Slice", [reduce_out, f"{stage}.b_start", f"{stage}.b_end"], [b_out], name=f"{stage}_slice_b")

        # Divisions
        gain_r = f"{stage}.gain_r"
        gain_b = f"{stage}.gain_b"
        div_r = oh.make_node("Div", [g_out, r_out], [gain_r], name=f"{stage}_div_r")
        div_b = oh.make_node("Div", [g_out, b_out], [gain_b], name=f"{stage}_div_b")

        # Constant gain_g = 1.0
        gain_g = f"{stage}.gain_g"
        const_g = oh.make_node(
            "Constant",
            inputs=[],
            outputs=[gain_g],
            value=oh.make_tensor("", onnx.TensorProto.FLOAT, [1], [1.0]),
            name=f"{stage}_const_g"
        )

        # Concat gains
        wb_out = f"{stage}.wb_gains"
        concat_node = oh.make_node("Concat", [gain_r, gain_g, gain_b], [wb_out], axis=0, name=f"{stage}_concat")

        # Initializers for slice indices
        inits = [
            oh.make_tensor(f"{stage}.r_start", onnx.TensorProto.INT64, [1], [0]),
            oh.make_tensor(f"{stage}.r_end",   onnx.TensorProto.INT64, [1], [1]),
            oh.make_tensor(f"{stage}.g_start", onnx.TensorProto.INT64, [1], [1]),
            oh.make_tensor(f"{stage}.g_end",   onnx.TensorProto.INT64, [1], [2]),
            oh.make_tensor(f"{stage}.b_start", onnx.TensorProto.INT64, [1], [2]),
            oh.make_tensor(f"{stage}.b_end",   onnx.TensorProto.INT64, [1], [3]),
        ]

        nodes = [reduce_node, slice_r, slice_g, slice_b, div_r, div_b, const_g, concat_node]
        vis = [
            oh.make_tensor_value_info(image_in, onnx.TensorProto.FLOAT, ["n","c","h","w"]),
            oh.make_tensor_value_info(wb_out, onnx.TensorProto.FLOAT, [3]),
        ]

        mul_node = oh.make_node(
            "Mul",
            [image_in, f"{stage}.wb_gains"],
            [applier],
            name=f"{stage}_apply_wb"
        )
        nodes += [mul_node]

        outputs = {
            "applier": {"name": applier},
            "wb_gains": {"name": wb_out},
        }
        return outputs, nodes, inits, vis
