import onnx.helper as oh
from microblocks.base import MicroblockBase

class StrideRemoveCropBase(MicroblockBase):
    name = "stride_remove_crop"
    version = "v0"
    deps = ["image_desc_base"]   # depends on ImageDescBase stage
    needs = [f"{name}.crop_starts", f"{name}.crop_ends"]  # external graph inputs

    def build_applier(self, stage: str, prev_stages=None):
        # Output tensor name
        out_name = f"{stage}.applier"

        # Graph input names for crop parameters
        starts   = f"{stage}.crop_starts"
        ends     = f"{stage}.crop_ends"

        # Upstream alias (ImageDescBase stage)
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.image"

        # Slice node:
        # - Input: [1, C, H, stride]
        # - Starts: [0, 0, 0, 0] → keep all batch, channels, height, start at col 0
        # - Ends:   [1, C, H, W] → crop last axis to width W (instead of stride)
        # - Axes:   [0, 1, 2, 3] → slice along all 4 dimensions
        node = oh.make_node(
            "Slice",
            inputs=[input_image, starts, ends],  # axes can be optional if full
            outputs=[out_name],
            name=f"{stage}_slice"
        )

        # Declare graph inputs for crop parameters
        vis = [
            # Starts array: always [0,0,0,0]
            oh.make_tensor_value_info(starts, oh.TensorProto.INT64, [4]),
            # Ends array: [1,C,H,W] → runtime feeds W instead of stride
            oh.make_tensor_value_info(ends,   oh.TensorProto.INT64, [4]),
            # Output tensor: cropped image [1,C,H,W]
            oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ["N","C","H","W"]),
        ]

        outputs = {"applier": {"name": out_name}}
        return outputs, [node], [], vis
