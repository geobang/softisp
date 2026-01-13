# stride_remove_crop_full.py
from onnx import helper, TensorProto
from .base import MicroblockBase

class StrideRemoveCrop(MicroblockBase):
    """
    Fully dynamic stride-remove crop.
    Consumes [image, h, w, c, stride], outputs [image, h, w, c].
    Crop offsets are computed inside ONNX graph using Sub/Div/Add/Concat.
    """

    name = "stride_remove_crop"
    version = "v1"
    coeff_names = ["image", "h", "w", "c", "stride"]
    output_coeff_names = ["image", "h", "w", "c"]

    process_method = "Slice"
    depends_on = ["ImageDesc"]

    def __init__(self, dtype: int = TensorProto.FLOAT):
        super().__init__()
        self.dtype = dtype

    def build_algo(self, prev_out: str):
        return {
            "op_type": "Slice",
            "inputs": [prev_out],
            "outputs": ["Raw.image_cropped"],
            "declared": {
                "image": "Raw.image_cropped",
                "params": ["Raw.h", "Raw.w", "Raw.c"]
            }
        }

    def build_applier(self, prev_out: str):
        # Constants
        const_zero = helper.make_tensor("const_zero", TensorProto.INT64, [1], [0])
        const_two  = helper.make_tensor("const_two",  TensorProto.INT64, [1], [2])
        axes_init  = helper.make_tensor("Raw.crop_axes", TensorProto.INT64, [3], [0,1,2])

        # Compute crop_left = (stride - w) // 2
        sub_node = helper.make_node(
            "Sub", ["Raw.stride", "Raw.w"], ["Raw.diff"], name="calc_diff"
        )
        div_node = helper.make_node(
            "Div", ["Raw.diff", "const_two"], ["Raw.crop_left"], name="calc_crop_left"
        )

        # Compute crop_right = crop_left + w
        add_node = helper.make_node(
            "Add", ["Raw.crop_left", "Raw.w"], ["Raw.crop_right"], name="calc_crop_right"
        )

        # Starts = [0, crop_left, 0]
        starts_concat = helper.make_node(
            "Concat", ["const_zero", "Raw.crop_left", "const_zero"],
            ["Raw.crop_starts"], axis=0, name="make_starts"
        )

        # Ends = [h, crop_right, c]
        ends_concat = helper.make_node(
            "Concat", ["Raw.h", "Raw.crop_right", "Raw.c"],
            ["Raw.crop_ends"], axis=0, name="make_ends"
        )

        # Slice to remove stride padding
        slice_node = helper.make_node(
            "Slice",
            inputs=[prev_out, "Raw.crop_starts", "Raw.crop_ends", "Raw.crop_axes"],
            outputs=["Raw.image_cropped"],
            name=f"{self.name}_{self.version}"
        )

        outputs = {
            "image": {"name": "Raw.image_cropped"},
            "params": [
                {"name": "Raw.h"},
                {"name": "Raw.w"},
                {"name": "Raw.c"}
            ]
        }

        value_info = [
            helper.make_tensor_value_info("Raw.image_cropped", self.dtype, ["H", "W", "C"]),
            helper.make_tensor_value_info("Raw.crop_starts", TensorProto.INT64, [3]),
            helper.make_tensor_value_info("Raw.crop_ends", TensorProto.INT64, [3]),
            helper.make_tensor_value_info("Raw.crop_axes", TensorProto.INT64, [3]),
        ]

        return outputs, "Raw.image_cropped", [
            sub_node, div_node, add_node,
            starts_concat, ends_concat, slice_node
        ], [const_zero, const_two, axes_init], value_info

    def build_coordinator(self, prev_out: str):
        return None
