# stride_remove_center_crop_dynamic.py
from onnx import helper, TensorProto
from .base import MicroblockBase

class StrideRemoveCenterCrop(MicroblockBase):
    """
    Dynamic stride-remove center crop.
    Consumes [image, h, w, c, stride], outputs [image, h, w, c].
    All crop offsets are computed inside ONNX graph.
    """

    name = "stride_remove_center_crop"
    version = "v3"
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
        # Compute crop_left = (stride - w) // 2
        sub_node = helper.make_node(
            "Sub",
            inputs=["Raw.stride", "Raw.w"],
            outputs=["Raw.diff"],
            name="calc_diff"
        )
        div_node = helper.make_node(
            "Div",
            inputs=["Raw.diff", "const_two"],
            outputs=["Raw.crop_left"],
            name="calc_crop_left"
        )

        # Compute crop_right = crop_left + w
        add_node = helper.make_node(
            "Add",
            inputs=["Raw.crop_left", "Raw.w"],
            outputs=["Raw.crop_right"],
            name="calc_crop_right"
        )

        # Starts = [0, crop_left, 0]
        starts_concat = helper.make_node(
            "Concat",
            inputs=["const_zero", "Raw.crop_left", "const_zero"],
            outputs=["Raw.crop_starts"],
            axis=0,
            name="make_starts"
        )

        # Ends = [h, crop_right, c]
        ends_concat = helper.make_node(
            "Concat",
            inputs=["Raw.h", "Raw.crop_right", "Raw.c"],
            outputs=["Raw.crop_ends"],
            axis=0,
            name="make_ends"
        )

        # Axes = [0, 1, 2]
        axes_init = helper.make_tensor(
            name="Raw.crop_axes",
            data_type=TensorProto.INT64,
            dims=[3],
            vals=[0, 1, 2]
        )

        # Slice node
        slice_node = helper.make_node(
            "Slice",
            inputs=[prev_out, "Raw.crop_starts", "Raw.crop_ends", "Raw.crop_axes"],
            outputs=["Raw.image_cropped"],
            name=f"{self.name}_{self.version}"
        )

        # Constants
        const_zero = helper.make_tensor("const_zero", TensorProto.INT64, [1], [0])
        const_two  = helper.make_tensor("const_two",  TensorProto.INT64, [1], [2])

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
