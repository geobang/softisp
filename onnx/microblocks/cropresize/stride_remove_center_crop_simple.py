# stride_remove_center_crop_simple.py
from onnx import helper, TensorProto
from .base import MicroblockBase

class StrideRemoveCenterCrop(MicroblockBase):
    """
    Simplified stride-remove center crop using ONNX Slice.
    Consumes [image, h, w, c, stride], outputs [image, h, w, c].
    """

    name = "stride_remove_center_crop"
    version = "v2"
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
        # Starts = [0, (stride - w)//2, 0]
        starts = helper.make_tensor(
            name="Raw.crop_starts",
            data_type=TensorProto.INT64,
            dims=[3],
            vals=[0, 0, 0]  # placeholder, coordinator fills in dynamically
        )
        # Ends = [h, (stride - w)//2 + w, c]
        ends = helper.make_tensor(
            name="Raw.crop_ends",
            data_type=TensorProto.INT64,
            dims=[3],
            vals=[0, 0, 0]  # placeholder, coordinator fills in dynamically
        )
        # Axes = [0, 1, 2] (H, W, C)
        axes = helper.make_tensor(
            name="Raw.crop_axes",
            data_type=TensorProto.INT64,
            dims=[3],
            vals=[0, 1, 2]
        )

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

        return outputs, "Raw.image_cropped", [slice_node], [starts, ends, axes], value_info

    def build_coordinator(self, prev_out: str):
        return None
