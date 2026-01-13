# stride_remove_crop.py
from onnx import helper, TensorProto
from microblocks.base import MicroblockBase

class StrideRemoveCrop(MicroblockBase):
    name = "stride_remove_crop"
    version = "v0"
    coeff_names = ["image", "h", "w", "c", "stride"]
    output_coeff_names = ["image", "h", "w", "c"]
    process_method = "Slice"
    depends_on = ["image_desc_base"]

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
        starts = helper.make_tensor("Raw.crop_starts", TensorProto.INT64, [3], [0,0,0])
        axes   = helper.make_tensor("Raw.crop_axes", TensorProto.INT64, [3], [0,1,2])

        ends_concat = helper.make_node(
            "Concat", ["Raw.h", "Raw.w", "Raw.c"],
            ["Raw.crop_ends"], axis=0, name="make_ends"
        )

        slice_node = helper.make_node(
            "Slice",
            inputs=[prev_out, "Raw.crop_starts", "Raw.crop_ends", "Raw.crop_axes"],
            outputs=["Raw.image_cropped"],
            name=f"{self.name}_{self.version}"
        )

        outputs = {
            "image": {"name": "Raw.image_cropped"},
            "params": [{"name": "Raw.h"}, {"name": "Raw.w"}, {"name": "Raw.c"}]
        }

        value_info = [
            helper.make_tensor_value_info("Raw.image_cropped", self.dtype, ["H","W","C"])
        ]

        return outputs, "Raw.image_cropped", [ends_concat, slice_node], [starts, axes], value_info

    def build_coordinator(self, prev_out: str):
        return None
