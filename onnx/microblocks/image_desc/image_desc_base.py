# image_desc_base.py
from onnx import helper, TensorProto
#from microblocks.base import MicroblockBase
from microblocks.base import MicroblockBase

class ImageDescBase(MicroblockBase):
    """
    Base image descriptor.
    Declares [h, w, c, stride] as ONNX tensors for downstream blocks.
    """

    name = "image_desc_base"
    version = "v0"
    coeff_names = ["image", "h", "w", "c", "stride"]
    output_coeff_names = ["image", "h", "w", "c", "stride"]

    def __init__(self, dtype: int = TensorProto.FLOAT):
        super().__init__()
        self.dtype = dtype

    def build_algo(self, prev_out: str):
        return {
            "op_type": "Identity",
            "inputs": [prev_out],
            "outputs": ["Raw.image"],
            "declared": {
                "image": "Raw.image",
                "params": ["Raw.h", "Raw.w", "Raw.c", "Raw.stride"]
            }
        }

    def build_applier(self, prev_out: str):
        # Value info for descriptors
        value_info = [
            helper.make_tensor_value_info("Raw.image", self.dtype, ["H", "Stride", "C"]),
            helper.make_tensor_value_info("Raw.h", TensorProto.INT64, []),
            helper.make_tensor_value_info("Raw.w", TensorProto.INT64, []),
            helper.make_tensor_value_info("Raw.c", TensorProto.INT64, []),
            helper.make_tensor_value_info("Raw.stride", TensorProto.INT64, []),
        ]
        return {
            "image": {"name": "Raw.image"},
            "params": [
                {"name": "Raw.h"},
                {"name": "Raw.w"},
                {"name": "Raw.c"},
                {"name": "Raw.stride"}
            ]
        }, "Raw.image", [], [], value_info

    def build_coordinator(self, prev_out: str):
        return None
