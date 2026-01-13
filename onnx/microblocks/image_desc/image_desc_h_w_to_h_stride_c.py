# image_desc_h_w_to_h_stride_c.py
from onnx import helper, TensorProto
from .base import MicroblockBase

class ImageDesc_H_W_to_H_Stride_C(MicroblockBase):
    """
    Mapping MB: consumes [image, h, w, c, stride],
    outputs [image, h, stride, c].
    """

    name = "imagedesc_h_w_to_h_stride_c"
    version = "v0"
    coeff_names = ["image", "h", "w", "c", "stride"]
    output_coeff_names = ["image", "h", "stride", "c"]

    process_method = "Identity"
    depends_on = ["ImageDesc"]

    def __init__(self, stride: int, channels: int = 3, dtype: int = TensorProto.FLOAT):
        super().__init__()
        self.stride = stride
        self.channels = channels
        self.dtype = dtype

    def build_algo(self, prev_out: str):
        return {
            "op_type": "Identity",
            "inputs": [prev_out],
            "outputs": ["Raw.image_stride"],
            "declared": {
                "image": "Raw.image_stride",
                "params": ["Raw.h", "Raw.stride", "Raw.c"]
            }
        }

    def build_applier(self, prev_out: str):
        node = helper.make_node(
            "Identity",
            inputs=[prev_out],
            outputs=["Raw.image_stride"],
            name=f"{self.name}_{self.version}"
        )

        outputs = {
            "image": {"name": "Raw.image_stride"},
            "params": [
                {"name": "Raw.h"},
                {"name": "Raw.stride"},
                {"name": "Raw.c"}
            ]
        }

        value_info = [
            helper.make_tensor_value_info("Raw.image_stride", self.dtype, ["H", self.stride, self.channels]),
            helper.make_tensor_value_info("Raw.h", TensorProto.INT64, []),
            helper.make_tensor_value_info("Raw.stride", TensorProto.INT64, []),
            helper.make_tensor_value_info("Raw.c", TensorProto.INT64, [])
        ]

        return outputs, "Raw.image_stride", [node], [], value_info
