# image_desc_stride_to_h_w_c.py
from onnx import helper, TensorProto
from .base import MicroblockBase

class ImageDesc_Stride_to_H_W_C(MicroblockBase):
    """
    Pseudo MB: converts [image, h, stride, c] back into [image, h, w, c, stride].
    Useful for feeding into geometry blocks that need explicit width.
    """

    name = "imagedesc_stride_to_h_w_c"
    version = "v0"
    coeff_names = ["image", "h", "stride", "c"]          # inputs
    output_coeff_names = ["image", "h", "w", "c", "stride"] # outputs

    process_method = "Identity"
    depends_on = ["ImageDesc_H_W_to_H_Stride_C"]

    def __init__(self, width: int, stride: int, channels: int = 3, dtype: int = TensorProto.FLOAT):
        super().__init__()
        self.width = width
        self.stride = stride
        self.channels = channels
        self.dtype = dtype

    def input_names(self):
        return ["input"]

    def output_names(self):
        return ["output"]

    def build_algo(self, prev_out: str):
        return {
            "op_type": "Identity",
            "inputs": [prev_out],
            "outputs": ["Raw.image_hwc"],
            "declared": {
                "image": "Raw.image_hwc",
                "params": ["Raw.h", "Raw.w", "Raw.c", "Raw.stride"]
            }
        }

    def build_applier(self, prev_out: str):
        # No actual tensor reshape — just re‑declare width alongside stride
        node = helper.make_node(
            "Identity",
            inputs=[prev_out],
            outputs=["Raw.image_hwc"],
            name=f"{self.name}_{self.version}"
        )

        outputs = {
            "image": {"name": "Raw.image_hwc"},
            "params": [
                {"name": "Raw.h"},
                {"name": "Raw.w"},       # width restored
                {"name": "Raw.c"},
                {"name": "Raw.stride"}
            ]
        }

        value_info = [
            helper.make_tensor_value_info("Raw.image_hwc", self.dtype, ["H", self.width, self.channels]),
            helper.make_tensor_value_info("Raw.h", TensorProto.INT64, []),
            helper.make_tensor_value_info("Raw.w", TensorProto.INT64, []),
            helper.make_tensor_value_info("Raw.c", TensorProto.INT64, []),
            helper.make_tensor_value_info("Raw.stride", TensorProto.INT64, [])
        ]

        # Bake width constant for audit
        width_init = helper.make_tensor("Raw.width_const", TensorProto.INT64, [1], [self.width])

        return outputs, "Raw.image_hwc", [node], [width_init], value_info

    def build_coordinator(self, prev_out: str):
        return None
