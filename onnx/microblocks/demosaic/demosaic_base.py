# demosaic_base.py
from onnx import helper, TensorProto
from .base import MicroblockBase

class DemosaicBase(MicroblockBase):
    """
    Base demosaic microblock.
    Contract:
      - Consumes stride-free, black-level-corrected Bayer: [image, h, w, c=1]
      - Produces RGB image: [image, h, w, c=3]
    Notes:
      - Subclasses must provide the actual kernel initializers and Conv node wiring.
      - This base sets the canonical declared contract and op scaffolding.
    """

    name = "demosaic_base"
    version = "v0"
    coeff_names = ["image", "h", "w", "c"]
    output_coeff_names = ["image", "h", "w", "c"]
    process_method = "Conv"
    depends_on = ["blacklevel_v2"]  # ensure BLC precedes demosaic

    def __init__(self, dtype: int = TensorProto.FLOAT):
        super().__init__()
        self.dtype = dtype

    def build_algo(self, prev_out: str):
        # Canonical declaration: we will output RGB via Conv.
        # Kernel is provided by subclass (e.g., demosaic_v2).
        return {
            "op_type": "Conv",
            "inputs": [prev_out, "Raw.demosaic_kernel"],
            "outputs": ["Raw.image_rgb"],
            "declared": {
                "image": "Raw.image_rgb",
                "params": ["Raw.h", "Raw.w", "Raw.c_rgb"]
            }
        }

    def build_applier(self, prev_out: str):
        """
        Base provides output value_info and c_rgb param.
        Subclasses must provide:
          - Raw.demosaic_kernel initializer
          - Conv node wiring from prev_out -> Raw.image_rgb
        """
        # Channel count after demosaic: 3
        c_rgb = helper.make_tensor(
            name="Raw.c_rgb",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[3]
        )

        # Value info: output image HxWx3
        value_info = [
            helper.make_tensor_value_info("Raw.image_rgb", self.dtype, ["H", "W", 3]),
            helper.make_tensor_value_info("Raw.c_rgb", TensorProto.INT64, [1]),
            # Subclasses will add kernel value_info if desired
        ]

        # Base does not create the Conv node or kernel; subclass must.
        return {
            "image": {"name": "Raw.image_rgb"},
            "params": [
                {"name": "Raw.h"},
                {"name": "Raw.w"},
                {"name": "Raw.c_rgb"}
            ]
        }, "Raw.image_rgb", [], [c_rgb], value_info

    def build_coordinator(self, prev_out: str):
        return None
