# demosaic_v2.py
from onnx import helper, TensorProto
from .demosaic_base import DemosaicBase

class Demosaic_V2(DemosaicBase):
    """
    Demosaic v2: bilinear-like kernel (placeholder) implemented via Conv.
    Depends on BlackLevel_V2 and assumes stride has been removed.

    Input (from blacklevel_v2):
      - Raw.image (Bayer), Raw.h, Raw.w, Raw.c=1
    Output:
      - Raw.image_rgb (H, W, 3), Raw.h, Raw.w, Raw.c_rgb=3
    """

    name = "demosaic_v2"
    version = "v0"
    depends_on = ["blacklevel_v2"]

    def __init__(self, dtype: int = TensorProto.FLOAT):
        super().__init__(dtype=dtype)

    def build_algo(self, prev_out: str):
        # Reuse base contract; same declared op scaffold.
        return super().build_algo(prev_out)

    def build_applier(self, prev_out: str):
        # Prepare c_rgb param (3) and output value_info from base
        base_outputs, base_out_name, base_nodes, base_inits, base_vi = super().build_applier(prev_out)

        # A simple 2x2 averaging kernel to expand Bayer to 3 channels.
        # This is a placeholder; replace with your actual pattern-aware demosaic.
        # Shape: [out_channels=3, in_channels=1, kH=2, kW=2]
        kernel_init = helper.make_tensor(
            name="Raw.demosaic_kernel",
            data_type=TensorProto.FLOAT,
            dims=[3, 1, 2, 2],
            vals=[
                # R plane weights
                0.25, 0.25,
                0.25, 0.25,
                # G plane weights
                0.25, 0.25,
                0.25, 0.25,
                # B plane weights
                0.25, 0.25,
                0.25, 0.25
            ]
        )

        # Conv node to produce RGB
        conv_node = helper.make_node(
            "Conv",
            inputs=[prev_out, "Raw.demosaic_kernel"],
            outputs=["Raw.image_rgb"],
            name=f"{self.name}_{self.version}",
            # Stride/pad can be tuned; keep defaults for placeholder
        )

        # Merge base and v2 specifics
        outputs = base_outputs  # image name + params [h, w, c_rgb]
        nodes = base_nodes + [conv_node]
        initializers = base_inits + [kernel_init]
        value_info = base_vi + [
            helper.make_tensor_value_info("Raw.demosaic_kernel", TensorProto.FLOAT, [3, 1, 2, 2])
        ]

        return outputs, inputs[0], nodes, initializers, value_info

    def build_coordinator(self, prev_out: str):
        return None
