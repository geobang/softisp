import onnx.helper as oh
from microblocks.base import MicroblockBase

class DemosaicBase(MicroblockBase):
    """
    Microblock that performs Bayer demosaicing.
    Input:  [n,4,h,width]  (RGGB mosaic channels)
    Output: [n,3,h,width]  (RGB image)
    Needs:  None (parameters are fixed for bilinear demo)
    """

    name = "demosaic_base"
    version = "v0"
    deps = ["black_level_base"]   # typically comes after black level correction
    needs = ["kernels"]

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f"{stage}.applier"

        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # For simplicity, use a Conv node with fixed kernels to simulate bilinear demosaic.
        # Each of the 4 Bayer planes is convolved into 3 RGB channels.
        # In practice, you’d load precomputed kernels into initializers.

        conv_out = out_name
        node = oh.make_node(
            "Conv",
            inputs=[input_image, f"{stage}.kernels"],
            outputs=[conv_out],
            name=f"{stage}_demosaic"
        )

        # Value infos
        vis = [
            oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ["n","4","h","width"]),
            oh.make_tensor_value_info(f"{stage}.kernels", oh.TensorProto.FLOAT, [3,4,3,3]),  # example kernel shape
            oh.make_tensor_value_info(conv_out, oh.TensorProto.FLOAT, ["n","3","h","width"]),
        ]

        outputs = {"applier": {"name": conv_out}}
        inits = []  # kernels would be added here as initializers

        return outputs, [node], inits, vis
