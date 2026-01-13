import onnx
import onnx.helper as oh
from microblocks.base import MicroblockBase

class BayerNormBase(MicroblockBase):
    """
    Normalizer microblock.
    Consumes desc.image [n,c,h,w], selects the first image,
    reshapes to [1,4,h,w], and normalizes 12-bit raw values to [0,1].
    """

    name = "bayernorm_base"
    version = "v0"
    deps = ["image_desc_base"]
    needs = ["norm_scale"]

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f"{stage}.image"
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.image"   # <-- consume desc.image

        # Slice: take only the first image (index 0 along n)
        slice_out = f"{stage}.sliced"
        starts = f"{stage}.starts"
        ends   = f"{stage}.ends"
        axes   = f"{stage}.axes"

        slice_node = oh.make_node(
            "Slice",
            inputs=[input_image, starts, ends, axes],
            outputs=[slice_out],
            name=f"{stage}_slice_first"
        )

        # Reshape to [1,4,h,w]
        reshape_out = f"{stage}.reshaped"
        shape_tensor = f"{stage}.target_shape"

        reshape_node = oh.make_node(
            "Reshape",
            inputs=[slice_out, shape_tensor],
            outputs=[reshape_out],
            name=f"{stage}_reshape"
        )

        # Normalize by norm_scale
        norm_scale = f"{stage}.norm_scale"
        div_node = oh.make_node(
            "Div",
            inputs=[reshape_out, norm_scale],
            outputs=[out_name],
            name=f"{stage}_normalize"
        )

        vis = [
            oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ["n","c","h","w"]),
            oh.make_tensor_value_info(starts, oh.TensorProto.INT64, [1]),
            oh.make_tensor_value_info(ends,   oh.TensorProto.INT64, [1]),
            oh.make_tensor_value_info(axes,   oh.TensorProto.INT64, [1]),
            oh.make_tensor_value_info(shape_tensor, oh.TensorProto.INT64, [4]),
            oh.make_tensor_value_info(norm_scale, oh.TensorProto.FLOAT, []),
            oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ["1","4","h","w"]),
        ]

        outputs = {"applier": {"name": out_name}}
        return outputs, [slice_node, reshape_node, div_node], [], vis
