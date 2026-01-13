import onnx.helper as oh
from microblocks.base import MicroblockBase

class ChromaSubsampleBase(MicroblockBase):
    """
    Chroma subsampling microblock.
    Input:  [n,3,target_h,target_w] (YUV)
    Output: [n,3,target_h,target_w/2] (YUV 4:2:0)
    Needs:  subsample_scale [2] (scale factors for H and W)
    """

    name = "chroma_subsample_base"
    version = "v0"
    deps = ["yuvconvert_base"]
    needs = ["subsample_scale"]

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f"{stage}.applier"
        scale    = f"{stage}.subsample_scale"

        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # ONNX Resize requires inputs: X, roi, scales, sizes
        # We provide X, empty roi, scales, and leave sizes blank
        node = oh.make_node(
            "Resize",
            inputs=[input_image, "", scale],
            outputs=[out_name],
            name=f"{stage}_chroma",
            mode="nearest"
        )

        vis = [
            oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
            oh.make_tensor_value_info(scale, oh.TensorProto.FLOAT, [4]),  # scales for n,c,h,w
            oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ["n","3","target_h","target_w/2"]),
        ]

        outputs = {"applier": {"name": out_name}}
        return outputs, [node], [], vis
