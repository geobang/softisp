import onnx.helper as oh
from microblocks.base import MicroblockBase

class GammaBase(MicroblockBase):
    """
    Gamma correction microblock.
    Input:  [n,3,target_h,target_w] (RGB)
    Output: [n,3,target_h,target_w] (gamma corrected RGB)
    Needs:  gamma_value (scalar float, e.g. 2.2)
    """

    name = "gamma_base"
    version = "v0"
    deps = ["tonemap_base"]
    needs = ["gamma_value"]

    def build_applier(self, stage: str, prev_stages=None):
        out_name = f"{stage}.applier"
        gamma    = f"{stage}.gamma_value"

        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # Gamma correction: output = input^(1/gamma)
        node = oh.make_node(
            "Pow",
            inputs=[input_image, gamma],
            outputs=[out_name],
            name=f"{stage}_gamma"
        )

        vis = [
            oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
            oh.make_tensor_value_info(gamma, oh.TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(out_name, oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
        ]

        outputs = {"applier": {"name": out_name}}
        return outputs, [node], [], vis

    def build_algo(self, stage: str, prev_stages=None):
        """Declare gamma_value as an output of this stage."""
        nodes, inits, vis = [], [], []

        gamma = f"{stage}.gamma_value"

        # Value info for gamma_value
        vis.append(oh.make_tensor_value_info(gamma, oh.TensorProto.FLOAT, [1]))

        # You can either:
        # (A) leave gamma_value as a coordinator-provided input,
        # (B) or initialize it here with a default constant.
        # Example for (B):
        default_gamma = 2.2
        inits.append(
            oh.make_tensor(gamma, oh.TensorProto.FLOAT, [1], [default_gamma])
        )

        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        out_name    = f"{stage}.applier"

        # Identity node to forward input → output
        nodes.append(oh.make_node(
            "Identity",
            inputs=[input_image],
            outputs=[out_name],
            name=f"{stage}.identity"
        ))

        vis += [
            oh.make_tensor_value_info(input_image, oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
            oh.make_tensor_value_info(out_name,   oh.TensorProto.FLOAT, ["n","3","target_h","target_w"]),
        ]

        # Explicitly add the applier item to outputs
        outputs = {
            "applier": {"name": out_name},
            "gamma_value": {"name": gamma},
        }
        return outputs, nodes, inits, vis
