from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto


class CCMBase(MicroblockBase):
    """
    Color Correction Matrix (CCM) microblock.
    Input:  [n,3,h,w]   (RGB)
    Output: [n,3,h,w]   (RGB after CCM)
    Optional parameter:
        ccm [3,3] matrix
    """
    name = "ccm_base"
    version = "v0"

    # -------------------------------
    # Applier (runtime CCM application)
    # -------------------------------
    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        ccm = f"{stage}.ccm"
        out_name = f"{stage}.applier"

        # Apply CCM via matrix multiplication
        node = oh.make_node("MatMul", [input_image, ccm], [out_name], name=f"{stage}_ccm")

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", "3", "h", "w"]),
            oh.make_tensor_value_info(ccm, TensorProto.FLOAT, [3, 3]),
            oh.make_tensor_value_info(out_name, TensorProto.FLOAT, ["n", "3", "h", "w"]),
        ]

        outputs = {"applier": {"name": out_name}}

        result = BuildResult(outputs, [node], [], vis)
        result.appendInput(input_image)  # upstream image only
        result.appendInput(ccm)
        return result

    # -------------------------------
    # Algo (declare CCM + pass-through image)
    # -------------------------------
    def build_algo(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # Internal CCM parameter with default (identity matrix)
        ccm = f"{stage}.ccm"
        default_matrix = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        inits.append(oh.make_tensor(ccm, TensorProto.FLOAT, [3, 3], default_matrix))
        vis.append(oh.make_tensor_value_info(ccm, TensorProto.FLOAT, [3, 3]))

        # Identity to expose CCM as visible output
        ccm_out = f"{stage}.ccm_out"
        nodes.append(oh.make_node("Identity", [ccm], [ccm_out], name=f"{stage}.ccm_identity"))
        vis.append(oh.make_tensor_value_info(ccm_out, TensorProto.FLOAT, [3, 3]))

        # Pass-through image
        out_name = f"{stage}.applier"
        nodes.append(oh.make_node("Identity", [input_image], [out_name], name=f"{stage}.identity"))
        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", "3", "h", "w"]),
            oh.make_tensor_value_info(out_name,   TensorProto.FLOAT, ["n", "3", "h", "w"]),
        ]

        outputs = {
            "applier": {"name": out_name},
            "ccm":     {"name": ccm_out},  # visible output
        }

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image)  # upstream image only
        # ccm is optional: do not appendInput here
        return result
