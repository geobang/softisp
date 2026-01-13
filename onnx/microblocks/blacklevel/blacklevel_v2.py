# onnx/microblocks/blacklevel/blacklevel_v2.py

from microblocks.blacklevel.blacklevel_base import BlackLevelBase
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nh
import numpy as np

class BlackLevelV2(BlackLevelBase):
    """
    Version 2 of BlackLevel block.
    Applies a black level offset subtraction to the raw image.
    """

    name = "blacklevel"
    version = "v2"

    def build_algo(self, prev_out):
        # For algo mode, just propagate image forward
        outputs = self.build_stub_outputs(prev_out)
        consumed = prev_out
        nodes, inits, vis = [], [], []
        return outputs, consumed, nodes, inits, vis

    def build_applier(self, prev_out):
        # Build outputs using base helper (defines coeffs like raw.h, raw.w, raw.c, offset)
        outputs = self.build_stub_outputs(prev_out)
        consumed = prev_out

        # Create an initializer for the black level offset
        offset_name = "offset"
        offset_value = np.array([64.0], dtype=np.float32)  # example offset
        offset_init = nh.from_array(offset_value, name=offset_name)

        # Create a node that subtracts the offset from the image
        corrected_name = f"{self.name}_corrected"
        sub_node = oh.make_node(
            "Sub",
            inputs=[prev_out, offset_name],
            outputs=[corrected_name],
            name="BlackLevelSub"
        )

        # Update outputs to point to corrected image
        outputs["image"] = {"name": corrected_name}

        # Define value_info for each coeff
        value_info = [
            oh.make_tensor_value_info("raw.h", onnx.TensorProto.INT64, []),
            oh.make_tensor_value_info("raw.w", onnx.TensorProto.INT64, []),
            oh.make_tensor_value_info("raw.c", onnx.TensorProto.INT64, []),
            oh.make_tensor_value_info("offset", onnx.TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(corrected_name, onnx.TensorProto.FLOAT, None),
        ]

        # Validate contract
        self.validate_contract(outputs, value_info)

        nodes = [sub_node]
        inits = [offset_init]

        return outputs, consumed, nodes, inits, value_info

    def build_coordinator(self, prev_out):
        outputs = self.build_stub_outputs(prev_out)
        consumed = prev_out
        nodes, inits, vis = [], [], []
        return outputs, consumed, nodes, inits, vis
