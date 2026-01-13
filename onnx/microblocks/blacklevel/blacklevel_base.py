# onnx/microblocks/blacklevel/blacklevel_base.py

from microblocks.base import MicroblockBase

class BlackLevelBase(MicroblockBase):
    """
    Base stub for BlackLevel microblocks.
    Provides the canonical contract so build_all.py can safely unpack
    five return values even if no real nodes are generated.
    """

    name = "blacklevel"
    version = "v0"

    def build_algo(self, prev_out):
        # Outputs dictionary: at minimum, propagate the image forward
        outputs = {
            "image": {"name": prev_out},
            "offset": {"name": "BlackLevel.offset"}  # placeholder coeff
        }
        consumed = prev_out
        nodes = []
        inits = []
        vis = []
        return outputs, consumed, nodes, inits, vis

    def build_applier(self, prev_out):
        # Default stub: no applier nodes
        outputs = {"image": {"name": prev_out}}
        consumed = prev_out
        nodes = []
        inits = []
        vis = []
        return outputs, consumed, nodes, inits, vis

    def build_coordinator(self, prev_out):
        # Default stub: no coordinator logic
        outputs = {"image": {"name": prev_out}}
        consumed = prev_out
        nodes = []
        inits = []
        vis = []
        return outputs, consumed, nodes, inits, vis

    def build_stub_outputs(self, prev_out):
        """Return outputs dict with image and canonical coeffs."""
        outputs = {"image": {"name": prev_out}}
        for coeff in self.coeff_names:
            outputs[coeff] = {"name": coeff}
        return outputs

    def build_stub_value_info(self, prev_out):
        """Return value_info list for canonical coeffs + image."""
        return [
            oh.make_tensor_value_info("raw.h", onnx.TensorProto.INT64, []),
            oh.make_tensor_value_info("raw.w", onnx.TensorProto.INT64, []),
            oh.make_tensor_value_info("raw.c", onnx.TensorProto.INT64, []),
            oh.make_tensor_value_info("offset", onnx.TensorProto.FLOAT, []),
            oh.make_tensor_value_info(prev_out, onnx.TensorProto.FLOAT, None),
        ]
