from microblocks.base import MicroblockBase
from onnx import helper

@register_block
class ResizeV1(MicroBlock):
    name = "resize"
    version = "v1"

    def input_names(self):
        return ["input"]

    def output_names(self):
        return ["output"]

    def build_algo_node(self, prev_out=None):
        from onnx import helper
        node = helper.make_node(
            "Identity",
            inputs=[prev_out or "input"],
            outputs=["output"],
            name="ResizeV1AlgoStub"
        )
        return node

    def build_applier_node(self, prev_out=None):
        from onnx import helper
        node = helper.make_node(
            "Identity",
            inputs=[prev_out or "input"],
            outputs=["output"],
            name="ResizeV1ApplierStub"
        )
        return node

    def build_coordinator_node(self, prev_out=None):
        from onnx import helper
        node = helper.make_node(
            "Identity",
            inputs=[prev_out or "input"],
            outputs=["output"],
            name="ResizeV1CoordStub"
        )
        return node
