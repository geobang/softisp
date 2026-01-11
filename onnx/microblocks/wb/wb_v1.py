from registry import register_block
from microblocks.base import MicroBlock
from onnx import helper


@register_block
class WBBlockV1(MicroBlock):
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
            name="AlgoStub"
        )
        return node

    def build_applier_node(self, prev_out=None):
        from onnx import helper
        node = helper.make_node(
            "Identity",
            inputs=[prev_out or "input"],
            outputs=["output"],
            name="ApplierStub"
        )
        return node

    def build_coordinator_node(self, prev_out=None):
        from onnx import helper
        node = helper.make_node(
            "Identity",
            inputs=[prev_out or "input"],
            outputs=["output"],
            name="CoordinatorStub"
        )
        return node

    name = "wbblock"
    version = "v1"

    def __init__(self, stage, block, version, **kwargs):
        # Pass stage, block, version to the base class
        super().__init__(stage, block, version, **kwargs)

    def build(self):
        # Example ONNX node creation for white balance
        node = helper.make_node(
            "Mul",
            inputs=["input", "gain_r", "gain_g", "gain_b"],
            outputs=["output"],
            name="WBBlockV1"
        )
        return [node]
