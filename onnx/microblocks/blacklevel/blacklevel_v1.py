from registry import register_block
from microblocks.base import MicroBlock
from onnx import helper









@register_block
class BlackLevelV1(MicroBlock):
    def input_names(self):
        return ["input"]

    def output_names(self):
        return ["output"]

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

    name = "blacklevel"
    version = "v1"
    coeff_names = ["offset"]

    def build_algo_node(self, prev_out):
        return helper.make_node(
            "BlackLevelOp",   # adjust to the actual op name in your build_blacklevel.py
            inputs=[prev_out],
            outputs=self.output_names()
        )
