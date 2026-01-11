from registry import register_block
from microblocks.base import MicroBlock
from onnx import helper









@register_block
class CCMBlockV1(MicroBlock):
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

    name = "ccm"
    version = "v1"
    coeff_names = ['matrix']

    def build_algo_node(self, prev_out):
        from onnx import helper
        node = helper.make_node(
            "Identity",
            inputs=[prev_out or "input"],
            outputs=["output"],
            name="CcmV1AlgoStub"
        )
        return node
