from microblocks.base import MicroblockBase
from onnx import helper









@register_block
class ChromaSubsampleV1(MicroBlock):
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

    name = "chromasubsample"
    version = "v1"
    coeff_names = []

    def build_algo_node(self, prev_out):
        return helper.make_node(
            "ChromaSubsampleOp",
            inputs=[prev_out],
            outputs=self.output_names()
        )
