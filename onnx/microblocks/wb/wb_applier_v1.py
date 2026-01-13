from microblocks.base import MicroblockBase
from onnx import helper


@register_block
class WBApplierV1(MicroBlock):
    name = "wbapplier"
    version = "v1"

    def input_names(self):
        return ["input"]

    def output_names(self):
        return ["wb_out"]

    def build_algo_node(self, prev_out=None):
        # Applier usually isn’t part of algo, but stub it
        node = helper.make_node(
            "Identity",
            inputs=[prev_out or "input"],
            outputs=["wb_out"],
            name="WBApplierAlgoStub"
        )
        return node

    def build_applier_node(self, prev_out=None):
        node = helper.make_node(
            "Mul",
            inputs=[prev_out or "input", "gain_r", "gain_g", "gain_b"],
            outputs=["wb_out"],
            name="WBApplierV1"
        )
        return node

    def build_coordinator_node(self, prev_out=None):
        node = helper.make_node(
            "Identity",
            inputs=[prev_out or "wb_out"],
            outputs=["wb_coord_out"],
            name="WBApplierCoordStub"
        )
        return node








