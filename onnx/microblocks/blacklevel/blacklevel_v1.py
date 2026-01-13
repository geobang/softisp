from microblocks.base import MicroblockBase
from onnx import helper









@register_block
class BlackLevelV1(MicroBlock):
    def input_names(self):
        return ["input"]

    def output_names(self):
        return ["output"]

    name = "blacklevel"
    version = "v1"
    coeff_names = ["offset"]

    def build_algo_node(self, prev_out=None):
        inp = prev_out or "input"
        return helper.make_node(
            "Sub",
            inputs=[inp, "offset"],
            outputs=["output"],
            name="BlackLevelV1Algo"
        )

    def build_applier_node(self, prev_out=None):
        inp = prev_out or "input"
        return helper.make_node(
            "Identity",
            inputs=[inp],
            outputs=["output"],
            name="BlackLevelV1Applier"
        )

    def build_coordinator_node(self, prev_out=None):
        inp = prev_out or "input"
        return helper.make_node(
            "Identity",
            inputs=[inp],
            outputs=["output"],
            name="BlackLevelV1Coord"
        )
