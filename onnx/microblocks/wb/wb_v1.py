from microblocks.base import MicroblockBase

class WBBlockV1(MicroBlockBase):
    def input_names(self):
        return ["input"]

    def output_names(self):
        return ["output"]

    def build_algo(self, prev_out=None):
        inp = prev_out or "input"
        return helper.make_node(
            "Identity",
            inputs=[inp],
            outputs=["output"],
            name="WBBlockV1Algo"
        )

    def build_applier(self, prev_out=None):
        inp = prev_out or "input"
        return helper.make_node(
            "Mul",
            inputs=[inp, "wb_gains"],
            outputs=["output"],
            name="WBBlockV1Applier"
        )

    def build_coordinator(self, prev_out=None):
        inp = prev_out or "input"
        return helper.make_node(
            "Identity",
            inputs=[inp],
            outputs=["output"],
            name="WBBlockV1Coord"
        )

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
