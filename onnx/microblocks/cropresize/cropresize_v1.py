from microblocks.base import MicroblockBase
from onnx import helper









@register_block
class CropResizeV1(MicroBlock):
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

    name = "cropresize"
    version = "v1"
    coeff_names = ["crop_x", "crop_y", "crop_w", "crop_h", "scale"]

    def build_algo_node(self, prev_out):
        return helper.make_node(
            "CropResizeOp",
            inputs=[prev_out],
            outputs=self.output_names()
        )
