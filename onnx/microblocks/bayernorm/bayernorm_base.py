# bayernorm.py
from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class BayerNormBase(MicroblockBase):
    """
    BayerNorm Base Class
    Input:  [n,1,h,w] float32 Bayer mosaic
    Output: [n,1,h,w] normalized float in [0,1]
    Purpose: Scale raw sensor values into canonical range for ISP stages.
    """
    version = "v0"
    provides = ["applier"]

    # Subclasses must override this
    bit_depth: int = None

    def build_algo(self, stage, prev_stages=None):
        if self.bit_depth is None:
            raise ValueError("bit_depth must be set in subclass")

        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        out_name    = f"{stage}.applier"

        max_val = (1 << self.bit_depth) - 1  # e.g. 1023 for 10‑bit, 4095 for 12‑bit
        scale_const = f"{stage}.scale"

        # Divide input by scale constant
        nodes = [
            oh.make_node("Div", [input_image, scale_const],
                         [out_name], name=f"{stage}_normalize")
        ]

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",1,"h","w"]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ["n",1,"h","w"]),
        ]

        # Initializer for scale constant
        init = [
            oh.make_tensor(scale_const, TensorProto.FLOAT, [], [float(max_val)])
        ]

        outputs = {
            "applier": {"name": out_name},
            #"scale_factor": {"name": scale_const},
        }

        result = BuildResult(outputs, nodes, init, vis)
        result.appendInput(input_image)
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_applier(self, stage, prev_stages=None):
        return self.build_algo(stage, prev_stages)


# Subclasses for specific bit depths
class BayerNorm10(BayerNormBase):
    name = "bayernorm_10bit"
    bit_depth = 10

class BayerNorm12(BayerNormBase):
    name = "bayernorm_12bit"
    bit_depth = 12

class BayerNormBaseV0(BayerNorm10):
    name = "bayernorm_base"
