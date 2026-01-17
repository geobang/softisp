# bayer2cfa.py
from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class Bayer2CFABase(MicroblockBase):
    """
    Bayer2CFA Base Class
    Input:  [n,1,h,w] Bayer mosaic
    Output: [n,4,h/2,w/2] CFA planes normalized to [R,G,G,B]
    """

    version = "v0"
    provides = ["applier"]

    # Default pattern (subclasses override this)
    pattern = "RGGB"

    CFA_ORDER = {
        "RGGB": ["tl","tr","bl","br"],   # [R,G,G,B] → [0,1,2,3]
        "BGGR": ["br","tr","bl","tl"],   # reorder to [R,G,G,B]
        "GRBG": ["tr","tl","br","bl"],   # reorder to [R,G,G,B]
        "GBRG": ["bl","br","tl","tr"],   # reorder to [R,G,G,B]
    }

    CFA_MATRIX = {
        "RGGB": [0,1,2,3],
        "BGGR": [0,1,2,3],
        "GRBG": [0,1,2,3],
        "GBRG": [0,1,2,3],
    }

    def build_algo(self, stage, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        out_name    = f"{stage}.applier"

        # Slice into TL, TR, BL, BR
        tl = f"{stage}.tl"
        tr = f"{stage}.tr"
        bl = f"{stage}.bl"
        br = f"{stage}.br"

        nodes = [
            oh.make_node("Slice", [input_image], [tl],
                         name=f"{stage}_slice_tl", axes=[2,3], starts=[0,0], steps=[2,2]),
            oh.make_node("Slice", [input_image], [tr],
                         name=f"{stage}_slice_tr", axes=[2,3], starts=[0,1], steps=[2,2]),
            oh.make_node("Slice", [input_image], [bl],
                         name=f"{stage}_slice_bl", axes=[2,3], starts=[1,0], steps=[2,2]),
            oh.make_node("Slice", [input_image], [br],
                         name=f"{stage}_slice_br", axes=[2,3], starts=[1,1], steps=[2,2]),
        ]

        # Use fixed pattern content
        order = self.CFA_ORDER[self.pattern]
        concat_inputs = [locals()[pos] for pos in order]

        nodes.append(
            oh.make_node("Concat", concat_inputs,
                         [out_name], axis=1, name=f"{stage}_concat_cfa")
        )

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",1,"h","w"]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ["n",4,"h/2","w/2"]),
        ]

        outputs = {
            "applier": {"name": out_name},
            "cfa_matrix": self.CFA_MATRIX[self.pattern]
        }

        result = BuildResult(outputs, nodes, [], vis)
        result.appendInput(input_image)
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_applier(self, stage, prev_stages=None):
        return self.build_algo(stage, prev_stages)


# Pattern‑specific subclasses with explicit names
class Bayer2CFA_RGGB(Bayer2CFABase):
    name = "bayer2cfa_rggb"
    pattern = "RGGB"

class Bayer2CFA_BGGR(Bayer2CFABase):
    name = "bayer2cfa_bggr"
    pattern = "BGGR"

class Bayer2CFA_GRBG(Bayer2CFABase):
    name = "bayer2cfa_grbg"
    pattern = "GRBG"

class Bayer2CFA_GBRG(Bayer2CFABase):
    name = "bayer2cfa_gbrg"
    pattern = "GBRG"
