# bayer2cfa.py
from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

INT64_MAX = 9223372036854775807  # "to the end" for Slice 'ends'

def _const_node(name: str, tensor):
    """Create a Constant node that outputs 'name' with given TensorProto."""
    return oh.make_node("Constant", inputs=[], outputs=[name], value=tensor, name=f"{name}_const")

def _slice_const_inputs(stage, tag, starts, ends, axes, steps):
    """
    Build Constant nodes for Slice inputs (starts/ends/axes/steps).
    Returns (names tuple, nodes list).
    """
    s_name = f"{stage}.slice_{tag}_starts"
    e_name = f"{stage}.slice_{tag}_ends"
    a_name = f"{stage}.slice_{tag}_axes"
    p_name = f"{stage}.slice_{tag}_steps"

    s_t = oh.make_tensor(s_name, TensorProto.INT64, [len(starts)], starts)
    e_t = oh.make_tensor(e_name, TensorProto.INT64, [len(ends)],   ends)
    a_t = oh.make_tensor(a_name, TensorProto.INT64, [len(axes)],   axes)
    p_t = oh.make_tensor(p_name, TensorProto.INT64, [len(steps)],  steps)

    nodes = [
        _const_node(s_name, s_t),
        _const_node(e_name, e_t),
        _const_node(a_name, a_t),
        _const_node(p_name, p_t),
    ]
    return (s_name, e_name, a_name, p_name), nodes

class Bayer2CFABase(MicroblockBase):
    """
    Bayer2CFA Base Class
    Input:  [n,1,h,w] Bayer mosaic
    Output: [n,4,h/2,w/2] CFA planes normalized to [R,G0,G1,B]
    Metadata:
      - cfa_onehot: [2,2,4] one-hot tile (R,G0,G1,B)
    """

    version = "v0"
    provides = ["applier", "cfa_onehot"]
    pattern = "RGGB"

    CFA_ORDER = {
        "RGGB": ["tl","tr","bl","br"],   # [R,G0,G1,B]
        "BGGR": ["br","tr","bl","tl"],
        "GRBG": ["tr","tl","br","bl"],
        "GBRG": ["bl","br","tl","tr"],
    }

    CFA_ONEHOT = {
        "RGGB": [
            [[1,0,0,0], [0,1,0,0]],
            [[0,0,1,0], [0,0,0,1]],
        ],
        "BGGR": [
            [[0,0,0,1], [0,0,1,0]],
            [[0,1,0,0], [1,0,0,0]],
        ],
        "GRBG": [
            [[0,1,0,0], [1,0,0,0]],
            [[0,0,0,1], [0,0,1,0]],
        ],
        "GBRG": [
            [[0,0,1,0], [0,0,0,1]],
            [[1,0,0,0], [0,1,0,0]],
        ],
    }

    def _make_cfa_onehot_initializer(self, stage):
        tile = self.CFA_ONEHOT[self.pattern]
        flat = [v for row in tile for vec in row for v in vec]
        const_name = f"{stage}.cfa_onehot_const"
        init = oh.make_tensor(const_name, TensorProto.INT64, [2,2,4], flat)
        vis = oh.make_tensor_value_info(const_name, TensorProto.INT64, [2,2,4])
        return const_name, init, vis

    def build_algo(self, stage, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        out_name    = f"{stage}.applier"

        nodes, inits, vis = [], [], []

        # Slice TL/TR/BL/BR using opset-13 Slice with Constant inputs
        tl = f"{stage}.tl"
        tr = f"{stage}.tr"
        bl = f"{stage}.bl"
        br = f"{stage}.br"

        (s_tl, e_tl, a_tl, p_tl), consts_tl = _slice_const_inputs(
            stage, "tl", starts=[0,0], ends=[INT64_MAX, INT64_MAX], axes=[2,3], steps=[2,2]
        )
        (s_tr, e_tr, a_tr, p_tr), consts_tr = _slice_const_inputs(
            stage, "tr", starts=[0,1], ends=[INT64_MAX, INT64_MAX], axes=[2,3], steps=[2,2]
        )
        (s_bl, e_bl, a_bl, p_bl), consts_bl = _slice_const_inputs(
            stage, "bl", starts=[1,0], ends=[INT64_MAX, INT64_MAX], axes=[2,3], steps=[2,2]
        )
        (s_br, e_br, a_br, p_br), consts_br = _slice_const_inputs(
            stage, "br", starts=[1,1], ends=[INT64_MAX, INT64_MAX], axes=[2,3], steps=[2,2]
        )

        nodes += consts_tl + consts_tr + consts_bl + consts_br

        nodes += [
            oh.make_node("Slice", [input_image, s_tl, e_tl, a_tl, p_tl], [tl], name=f"{stage}_slice_tl"),
            oh.make_node("Slice", [input_image, s_tr, e_tr, a_tr, p_tr], [tr], name=f"{stage}_slice_tr"),
            oh.make_node("Slice", [input_image, s_bl, e_bl, a_bl, p_bl], [bl], name=f"{stage}_slice_bl"),
            oh.make_node("Slice", [input_image, s_br, e_br, a_br, p_br], [br], name=f"{stage}_slice_br"),
        ]

        # Canonical RGGB order
        order = self.CFA_ORDER[self.pattern]
        concat_inputs = [locals()[pos] for pos in order]
        nodes.append(oh.make_node("Concat", concat_inputs, [out_name], axis=1, name=f"{stage}_concat_cfa"))

        # One-hot CFA tile: initializer → Identity → output (SSA-safe)
        cfa_const_name, cfa_const_init, cfa_const_vis = self._make_cfa_onehot_initializer(stage)
        cfa_out_name = f"{stage}.cfa_onehot"
        nodes.append(oh.make_node("Identity", [cfa_const_name], [cfa_out_name], name=f"{stage}_cfa_onehot_id"))

        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",1,"h","w"]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ["n",4,"h2","w2"]),
            cfa_const_vis,
            oh.make_tensor_value_info(cfa_out_name, TensorProto.INT64, [2,2,4]),
        ]

        outputs = {
            "applier":   {"name": out_name},
            "cfa_onehot":{"name": cfa_out_name},
        }

        result = BuildResult(outputs, nodes, [cfa_const_init], vis)
        result.appendInput(input_image)
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_applier(self, stage, prev_stages=None):
        return self.build_algo(stage, prev_stages)


# Pattern‑specific subclasses
class Bayer2CFA_RGGB(Bayer2CFABase):
    name = "bayer2cfa"
    pattern = "RGGB"
    version ="v0.rggb"

class Bayer2CFA_BGGR(Bayer2CFABase):
    name = "bayer2cfa"
    pattern = "BGGR"
    version ="v0.bggr"

class Bayer2CFA_GRBG(Bayer2CFABase):
    name = "bayer2cfa"
    pattern = "GRBG"
    version ="v0.grbg"

class Bayer2CFA_GBRG(Bayer2CFABase):
    name = "bayer2cfa"
    pattern = "GBRG"
    version ="v0.gbrg"
