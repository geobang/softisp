# demosaic_base.py
from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

def _slice_inputs_1d(stage, tag, starts, ends, axes):
    """
    Create INT64 initializers for Slice inputs (starts/ends/axes) for a single axis.
    These are referenced directly as inputs to Slice (opset >= 13).
    """
    s_name = f"{stage}.slice_{tag}_starts"
    e_name = f"{stage}.slice_{tag}_ends"
    a_name = f"{stage}.slice_{tag}_axes"
    inits = [
        oh.make_tensor(s_name, TensorProto.INT64, [len(starts)], starts),
        oh.make_tensor(e_name, TensorProto.INT64, [len(ends)],   ends),
        oh.make_tensor(a_name, TensorProto.INT64, [len(axes)],   axes),
    ]
    return (s_name, e_name, a_name), inits

class DemosaicBase(MicroblockBase):
    """
    Minimal demosaic base.
    Input:  [n,4,h/2,w/2] (R,G0,G1,B)
    Output: [n,3,h/2,w/2] (RGB, downsampled)
    """

    name = "demosaic_base"
    version = "v0"
    provides = ["applier"]

    def build_algo(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        cfa4 = f"{upstream}.applier"   # [n,4,h/2,w/2]
        out  = f"{stage}.applier"      # [n,3,h/2,w/2]

        nodes, inits, vis = [], [], []

        # Slice planes (opset-13: starts/ends/axes as inputs)
        R, G0, G1, B = [f"{stage}.{n}" for n in ("R","G0","G1","B")]
        (sR,eR,aR),     inits_R   = _slice_inputs_1d(stage, "R",  starts=[0], ends=[1], axes=[1])
        (sG0,eG0,aG0),  inits_G0  = _slice_inputs_1d(stage, "G0", starts=[1], ends=[2], axes=[1])
        (sG1,eG1,aG1),  inits_G1  = _slice_inputs_1d(stage, "G1", starts=[2], ends=[3], axes=[1])
        (sB,eB,aB),     inits_B   = _slice_inputs_1d(stage, "B",  starts=[3], ends=[4], axes=[1])
        inits += inits_R + inits_G0 + inits_G1 + inits_B

        nodes += [
            oh.make_node("Slice", [cfa4, sR,  eR,  aR],  [R],  name=f"{stage}_slice_R"),
            oh.make_node("Slice", [cfa4, sG0, eG0, aG0], [G0], name=f"{stage}_slice_G0"),
            oh.make_node("Slice", [cfa4, sG1, eG1, aG1], [G1], name=f"{stage}_slice_G1"),
            oh.make_node("Slice", [cfa4, sB,  eB,  aB],  [B],  name=f"{stage}_slice_B"),
        ]

        # G = 0.5 * (G0 + G1)
        G_sum  = f"{stage}.G_sum"
        G_half = f"{stage}.G"
        half_const = oh.make_tensor(name=f"{stage}.const_half",
                                    data_type=TensorProto.FLOAT,
                                    dims=[1], vals=[0.5])
        inits.append(half_const)

        nodes += [
            oh.make_node("Add", [G0, G1], [G_sum], name=f"{stage}_sum_G"),
            oh.make_node("Mul", [G_sum, f"{stage}.const_half"], [G_half], name=f"{stage}_avg_G"),
        ]

        # Concat R,G,B → RGB
        nodes += [oh.make_node("Concat", [R, G_half, B], [out], axis=1, name=f"{stage}_concat_RGB")]

        vis += [
            oh.make_tensor_value_info(cfa4, TensorProto.FLOAT, ["n",4,"h2","w2"]),
            oh.make_tensor_value_info(out,  TensorProto.FLOAT, ["n",3,"h2","w2"]),
        ]

        outputs = {"applier": {"name": out}}
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(cfa4)
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_applier(self, stage, prev_stages=None):
        return self.build_algo(stage, prev_stages)
