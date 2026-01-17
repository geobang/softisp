# demosaic_mhc.py
from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto
from .demosaic_base import DemosaicBase
from microblocks.registry import Registry

def _kernel_init(stage, name, weights_2d):
    flat = [w for row in weights_2d for w in row]
    return oh.make_tensor(
        name=f"{stage}.{name}",
        data_type=TensorProto.FLOAT,
        dims=[1,1,5,5],
        vals=flat
    )

def mhc_kernel_inits(stage):
    # Canonical 5x5 MHC-like kernels (example weights; tune as needed)
    return [
        _kernel_init(stage, "K_mhc_g_r",  [[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]]),
        _kernel_init(stage, "K_mhc_g_b",  [[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]]),
        _kernel_init(stage, "K_mhc_r_b",  [[0,0,-1,0,0],[0,0,2,0,0],[-1,2,5,2,-1],[0,0,2,0,0],[0,0,-1,0,0]]),
        _kernel_init(stage, "K_mhc_b_r",  [[0,0,-1,0,0],[0,0,2,0,0],[-1,2,5,2,-1],[0,0,2,0,0],[0,0,-1,0,0]]),
        _kernel_init(stage, "K_mhc_r_g_h",[[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]]),
        _kernel_init(stage, "K_mhc_r_g_v",[[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]]),
        _kernel_init(stage, "K_mhc_b_g_h",[[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]]),
        _kernel_init(stage, "K_mhc_b_g_v",[[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]]),
    ]

def _slice_inputs_1d(stage, tag, starts, ends, axes):
    """
    Create INT64 initializers for Slice inputs (starts/ends/axes) for a single axis.
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

def _tile_repeats_init(stage, repeats):
    """
    Create INT64 initializer for Tile repeats.
    repeats is a list like [h2, w2, 1] — if you use fixed sizes, put concrete ints here.
    """
    name = f"{stage}.tile_repeats"
    return name, oh.make_tensor(name, TensorProto.INT64, [len(repeats)], repeats)

class DemosaicMHC(DemosaicBase):
    """
    MHC demosaic microblock.
    Algo: inherits minimal downsample averaging from DemosaicBase.
    Applier: full-resolution reconstruction using mask-gated convolutions.
    """

    name = "demosaic_mhc"
    version = "v0"
    provides = ["applier"]

    def build_algo(self, stage, prev_stages=None):
        # Algo path: just reuse the minimal downsample averaging
        return super().build_algo(stage, prev_stages)

    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        cfa4 = f"{upstream}.applier"          # [n,4,h/2,w/2]
        #cfa_onehot = f"{upstream}.cfa_onehot" # [2,2,4]
        cfa_onehot = Registry().getInstance().getMapping("bayer2cfa", prev_stages).getParam("cfa_onehot")
        out = f"{stage}.applier"              # [n,3,h,w]

        nodes, inits, vis = [], [], []

        # 1) Slice CFA planes (R,G0,G1,B) — Slice: data, starts, ends, axes
        R_half, G0_half, G1_half, B_half = [f"{stage}.{n}" for n in ("R_half","G0_half","G1_half","B_half")]

        (sR, eR, aR), inits_R = _slice_inputs_1d(stage, "R",  starts=[0], ends=[1], axes=[1])
        (sG0,eG0,aG0), inits_G0 = _slice_inputs_1d(stage, "G0", starts=[1], ends=[2], axes=[1])
        (sG1,eG1,aG1), inits_G1 = _slice_inputs_1d(stage, "G1", starts=[2], ends=[3], axes=[1])
        (sB, eB, aB), inits_B = _slice_inputs_1d(stage, "B",  starts=[3], ends=[4], axes=[1])
        inits += inits_R + inits_G0 + inits_G1 + inits_B

        nodes += [
            oh.make_node("Slice", [cfa4, sR, eR, aR], [R_half], name=f"{stage}_slice_R"),
            oh.make_node("Slice", [cfa4, sG0,eG0,aG0], [G0_half], name=f"{stage}_slice_G0"),
            oh.make_node("Slice", [cfa4, sG1,eG1,aG1], [G1_half], name=f"{stage}_slice_G1"),
            oh.make_node("Slice", [cfa4, sB, eB, aB], [B_half], name=f"{stage}_slice_B"),
        ]

        # 2) Upsample to full resolution
        R_full, G0_full, G1_full, B_full = [f"{stage}.{n}" for n in ("R_full","G0_full","G1_full","B_full")]
        G_full_native = f"{stage}.G_full_native"
        nodes += [
            oh.make_node("Resize", [R_half], [R_full], name=f"{stage}_resize_R"),
            oh.make_node("Resize", [G0_half], [G0_full], name=f"{stage}_resize_G0"),
            oh.make_node("Resize", [G1_half], [G1_full], name=f"{stage}_resize_G1"),
            oh.make_node("Resize", [B_half], [B_full], name=f"{stage}_resize_B"),
            oh.make_node("Add", [G0_full, G1_full], [G_full_native], name=f"{stage}_sum_G_native"),
        ]

        # 3) Tile one-hot CFA to full image → masks
        # ONNX Tile requires a repeats tensor input; provide concrete ints if known.
        # If you use symbolic dims, replace Tile with Resize/Expand in your framework.
        M_stack = f"{stage}.mask_stack"  # [h,w,4] after tiling
        M_R, M_G0, M_G1, M_B, M_G = [f"{stage}.{n}" for n in ("M_R","M_G0","M_G1","M_B","M_G")]

        # Example repeats: from 2x2 to h x w → [h/2, w/2, 1]. Put concrete ints here if available.
        tile_repeats_name, tile_repeats_init = _tile_repeats_init(stage, repeats=[1,1,1])  # placeholder; set real repeats
        inits.append(tile_repeats_init)

        (sMR,eMR,aMR), inits_mR = _slice_inputs_1d(stage, "mask_R",  starts=[0], ends=[1], axes=[2])
        (sMG0,eMG0,aMG0), inits_mG0 = _slice_inputs_1d(stage, "mask_G0", starts=[1], ends=[2], axes=[2])
        (sMG1,eMG1,aMG1), inits_mG1 = _slice_inputs_1d(stage, "mask_G1", starts=[2], ends=[3], axes=[2])
        (sMB,eMB,aMB), inits_mB = _slice_inputs_1d(stage, "mask_B",  starts=[3], ends=[4], axes=[2])
        inits += inits_mR + inits_mG0 + inits_mG1 + inits_mB

        nodes += [
            oh.make_node("Tile", [cfa_onehot, tile_repeats_name], [M_stack], name=f"{stage}_tile_cfa_onehot"),
            oh.make_node("Slice", [M_stack, sMR,eMR,aMR], [M_R],  name=f"{stage}_mask_R"),
            oh.make_node("Slice", [M_stack, sMG0,eMG0,aMG0], [M_G0], name=f"{stage}_mask_G0"),
            oh.make_node("Slice", [M_stack, sMG1,eMG1,aMG1], [M_G1], name=f"{stage}_mask_G1"),
            oh.make_node("Slice", [M_stack, sMB,eMB,aMB], [M_B],  name=f"{stage}_mask_B"),
            oh.make_node("Add", [M_G0, M_G1], [M_G], name=f"{stage}_mask_G"),
        ]

        # 4) Kernel initializers
        inits += mhc_kernel_inits(stage)

        # 5) Green interpolation at R/B sites
        G_at_R, G_at_B, G_interp = [f"{stage}.{n}" for n in ("G_at_R","G_at_B","G_interp")]
        nodes += [
            oh.make_node("Conv", [R_full, f"{stage}.K_mhc_g_r"], [G_at_R], name=f"{stage}_conv_G_at_R"),
            oh.make_node("Mul",  [G_at_R, M_R], [G_at_R], name=f"{stage}_mask_G_at_R"),
            oh.make_node("Conv", [B_full, f"{stage}.K_mhc_g_b"], [G_at_B], name=f"{stage}_conv_G_at_B"),
            oh.make_node("Mul",  [G_at_B, M_B], [G_at_B], name=f"{stage}_mask_G_at_B"),
            oh.make_node("Add",  [G_at_R, G_at_B], [G_interp], name=f"{stage}_sum_G_interp"),
        ]

        # 6) Red interpolation (B sites + G sites H/V)
        R_at_B, R_at_G_h, R_at_G_v, R_interp = [f"{stage}.{n}" for n in ("R_at_B","R_at_G_h","R_at_G_v","R_interp")]
        nodes += [
            oh.make_node("Conv", [B_full, f"{stage}.K_mhc_r_b"], [R_at_B], name=f"{stage}_conv_R_at_B"),
            oh.make_node("Mul",  [R_at_B, M_B], [R_at_B], name=f"{stage}_mask_R_at_B"),
            oh.make_node("Conv", [G_full_native, f"{stage}.K_mhc_r_g_h"], [R_at_G_h], name=f"{stage}_conv_R_at_G_h"),
            oh.make_node("Mul",  [R_at_G_h, M_G], [R_at_G_h], name=f"{stage}_mask_R_at_G_h"),
            oh.make_node("Conv", [G_full_native, f"{stage}.K_mhc_r_g_v"], [R_at_G_v], name=f"{stage}_conv_R_at_G_v"),
            oh.make_node("Mul",  [R_at_G_v, M_G], [R_at_G_v], name=f"{stage}_mask_R_at_G_v"),
            oh.make_node("Add",  [R_at_G_h, R_at_G_v], [R_interp], name=f"{stage}_sum_R_interp"),
            oh.make_node("Add",  [R_interp, R_at_B], [R_interp], name=f"{stage}_sum_R_total"),
        ]

        # 7) Blue interpolation (R sites + G sites H/V)
        B_at_R, B_at_G_h, B_at_G_v, B_interp = [f"{stage}.{n}" for n in ("B_at_R","B_at_G_h","B_at_G_v","B_interp")]
        nodes += [
            oh.make_node("Conv", [R_full, f"{stage}.K_mhc_b_r"], [B_at_R], name=f"{stage}_conv_B_at_R"),
            oh.make_node("Mul",  [B_at_R, M_R], [B_at_R], name=f"{stage}_mask_B_at_R"),
            oh.make_node("Conv", [G_full_native, f"{stage}.K_mhc_b_g_h"], [B_at_G_h], name=f"{stage}_conv_B_at_G_h"),
            oh.make_node("Mul",  [B_at_G_h, M_G], [B_at_G_h], name=f"{stage}_mask_B_at_G_h"),
            oh.make_node("Conv", [G_full_native, f"{stage}.K_mhc_b_g_v"], [B_at_G_v], name=f"{stage}_conv_B_at_G_v"),
            oh.make_node("Mul",  [B_at_G_v, M_G], [B_at_G_v], name=f"{stage}_mask_B_at_G_v"),
            oh.make_node("Add",  [B_at_G_h, B_at_G_v], [B_interp], name=f"{stage}_sum_B_interp"),
            oh.make_node("Add",  [B_interp, B_at_R], [B_interp], name=f"{stage}_sum_B_total"),
        ]

        # 8) Blend native + interpolated → RGB
        R_native, G_native, B_native = [f"{stage}.{n}" for n in ("R_native","G_native","B_native")]
        R_out, G_out, B_out = [f"{stage}.{n}" for n in ("R_out","G_out","B_out")]
        nodes += [
            oh.make_node("Mul", [R_full, M_R], [R_native], name=f"{stage}_native_R"),
            oh.make_node("Add", [R_native, R_interp], [R_out], name=f"{stage}_out_R"),

            oh.make_node("Mul", [G_full_native, M_G], [G_native], name=f"{stage}_native_G"),
            oh.make_node("Add", [G_native, G_interp], [G_out], name=f"{stage}_out_G"),

            oh.make_node("Mul", [B_full, M_B], [B_native], name=f"{stage}_native_B"),
            oh.make_node("Add", [B_native, B_interp], [B_out], name=f"{stage}_out_B"),

            oh.make_node("Concat", [R_out, G_out, B_out], [out], axis=1, name=f"{stage}_concat_RGB"),
        ]

        # Value infos
        vis += [
            oh.make_tensor_value_info(cfa4, TensorProto.FLOAT, ["n",4,"h2","w2"]),
            oh.make_tensor_value_info(out,  TensorProto.FLOAT, ["n",3,"h","w"]),
        ]

        outputs = {"applier": {"name": out}}
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(cfa4)
        result.appendInput(cfa_onehot) # one‑hot tile
        return result
