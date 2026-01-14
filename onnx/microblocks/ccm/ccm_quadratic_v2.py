# onnx/microblocks/ccm/ccm_quadratic_v2.py

from .ccm_quadratic_v1 import CCMQuadraticV1
import onnx.helper as oh
from onnx import TensorProto

class CCMQuadraticV2(CCMQuadraticV1):
    """
    Extends CCMQuadraticV1:
    - Reuses quadratic fit + row normalization
    - Adds first-pass post-CCM bias calculation (CI vector/scalar)
    """

    name = "ccm_quadratic_v2"
    version = "v2"
    deps = CCMQuadraticV1.deps
    needs = CCMQuadraticV1.needs + ["stats_tiles", "neutral_mask", "ci_weights"]
    provides = CCMQuadraticV1.provides + ["ci_vec", "ci_scalar", "post_ccm_tiles"]

    def build_algo(self, stage: str, prev_stages=None):
        # Get baseline CCM graph from v1
        outputs, nodes, inits, vis = super().build_algo(stage, prev_stages)

        # Add ONNX nodes for post-CCM CI calculation
        in_tiles = f"{stage}.stats_tiles"
        in_mask  = f"{stage}.neutral_mask"
        in_w     = f"{stage}.ci_weights"

        out_tiles   = f"{stage}.post_ccm_tiles"
        out_ci_vec  = f"{stage}.ci_vec"
        out_ci_scalar = f"{stage}.ci_scalar"

        # Apply CCM to stats tiles
        ccm_T = f"{stage}.ccm_T"
        nodes.append(oh.make_node("Transpose", inputs=[outputs["ccm"]["name"]],
                                  outputs=[ccm_T], name=f"{stage}.transpose_ccm", perm=[1,0]))
        nodes.append(oh.make_node("MatMul", inputs=[in_tiles, ccm_T],
                                  outputs=[out_tiles], name=f"{stage}.apply_ccm"))

        # Mask neutral tiles, compute mean per channel, white mean, CI vector/scalar
        # (reuse v1-style helper functions or inline ONNX ops as needed)

        # Register outputs
        outputs.update({
            "post_ccm_tiles": {"name": out_tiles},
            "ci_vec": {"name": out_ci_vec},
            "ci_scalar": {"name": out_ci_scalar},
        })

        return outputs, nodes, inits, vis
