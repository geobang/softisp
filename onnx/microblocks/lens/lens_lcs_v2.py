# onnx/microblocks/lens/lens_lcs_v2.py

import onnx.helper as oh
from onnx import TensorProto
from .lens_lcs_base import LensLCSBase


class LensLCSV2(LensLCSBase):
    """
    LensLCSV2 (v2)
    --------------
    Inherits LensLCSBase and extends it with analytic coefficient generation.

    Needs:
        - applier [n,3,h,w] : image tensor from upstream
        - lcs_params []     : [cx, cy, strength, falloff]

    Provides:
        - applier [n,3,h,w] : corrected image tensor
        - lcs_coeffs [h,w]  : generated coefficient map

    Behavior:
        - build_algo: generates lcs_coeffs from radial parameters
        - build_applier: reuses LensLCSBase multiplication logic
    """

    name = "lens_lcs_v2"
    version = "v2"
    needs = ["applier", "lcs_params"]
    provides = ["applier", "lcs_coeffs"]

    def build_algo(self, stage: str, prev_stages=None):
        """
        Generate lcs_coeffs from radial shading parameters.
        coeff(r) = 1 + strength * (1 - (r / rmax)^falloff)
        """
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        lcs_params = f"{upstream}.lcs_params"

        cx, cy, strength, falloff = [f"{stage}.{p}" for p in ("cx","cy","strength","falloff")]
        nodes.append(
            oh.make_node("Split", inputs=[lcs_params],
                         outputs=[cx, cy, strength, falloff],
                         name=f"{stage}.split_params", axis=0)
        )

        # Radial distance r = sqrt((x-cx)^2 + (y-cy)^2)
        dx, dy, r = [f"{stage}.{p}" for p in ("dx","dy","r")]
        nodes.append(oh.make_node("Sub", inputs=["Xgrid", cx], outputs=[dx], name=f"{stage}.sub_x"))
        nodes.append(oh.make_node("Sub", inputs=["Ygrid", cy], outputs=[dy], name=f"{stage}.sub_y"))
        nodes.append(oh.make_node("Pow", inputs=[dx, "2.0"], outputs=[f"{stage}.dx2"], name=f"{stage}.pow_dx"))
        nodes.append(oh.make_node("Pow", inputs=[dy, "2.0"], outputs=[f"{stage}.dy2"], name=f"{stage}.pow_dy"))
        nodes.append(oh.make_node("Add", inputs=[f"{stage}.dx2", f"{stage}.dy2"], outputs=[f"{stage}.rsq"], name=f"{stage}.add_rsq"))
        nodes.append(oh.make_node("Sqrt", inputs=[f"{stage}.rsq"], outputs=[r], name=f"{stage}.sqrt_r"))

        # Normalize radius
        rmax = f"{stage}.rmax"
        inits.append(oh.make_tensor(rmax, TensorProto.FLOAT, [], [1.0]))  # placeholder, coordinator sets actual max
        rnorm = f"{stage}.rnorm"
        nodes.append(oh.make_node("Div", inputs=[r, rmax], outputs=[rnorm], name=f"{stage}.div_rnorm"))

        # coeff = 1 + strength * (1 - rnorm^falloff)
        rpow, one_minus, coeff = [f"{stage}.{p}" for p in ("rpow","one_minus","coeff")]
        nodes.append(oh.make_node("Pow", inputs=[rnorm, falloff], outputs=[rpow], name=f"{stage}.pow_r"))
        nodes.append(oh.make_node("Sub", inputs=["1.0", rpow], outputs=[one_minus], name=f"{stage}.sub_one"))
        nodes.append(oh.make_node("Mul", inputs=[strength, one_minus], outputs=[f"{stage}.mul_strength"], name=f"{stage}.mul_strength"))
        nodes.append(oh.make_node("Add", inputs=["1.0", f"{stage}.mul_strength"], outputs=[coeff], name=f"{stage}.add_coeff"))

        vis.append(oh.make_tensor_value_info(coeff, TensorProto.FLOAT, ["h","w"]))
        outputs = {"lcs_coeffs": {"name": coeff}}
        return outputs, nodes, inits, vis

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply generated lcs_coeffs to applier (image).
        """
        return super().build_applier(stage, prev_stages)
