# base.py
from typing import Dict, List, Any, Tuple
from onnx import helper, TensorProto

class MicroblockBase:
    """
    Canonical microblock interface for SoftISP.
    Each block must expose:
      - name, version, coeff_names (input coeffs), output_coeff_names (produced coeffs)
      - input_names(), output_names()
      - build_algo(), build_applier(), build_coordinator()
    """

    # --- Metadata ---
    name: str = "unnamed"
    version: str = "v0"
    coeff_names: List[str] = []        # required input coeffs
    output_coeff_names: List[str] = [] # coeffs produced by this block

    process_method: str = "Identity"
    depends_on: List[str] = []

    # --- IO helpers ---
    def input_names(self) -> List[str]:
        return ["input"]

    def output_names(self) -> List[str]:
        return ["output"]

    # --- Build methods ---
    def build_algo(self, prev_out: str):
        raise NotImplementedError

    def build_applier(self, prev_out: str):
        raise NotImplementedError

    def build_coordinator(self, prev_out: str):
        return None

    # --- Contract validation ---
    def validate_contract(self, outputs: Dict[str, Any], value_info: List[Any]) -> None:
        if "image" not in outputs or "name" not in outputs["image"]:
            raise ValueError(f"{self.name}: missing image output declaration")

        names = [outputs["image"]["name"]]
        if "coeffs" in outputs:
            for item in outputs["coeffs"]:
                names.append(item["name"])
        if "params" in outputs:
            for item in outputs["params"]:
                names.append(item["name"])

        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate names detected")

        vinfo_names = set([vi.name if hasattr(vi, "name") else vi["name"] for vi in value_info])
        for n in names:
            if n not in vinfo_names:
                raise ValueError(f"{self.name}: value_info missing for '{n}'")


