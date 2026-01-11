from typing import List, Dict, Any, Sequence, Optional

class MicroBlock:
    """
    Canonical SoftISP microblock base.

    Responsibilities:
    - Identity: stage, block, version
    - Contracts: explicit input/output names
    - Build dispatcher: algo/applier/coordinator (mode-aware)
    - Coeffs/state/context handling
    - Optional hooks: validate(), prepare()
    - Utilities: canonical name helpers and ONNX-friendly semantics

    Subclasses must implement:
    - input_names()
    - output_names()
    - build_algo_node()
    - build_applier_node()
    - build_coordinator_node()
    """

    # -------- Identity & setup --------
    def __init__(self, stage: int, block: str, version: str, **kwargs: Any):
        # Canonical identity
        self.stage: int = stage
        self.block: str = block
        self.version: str = version

        # Context payloads
        self.coeffs: Dict[str, Any] = kwargs.get("coeffs", {})
        self.state: Dict[str, Any] = kwargs.get("state", {})
        self.context: Dict[str, Any] = kwargs.get("context", {})
        self.runtime: Dict[str, Any] = kwargs.get("runtime", {})

        # IO hints (optional; subclasses should override input/output methods)
        self._inputs: List[str] = kwargs.get("inputs", [])
        self._outputs: List[str] = kwargs.get("outputs", [])

        # Mode default if caller doesn’t pass one
        self.mode: str = kwargs.get("mode", "algo")

        # Optional namespacing/canonicalization preferences
        self.ns: str = kwargs.get("ns", self.block)  # namespace for names
        self.sep: str = kwargs.get("sep", "_")        # separator for names

    # -------- Public entry points --------
    def build(self, mode: Optional[str] = None) -> Sequence[Any]:
        """
        Mode-aware dispatcher returning ONNX nodes (or node lists).
        """
        mode = (mode or self.mode or "algo").lower()
        if mode == "algo":
            return self.build_algo_node()
        elif mode == "applier":
            return self.build_applier_node()
        elif mode == "coordinator":
            return self.build_coordinator_node()
        raise ValueError(f"Unknown build mode: {mode}")

    def input_names(self) -> List[str]:
        """
        Subclasses must return canonical input tensor names used in graph wiring.
        If not overridden, fall back to optional _inputs provided via kwargs.
        """
        if self._inputs:
            return list(self._inputs)
        raise NotImplementedError(f"{self._id()} missing input_names()")

    def output_names(self) -> List[str]:
        """
        Subclasses must return canonical output tensor names used in graph wiring.
        If not overridden, fall back to optional _outputs provided via kwargs.
        """
        if self._outputs:
            return list(self._outputs)
        raise NotImplementedError(f"{self._id()} missing output_names()")

    # -------- Abstract build stubs --------
    def build_algo_node(self) -> Sequence[Any]:
        raise NotImplementedError(f"{self._id()} missing build_algo_node()")

    def build_applier_node(self) -> Sequence[Any]:
        raise NotImplementedError(f"{self._id()} missing build_applier_node()")

    def build_coordinator_node(self) -> Sequence[Any]:
        raise NotImplementedError(f"{self._id()} missing build_coordinator_node()")

    # -------- Optional lifecycle hooks --------
    def validate(self) -> bool:
        """
        Override to validate coeffs, IO names, shapes, and state.
        Return True on success; raise on failure for explicit break.
        """
        # Minimal default checks: non-empty IO names by subclasses
        # (Harness typically calls input_names()/output_names() before build)
        return True

    def prepare(self) -> None:
        """
        Override for pre-build normalization (e.g., coeff packing, name binding).
        """
        return None

    # -------- Utilities (name canonicalization) --------
    def canon(self, *parts: str) -> str:
        """
        Canonical name builder with namespace and separator.
        Example: canon('gain', 'r') -> 'wbblock_gain_r' (if ns='wbblock').
        """
        parts = [p for p in parts if p]
        base = self.sep.join(parts)
        if self.ns:
            return self.ns + self.sep + base if base else self.ns
        return base

    def io(self, role: str, name: str) -> str:
        """
        IO-specific canonicalization helper.
        Example: io('in', 'input') -> 'wbblock_in_input'
        """
        return self.canon(role, name)

    def node_name(self, local: str) -> str:
        """
        Canonical node name helper.
        Example: node_name('ApplyWB') -> 'wbblock_ApplyWB'
        """
        return self.canon(local)

    # -------- Coeff helpers --------
    def get_coeff(self, key: str, default: Any = None) -> Any:
        return self.coeffs.get(key, default)

    def require_coeff(self, key: str) -> Any:
        if key not in self.coeffs:
            raise KeyError(f"{self._id()} missing coeff: {key}")
        return self.coeffs[key]

    # -------- ID & repr --------
    def _id(self) -> str:
        return f"{self.__class__.__name__}[{self.block}:{self.version}@{self.stage}]"

    def __repr__(self) -> str:
        return f"<{self._id()} inputs={self._inputs or '?'} outputs={self._outputs or '?'}>"
