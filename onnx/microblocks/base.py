# microblocks/base.py
import onnx
import onnx.helper as oh
from dataclasses import dataclass, field
from microblocks.registry import Registry

@dataclass
class BuildResult:
    outputs: dict
    nodes: list
    inits: list
    vis: list
    inputs: dict = field(default_factory=dict)

    def __init__(self, outputs, nodes, inits, vis, inputs=None):
        self.outputs = outputs
        self.nodes = nodes
        self.inits = inits
        self.vis = vis
        self.inputs = inputs if inputs is not None else {}

    # 🔹 Helper methods for symmetry
    def appendInput(self, name: str, desc: str = None):
        self.inputs[name] = {"name": name, "desc": desc}
        return self

    def appendOutput(self, name: str, desc: str = None):
        self.outputs[name] = {"name": name, "desc": desc}
        return self


class MicroblockBase:
    """
    Abstract base class for all microblocks.
    Each block can emit a family/version/class initializer into the ONNX graph.
    """
    name = "unnamed"
    version = "v0"
    deps = []
    needs = []

    # Shared registry instance
    registry: Registry = Registry.getInstance()

    def _make_family_version_init(self):
        """
        Create a scalar initializer named as 'ClassName.family.version'.
        Example: GammaBase.gamma_base.v2
        """
        init_name = f"{self.__class__.__name__}.{self.name}.{self.version}"
        return oh.make_tensor(
            name=init_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=[1],
            vals=[1.0],
        )

    def _attach_marker(self, result: BuildResult):
        """
        Attach an isolated Identity node consuming the family/version initializer
        so it is visible in Netron but does not interfere with the main graph.
        """
        marker_name = f"{self.__class__.__name__}.{self.name}.{self.version}"
        tensor = self._make_family_version_init()
        result.inits.append(tensor)

        # Identity node to make it visible
        identity_node = oh.make_node(
            "Identity",
            inputs=[marker_name],
            outputs=[marker_name + "_marker"]
        )
        result.nodes.append(identity_node)

        # ValueInfo for the Identity output
        result.vis.append(
            oh.make_tensor_value_info(marker_name + "_marker", onnx.TensorProto.FLOAT, [1])
        )

    def build_applier(self, stage: str, prev_stages=None):
        raise NotImplementedError

    def build_algo(self, stage: str, prev_stages=None):
        raise NotImplementedError

    def build_coordinator(self, stage: str, prev_stages=None):
        raise NotImplementedError

    def get_build_applier(self, stage: str, prev_stages=None) -> BuildResult:
        result: BuildResult = self.build_applier(stage, prev_stages)
        flat = {k: v["name"] for k, v in result.outputs.items()}
        MicroblockBase.registry.register_outputs(stage, self.__class__.__name__, flat)
        self._attach_marker(result)
        return result

    def get_build_algo(self, stage: str, prev_stages=None) -> BuildResult:
        result: BuildResult = self.build_algo(stage, prev_stages)
        flat = {k: v["name"] for k, v in result.outputs.items()}
        MicroblockBase.registry.register_outputs(stage, self.__class__.__name__, flat)
        self._attach_marker(result)
        return result

    def get_build_coordinator(self, stage: str, prev_stages=None) -> BuildResult:
        result: BuildResult = self.build_coordinator(stage, prev_stages)
        flat = {k: v["name"] for k, v in result.outputs.items()}
        MicroblockBase.registry.register_outputs(stage, self.__class__.__name__, flat)
        self._attach_marker(result)
        return result
