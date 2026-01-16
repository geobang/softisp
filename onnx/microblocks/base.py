# microblocks/base.py
import onnx
import onnx.helper as oh
from microblocks.registry import Registry
import inspect

class BuildResult:
    def __init__(self, outputs, nodes, inits, vis, inputs=None, owner_cls=None):
        # If no owner class is passed, capture from caller
        if owner_cls is None:
            frame = inspect.currentframe().f_back
            caller_self = frame.f_locals.get("self")
            if caller_self is not None:
                owner_cls = caller_self.__class__

        self._owner_cls = owner_cls

        # Internal reference copies
        self._ref_outputs = dict(outputs)
        self._ref_nodes   = list(nodes)
        self._ref_inits   = list(inits)
        self._ref_vis     = list(vis)
        self._ref_inputs  = dict(inputs) if inputs else {}

        # Working view
        self.func  = None
        self.call  = None
        self._regenerate()

    def _regenerate(self):
        """Reset working view from internal references and regenerate function + call node."""
        self.outputs = dict(self._ref_outputs)
        self.nodes   = list(self._ref_nodes)
        self.inits   = list(self._ref_inits)
        self.vis     = list(self._ref_vis)
        self.inputs  = dict(self._ref_inputs)

        func_name = f"{self._owner_cls.__name__}_{getattr(self._owner_cls, 'version', 'v0')}"
        self.func, self.call = self._to_function_and_call(func_name)
        # Replace with call node in the outer graph
        self.nodes = [self.call]

    def appendInput(self, name: str, shape=None, desc: str = None):
        self._ref_inputs[name] = {"name": name, "shape": shape, "desc": desc}
        self._regenerate()
        return self

    def appendOutput(self, name: str, shape=None, desc: str = None):
        self._ref_outputs[name] = {"name": name, "shape": shape, "desc": desc}
        self._regenerate()
        return self

    def _to_function_and_call(self, func_name: str):
        input_names  = [inp["name"] for inp in self._ref_inputs.values()]
        output_names = [out["name"] for out in self._ref_outputs.values()]

        # 1) Build Constant nodes for initializers FIRST
        const_nodes = []
        for tensor in self._ref_inits:
            const_nodes.append(
                oh.make_node("Constant", inputs=[], outputs=[tensor.name], value=tensor)
            )

        # 2) Then append the actual function body nodes
        func_nodes = const_nodes + list(self._ref_nodes)

        # 3) Assemble FunctionProto with proper opset import
        func = onnx.FunctionProto()
        func.name = func_name
        func.domain = "softisp"
        func.input.extend(input_names)
        func.output.extend(output_names)
        func.node.extend(func_nodes)
        # Declare that this function uses the standard ONNX opset
        func.opset_import.extend([oh.make_operatorsetid("", 13)])

        # 4) Create the call node in the same domain
        call_node = oh.make_node(func_name, inputs=input_names, outputs=output_names, domain="softisp")
        return func, call_node

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
