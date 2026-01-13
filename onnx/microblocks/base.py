# microblocks/base.py
import onnx.helper as oh

class MicroblockBase:
    """
    Abstract base class for all microblocks.
    Provides a uniform interface and default attributes.
    Concrete subclasses must override `name`, `version`, and implement build_applier().
    """

    # Default attributes (will be overridden in subclasses)
    name = "unnamed"
    version = "v0"
    deps = []   # list of allowed upstream class names
    needs = []  # list of required tensors (stage-scoped)

    def __init__(self):
        # You can add common initialization here if needed
        pass

    def build_applier(self, stage: str, prev_stages=None):
        """
        Build the ONNX nodes for this microblock.
        Must be implemented by subclasses.
        Should return: (outputs, nodes, inits, vis)
        - outputs: dict of logical outputs → {"name": tensor_name}
        - nodes: list of onnx.NodeProto
        - inits: list of onnx.TensorProto
        - vis: list of onnx.ValueInfoProto
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement build_applier()")

    @classmethod
    def contract(cls):
        """
        Return a summary of the block’s declared contract:
        - name, version, deps, needs
        Useful for coordinator validation.
        """
        return {
            "name": cls.name,
            "version": cls.version,
            "deps": cls.deps,
            "needs": cls.needs,
        }
