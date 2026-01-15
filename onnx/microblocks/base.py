# microblocks/base.py
from microblocks.registry import Registry

class MicroblockBase:
    """
    Abstract base class for all microblocks.
    """
    name = "unnamed"
    version = "v0"
    deps = []
    needs = []

    # Shared registry instance
    registry: Registry = Registry.getInstance()

    def build_applier(self, stage: str, prev_stages=None):
        raise NotImplementedError

    def build_algo(self, stage: str, prev_stages=None):
        raise NotImplementedError

    def build_coordinator(self, stage: str, prev_stages=None):
        raise NotImplementedError

    def get_build_applier(self, stage: str, prev_stages=None):
        outputs, nodes, inits, vis = self.build_applier(stage, prev_stages)
        flat = {k: v["name"] for k, v in outputs.items()}
        MicroblockBase.registry.register_outputs(stage, self.__class__.__name__, flat)
        return outputs, nodes, inits, vis

    def get_build_algo(self, stage: str, prev_stages=None):
        outputs, nodes, inits, vis = self.build_algo(stage, prev_stages)
        flat = {k: v["name"] for k, v in outputs.items()}
        MicroblockBase.registry.register_outputs(stage, self.__class__.__name__, flat)
        return outputs, nodes, inits, vis

    def get_build_coordinator(self, stage: str, prev_stages=None):
        outputs, nodes, inits, vis = self.build_coordinator(stage, prev_stages)
        flat = {k: v["name"] for k, v in outputs.items()}
        MicroblockBase.registry.register_outputs(stage, self.__class__.__name__, flat)
        return outputs, nodes, inits, vis
