# microblocks/registry.py
import importlib, inspect, pkgutil, pathlib
from onnx import TensorProto

class MappingHandle:
    """
    Wraps a stage mapping so you can call .getParam("coeff").
    """
    def __init__(self, stage_name: str, class_name: str, outputs: dict):
        self.stage_name = stage_name
        self.class_name = class_name
        self.outputs = outputs

    def getParam(self, coeff_name: str) -> str:
        if coeff_name not in self.outputs:
            raise KeyError(f"Coeff '{coeff_name}' not found in stage '{self.stage_name}'")
        return self.outputs[coeff_name]


class Registry:
    """
    Singleton registry for microblocks and dynamic outputs.
    Provides fluent API: Registry.getInstance().getMapping(...).getParam(...)
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
            cls._instance._outputs_map = {}
            cls._instance._dynamic_map = {}
        return cls._instance

    @classmethod
    def getInstance(cls):
        """Return the singleton instance."""
        return cls()

    def import_all_microblocks(self, base="microblocks"):
        """
        Recursively import all modules under ./microblocks and auto‑register
        any class that declares `name` and `version`.
        """
        base_path = pathlib.Path(__file__).parent
        for finder, modname, ispkg in pkgutil.walk_packages([str(base_path)], prefix=f"{base}."):
            try:
                module = importlib.import_module(modname)
            except Exception as e:
                print(f"[WARN] Failed to import {modname}: {e}")
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, "name") and hasattr(obj, "version"):
                    if obj.__name__ == "MicroblockBase":
                        continue
                    key = (obj.name, obj.version)
                    self._registry[key] = obj
                    print(f"[DEBUG] Registered {key} → {obj.__name__}")

    def dump_registry(self):
        return self._registry

    def clear_all_outputs(self):
        self._outputs_map.clear()

    def register_outputs(self, stage_name: str, class_name: str, outputs: dict):
        self._outputs_map[stage_name] = {"class_name": class_name, "outputs": outputs}

    def set_dynamic_map(self, dynamic_map: dict):
        self._dynamic_map = dynamic_map

    def getMapping(self, family_name: str, prev_stages: list) -> MappingHandle:
        """
        Resolve a stage by family name + version from prev_stages.
        Returns a MappingHandle with .getParam().
        """
        for stage_name in prev_stages:
            if stage_name not in self._dynamic_map:
                continue
            spec = self._dynamic_map[stage_name]
            fam = spec["family"]
            version = spec["version"]
            if fam == family_name:
                class_name = self._outputs_map[stage_name]["class_name"]
                outputs = self._outputs_map[stage_name]["outputs"]
                return MappingHandle(stage_name, class_name, outputs)
        raise KeyError(f"Family '{family_name}' not found in prev_stages")

    # NEW: helper to check if a tensor name has already been produced
    def has_output(self, name: str) -> bool:
        for stage_spec in self._outputs_map.values():
            outputs = stage_spec["outputs"]
            if name in outputs.values():
                return True
        return False
