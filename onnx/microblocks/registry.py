# microblocks/registry.py
import importlib
import inspect
import pkgutil
import pathlib

class Registry:
    def __init__(self):
        # static registry of microblock classes
        self._registry = {}
        # dynamic outputs tracking
        self._outputs_map = {}

    def register(self, name, version, cls):
        self._registry[(name, version)] = cls

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

    # 🔹 Outputs management
    def clear_all_outputs(self):
        """Reset outputs before a new build run."""
        self._outputs_map.clear()

    def register_outputs(self, stage_name: str, class_name: str, outputs: dict):
        """Store outputs for a stage."""
        self._outputs_map[stage_name] = {"class_name": class_name, "outputs": outputs}

    def get_output(self, stage_name: str, coeff_name: str) -> str:
        """Retrieve a tensor alias by stage + coeff name."""
        if stage_name not in self._outputs_map:
            raise KeyError(f"Stage '{stage_name}' not registered")
        outputs = self._outputs_map[stage_name]["outputs"]
        if coeff_name not in outputs:
            raise KeyError(f"Coeff '{coeff_name}' not found in stage '{stage_name}'")
        return outputs[coeff_name]
