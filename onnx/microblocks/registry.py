# microblocks/registry.py
import importlib
import inspect
import pkgutil
import pathlib

REGISTRY = {}

def import_all_microblocks(base="microblocks"):
    """
    Recursively import all modules under ./microblocks and auto‑register
    any class that declares `name` and `version`.
    """
    base_path = pathlib.Path(__file__).parent

    # Walk all packages and subpackages
    for finder, modname, ispkg in pkgutil.walk_packages([str(base_path)], prefix=f"{base}."):
        try:
            module = importlib.import_module(modname)
        except Exception as e:
            print(f"[WARN] Failed to import {modname}: {e}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, "name") and hasattr(obj, "version"):
                # Skip abstract base
                if obj.__name__ == "MicroblockBase":
                    continue
                key = (obj.name, obj.version)
                REGISTRY[key] = obj
                print(f"[DEBUG] Registered {key} → {obj.__name__}")

def dump_registry():
    """Return the current registry dict for inspection/debugging."""
    return REGISTRY
