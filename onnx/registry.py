# onnx/registry.py
import importlib
import inspect
import pkgutil
import pathlib

REGISTRY = {}

def import_all_microblocks(base="microblocks"):
    """
    Import all microblock modules under ./microblocks recursively
    and auto‑register any class that declares `name` and `version`.
    """
    base_path = pathlib.Path(__file__).parent / base
    for finder, modname, ispkg in pkgutil.walk_packages([str(base_path)], prefix=f"{base}."):
        if ispkg:
            continue
        try:
            module = importlib.import_module(modname)
        except Exception as e:
            print(f"[WARN] Failed to import {modname}: {e}")
            continue

        # Auto‑register any class with name+version attributes
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, "name") and hasattr(obj, "version"):
                key = (obj.name, obj.version)
                REGISTRY[key] = obj
                print(f"[DEBUG] Registered {key} → {obj.__name__}")

def dump_registry():
    return REGISTRY
