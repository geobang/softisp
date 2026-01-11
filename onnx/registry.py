"""
SoftISP ONNX block registry (rewrite branch)

- Global BLOCK_REGISTRY keyed by (name, version) -> class
- @register_block decorator for MicroBlock subclasses
- import_all_microblocks() dynamically loads microblocks/*
- get_block() lookup with clear KeyError on miss
- dump_registry(), list_blocks(), clear_registry() for diagnostics
"""

import importlib
import pkgutil
import os
from typing import Dict, Tuple, Type

# Global registry of blocks
BLOCK_REGISTRY: Dict[Tuple[str, str], Type] = {}


def register_block(cls):
    """
    Decorator to register a MicroBlock subclass.
    The class must define attributes: name, version.
    """
    if not hasattr(cls, "name") or not hasattr(cls, "version"):
        raise AttributeError(f"Class {cls.__name__} missing name/version attributes")

    key = (cls.name, cls.version)
    if key in BLOCK_REGISTRY:
        raise ValueError(f"Duplicate block registration: {key}")

    BLOCK_REGISTRY[key] = cls
    return cls


def import_all_microblocks():
    """
    Dynamically import all microblock modules under microblocks/*.
    Assumes you are running inside the onnx/ folder (rewrite branch layout).
    """
    package = "microblocks"
    package_path = os.path.join(os.path.dirname(__file__), package)
    for finder, name, ispkg in pkgutil.walk_packages([package_path], prefix=f"{package}."):
        try:
            importlib.import_module(name)
        except Exception as e:
            print(f"⚠️ Failed to import {name}: {e}")


def get_block(name, version):
    """
    Retrieve a registered block class by name and version.
    Raises KeyError if not found.
    """
    key = (name, version)
    if key not in BLOCK_REGISTRY:
        raise KeyError(f"Block not found: {key}")
    return BLOCK_REGISTRY[key]


def dump_registry():
    """
    Return a shallow copy of the registry for debugging.
    Shape: dict[(name, version)] = class
    """
    return dict(BLOCK_REGISTRY)


def list_blocks():
    """
    Return a sorted list of (name, version) tuples for all registered blocks.
    """
    return sorted(BLOCK_REGISTRY.keys())


def clear_registry():
    """
    Clear the registry (useful for tests).
    """
    BLOCK_REGISTRY.clear()
