# onnx/build_all.py
import sys
import os
import json
from registry import import_all_microblocks, dump_registry

DEFAULT_MANIFEST = "pipeline.json"
DEFAULT_MODE = "applier"
OUTPUT_ROOT = "onnx_out"

def instantiate_block(block_name, version):
    """Instantiate a block from the registry by name and version."""
    reg = dump_registry()
    key = (block_name, version)
    if key not in reg:
        raise KeyError(f"Block not found in registry: {key}")
    return reg[key]()

def normalize_outputs(outputs):
    """Force all coeff keys and tensor names to lowercase for consistency."""
    normalized = {}
    for key, val in outputs.items():
        lower_key = key.lower()
        if isinstance(val, dict) and "name" in val:
            normalized[lower_key] = {"name": val["name"].lower()}
        elif isinstance(val, str):
            normalized[lower_key] = {"name": val.lower()}
        elif isinstance(val, list):
            # normalize each element in the list
            normalized[lower_key] = [v.lower() if isinstance(v, str) else v for v in val]
        else:
            # fallback: keep as-is
            normalized[lower_key] = val
    return normalized

def ensure_output_folder(manifest, mode):
    """Create output folder based on canonical_name or mode."""
    folder_name = manifest.get("canonical_name", mode)
    out_dir = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[DEBUG] Output folder created: {out_dir}")
    return out_dir

def build_graph(manifest, mode):
    """Build a graph for the given manifest and mode (algo/applier/coordinator)."""
    prev_out = "input_image"
    all_nodes, all_inits, all_vis = [], [], []

    for idx, stage in enumerate(manifest["stages"]):
        block_name = stage["block"]
        version = stage["version"]

        # Instantiate block
        block = instantiate_block(block_name, version)

        # Log which block is being processed
        print(f"[DEBUG] Stage {idx} → block={block_name}, version={version}, mode={mode}")
        print(f"[DEBUG] Processing {block.__class__.__name__}…")

        # Call the appropriate builder
        if mode == "algo":
            outputs, consumed, nodes, inits, vis = block.build_algo(prev_out)
        elif mode == "applier":
            outputs, consumed, nodes, inits, vis = block.build_applier(prev_out)
        elif mode == "coordinator":
            outputs, consumed, nodes, inits, vis = block.build_coordinator(prev_out)
        else:
            raise ValueError(f"Unknown mode {mode}")

        # Normalize coeff names
        outputs = normalize_outputs(outputs)

        # Collect results
        all_nodes.extend(nodes)
        all_inits.extend(inits)
        all_vis.extend(vis)

        # Update prev_out for next stage
        if "image" in outputs:
            prev_out = outputs["image"]["name"]

        # Log completion
        print(f"[DEBUG] Finished {block.__class__.__name__} at stage {idx}")

    return all_nodes, all_inits, all_vis, prev_out

def build_all(manifest_file=None, mode=None):
    """Import all microblocks, load manifest, and build the graph."""
    # Defaults
    manifest_file = manifest_file or DEFAULT_MANIFEST
    mode = mode or DEFAULT_MODE

    print("[DEBUG] Importing microblocks to populate registry …")
    import_all_microblocks()
    reg = dump_registry()
    print(f"[DEBUG] Registry has {len(reg)} entries")
    for (name, version), cls in sorted(reg.items()):
        print(f"[DEBUG]   ({name}, {version}) -> {cls.__name__}")

    # Load manifest
    with open(manifest_file) as f:
        manifest = json.load(f)
    print(f"[DEBUG] Manifest loaded with {len(manifest['stages'])} stages")

    # Create output folder
    out_dir = ensure_output_folder(manifest, mode)

    # Build graph
    nodes, inits, vis, final_out = build_graph(manifest, mode)

    # Example: write ONNX graph to file (stub)
    out_path = os.path.join(out_dir, f"{mode}.onnx")
    print(f"[DEBUG] Would save graph to {out_path}")
    # TODO: implement save_model(nodes, inits, vis, out_path)

    return nodes, inits, vis, final_out

if __name__ == "__main__":
    manifest_file = sys.argv[1] if len(sys.argv) > 1 else None
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    build_all(manifest_file, mode)
