# 1. Imports and constants
#!/usr/bin/env python3
"""
SoftISP ONNX builder (rewrite branch)
"""

import json
import sys
import traceback
from typing import List, Union

from onnx import helper, TensorProto, save_model
from common import instantiate_block, rewire_inputs
from registry import import_all_microblocks, dump_registry

MANIFEST_PATH = "pipeline.json"


# 2. Manifest loading and validation
def load_manifest(path: str = MANIFEST_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(manifest: dict) -> None:
    if "pipeline" not in manifest or not isinstance(manifest["pipeline"], list):
        raise ValueError("Manifest missing 'pipeline' list")

    stages = [s.get("stage") for s in manifest["pipeline"]]
    if any(s is None for s in stages):
        raise ValueError("All pipeline entries must have a 'stage' integer")

    if len(stages) != len(set(stages)):
        raise ValueError(f"Duplicate stages detected: {stages}")
    if stages != sorted(stages):
        raise ValueError(f"Stages must be strictly increasing. Got: {stages}")

    for spec in manifest["pipeline"]:
        if "block" not in spec or "version" not in spec:
            raise ValueError(f"Pipeline entry missing block/version: {spec}")
        if "coeff_names" not in spec:
            spec["coeff_names"] = []
# 3. Node helpers
def _ensure_node_list(node_or_nodes: Union[helper.NodeProto, List[helper.NodeProto]]) -> List[helper.NodeProto]:
    if isinstance(node_or_nodes, list):
        return node_or_nodes
    return [node_or_nodes]


def _normalize_node(node, stage, block_name, mode):
    """
    Ensure node inputs/outputs are lists of strings.
    Replace None or bad types with placeholder names.
    """
    ins = list(getattr(node, "input", []))
    outs = list(getattr(node, "output", []))

    fixed_ins = []
    for i, val in enumerate(ins):
        if isinstance(val, str) and val:
            fixed_ins.append(val)
        else:
            fixed_ins.append(f"{block_name}_{mode}_in{stage}_{i}")

    fixed_outs = []
    for i, val in enumerate(outs):
        if isinstance(val, str) and val:
            fixed_outs.append(val)
        else:
            fixed_outs.append(f"{block_name}_{mode}_out{stage}_{i}")

    del node.input[:]
    del node.output[:]
    node.input.extend(fixed_ins)
    node.output.extend(fixed_outs)

    return node
# 4. Graph building
def build_graph(manifest: dict, mode: str) -> List[helper.NodeProto]:
    assert mode in ("algo", "applier", "coordinator"), f"Unknown mode: {mode}"

    graph_nodes: List[helper.NodeProto] = []
    # Initialize prev_out to a safe string so stage 1 has a valid input
    prev_out: str = "input"

    print(f"[DEBUG] ===== Building {mode} graph =====")
    for spec in manifest["pipeline"]:
        stage = spec["stage"]
        block_name = spec["block"]
        version = spec["version"]
        coeffs = spec.get("coeff_names", [])

        print(f"[DEBUG] Stage {stage} → block={block_name}, version={version}, coeffs={coeffs}")

        try:
            block = instantiate_block(stage, block_name, version)
        except Exception:
            print(f"[ERROR] Failed to instantiate block=({block_name},{version}) at stage {stage}")
            traceback.print_exc()
            raise

        try:
            if mode == "algo":
                node_or_nodes = block.build_algo_node(prev_out=prev_out)
            elif mode == "applier":
                node_or_nodes = block.build_applier_node(prev_out=prev_out)
            else:
                node_or_nodes = block.build_coordinator_node(prev_out=prev_out)

            nodes = _ensure_node_list(node_or_nodes)
            if not nodes:
                raise RuntimeError(f"Block {block_name} produced no nodes in {mode} mode")

            in_names = block.input_names() or ["input"]
            stage_in = in_names[0]

            rewired_nodes: List[helper.NodeProto] = []
            for n in nodes:
                rn = rewire_inputs(n, stage_in, prev_out)
                rn = _normalize_node(rn, stage, block_name, mode)
                rewired_nodes.append(rn)

            last_outs = list(rewired_nodes[-1].output)
            if not last_outs:
                raise RuntimeError(f"Block {block_name} last node has no outputs")
            prev_out = last_outs[0]

            for rn in rewired_nodes:
                print(f"[DEBUG]   Node name={rn.name}, inputs={list(rn.input)}, outputs={list(rn.output)}")
            graph_nodes.extend(rewired_nodes)

        except Exception:
            print(f"[ERROR] Failed to build stage {stage} block={block_name} version={version} ({mode})")
            traceback.print_exc()
            raise

    print(f"[DEBUG] ===== {mode} graph nodes: {len(graph_nodes)} =====")
    return graph_nodes

# 5. Model and glue emission
def make_model(nodes: List[helper.NodeProto], model_name: str, input_name: str = "input", output_name: str = "output"):
    input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, None)
    output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None)
    graph = helper.make_graph(nodes, f"{model_name}_graph", [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name="softisp_builder")
    return model


def emit_models(algo_nodes: List[helper.NodeProto],
                applier_nodes: List[helper.NodeProto],
                coordinator_nodes: List[helper.NodeProto]) -> None:
    algo_model = make_model(algo_nodes, "algo")
    applier_model = make_model(applier_nodes, "applier")
    coordinator_model = make_model(coordinator_nodes, "coordinator")

    save_model(algo_model, "algo.onnx")
    save_model(applier_model, "applier.onnx")
    save_model(coordinator_model, "coordinator.onnx")

    print("[INFO] Saved algo.onnx, applier.onnx, coordinator.onnx")


def emit_glue(manifest: dict) -> None:
    lines = []
    lines.append("# Auto-generated glue for SoftISP ONNX pipeline")
    lines.append(f"CANONICAL_NAME = '{manifest.get('canonical_name', 'softisp_pipeline')}'")
    lines.append("STAGE_MAP = [")
    for spec in manifest["pipeline"]:
        lines.append(f"    ({spec['stage']}, '{spec['block']}', '{spec['version']}'),")
    lines.append("]")
    lines.append("")
    lines.append("COEFF_SCHEMA = {")
    for spec in manifest["pipeline"]:
        coeffs = spec.get("coeff_names", [])
        lines.append(f"    ('{spec['block']}', '{spec['version']}'): {coeffs},")
    lines.append("}")
    lines.append("")
    lines.append("def get_stage_by_block(block, version):")
    lines.append("    for s, b, v in STAGE_MAP:")
    lines.append("        if b == block and v == version:")
    lines.append("            return s")
    lines.append("    raise KeyError((block, version))")
    lines.append("")
    lines.append("def apply_coeffs(block, version, coeffs):")
    lines.append("    schema = COEFF_SCHEMA.get((block, version), [])")
    lines.append("    missing = [k for k in schema if k not in coeffs]")
    lines.append("    if missing:")
    lines.append("        raise ValueError(f'Missing coeffs: {missing} for {(block, version)}')")
    lines.append("    return True")
    lines.append("")

    with open("glue.py", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[INFO] Generated glue.py (CANONICAL_NAME, STAGE_MAP, COEFF_SCHEMA)")

# 6. Main entry point
def main():
    print("[DEBUG] Importing microblocks to populate registry …")
    import_all_microblocks()
    try:
        reg = dump_registry()
        print(f"[DEBUG] Registry has {len(reg)} entries")
        for (name, version), cls in sorted(reg.items()):
            print(f"[DEBUG]   ({name}, {version}) -> {cls.__name__}")
    except Exception:
        print("[WARN] dump_registry() unavailable; skipping registry dump")

    try:
        manifest = load_manifest()
    except Exception as e:
        print(f"[FATAL] Failed to load manifest {MANIFEST_PATH}: {e}")
        sys.exit(1)

    print(f"[DEBUG] Loaded manifest canonical_name={manifest.get('canonical_name')}")
    try:
        validate_manifest(manifest)
        print("[DEBUG] Manifest validation OK")
    except Exception as e:
        print(f"[FATAL] Manifest validation failed: {e}")
        sys.exit(1)

    # Build graphs
    try:
        algo_nodes = build_graph(manifest, "algo")
        print(f"[DEBUG] Algo graph built with {len(algo_nodes)} nodes")
    except Exception as e:
        print(f"[FATAL] Algo graph build failed: {e}")
        sys.exit(1)

    try:
        applier_nodes = build_graph(manifest, "applier")
        print(f"[DEBUG] Applier graph built with {len(applier_nodes)} nodes")
    except Exception as e:
        print(f"[FATAL] Applier graph build failed: {e}")
        sys.exit(1)

    try:
        coordinator_nodes = build_graph(manifest, "coordinator")
        print(f"[DEBUG] Coordinator graph built with {len(coordinator_nodes)} nodes")
    except Exception as e:
        print(f"[FATAL] Coordinator graph build failed: {e}")
        sys.exit(1)

    # Emit ONNX and glue
    try:
        emit_models(algo_nodes, applier_nodes, coordinator_nodes)
        emit_glue(manifest)
    except Exception as e:
        print(f"[FATAL] Emission failed: {e}")
        sys.exit(1)

    print("[INFO] Build complete.")


if __name__ == "__main__":
    main()
