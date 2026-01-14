import os, sys, json, logging, onnx
import onnx.helper as oh
from microblocks.registry import REGISTRY, import_all_microblocks, dump_registry

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(message)s")

def load_manifest(path: str):
    with open(path, "r") as f:
        return json.load(f)

def save_model(nodes, inits, vis, graph_inputs, final_out, out_path, canonical_name):
    """
    Build and save the ONNX model.
    - nodes: list[onnx.NodeProto]
    - inits: list[onnx.TensorProto]
    - vis: list[onnx.ValueInfoProto] (non-input metadata)
    - graph_inputs: list[onnx.ValueInfoProto] (actual graph inputs)
    - final_out: str (tensor name for graph output)
    """
    outputs = [oh.make_tensor_value_info(final_out, onnx.TensorProto.FLOAT, ["N","C","H","W"])]

    # Use keyword args to avoid positional order mistakes
    graph = oh.make_graph(
        nodes=nodes,
        name=canonical_name,
        inputs=graph_inputs,
        outputs=outputs,
        initializer=inits,
        value_info=vis
    )

    model = oh.make_model(
        graph,
        producer_name="softisp_rewrite",
        opset_imports=[onnx.helper.make_operatorsetid("", 13)]
    )
    onnx.checker.check_model(model)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    onnx.save(model, out_path)
    logging.info(f"Saved ONNX model to {out_path}")

def build_all(manifest_file: str, mode: str = "applier"):
    manifest = load_manifest(manifest_file)
    stages_spec = manifest["stages"]

    nodes, inits, vis = [], [], []
    graph_inputs = []  # actual graph inputs (ValueInfoProto)
    produced = set()
    declared_inputs = set()  # names declared by stages to be fed from outside
    final_out = None

    # Seed the canonical input_image as a graph input (common convention)
    input_image_vi = oh.make_tensor_value_info("input_image", onnx.TensorProto.FLOAT, ["N","C","H","W"])
    graph_inputs.append(input_image_vi)

    for stage_name, spec in stages_spec.items():
        cls_key = (spec["class"], spec["version"])
        if cls_key not in REGISTRY:
            raise KeyError(f"Registry missing class {cls_key}")
        mb_cls = REGISTRY[cls_key]
        mb = mb_cls()

        logging.info(f"Building stage {stage_name}: {spec['class']} v{spec['version']}")
        logging.debug(f"Declared inputs: {spec.get('inputs', [])}")

        try:

            if mode == "applier":
                outputs, stage_nodes, stage_inits, stage_vis = mb.build_applier(
                    stage_name, prev_stages=spec.get("inputs", [])
                )
            elif mode == "algo" and hasattr(mb, "build_algo"):
                outputs, stage_nodes, stage_inits, stage_vis = mb.build_algo(
                stage_name, prev_stages=spec.get("inputs", [])
                )
            else:
                logging.warning(f"Stage {stage_name} has no build_algo, skipping")
                continue

        except Exception as e:
            logging.error(f"Exception in stage {stage_name} ({spec['class']} v{spec['version']}): {e}")
            raise

        nodes.extend(stage_nodes)
        inits.extend(stage_inits)
        vis.extend(stage_vis)

        # Track produced tensors and candidate final output
        for out in outputs.values():
            produced.add(out["name"])
            logging.debug(f"Produced tensor: {out['name']}")
            if out["name"].endswith(".applier") or out["name"].endswith(".image"):
                final_out = out["name"]

        # Discover stage-scoped needs to seed as graph inputs, if present in ValueInfo
        # Convention: any ValueInfo named f"{stage}.{need_suffix}" that isn’t produced becomes a graph input.
        for v in stage_vis:
            name = v.name
            # Treat value_infos that look like needs (stage-scoped) and aren’t produced as inputs
            if name.startswith(f"{stage_name}.") and name not in produced:
                declared_inputs.add(name)

    # Promote declared stage-scoped needs to graph inputs
    # Avoid duplicates with already added "input_image"
    existing_input_names = {vi.name for vi in graph_inputs}
    for v in vis:
        if v.name in declared_inputs and v.name not in existing_input_names:
            graph_inputs.append(v)
            existing_input_names.add(v.name)
            logging.debug(f"Added graph input: {v.name}")

    if not final_out:
        raise RuntimeError("No final output tensor produced")

    out_path = os.path.join("onnx_out", manifest["canonical_name"], f"{mode}.onnx")
    save_model(nodes, inits, vis, graph_inputs, final_out, out_path, manifest["canonical_name"])

if __name__ == "__main__":
    import_all_microblocks()
    logging.info(f"Registry contents: {dump_registry()}")
    manifest_file = sys.argv[1] if len(sys.argv) > 1 else "pipeline.json"
    build_all(manifest_file, mode="applier")
    build_all(manifest_file, mode="algo")
