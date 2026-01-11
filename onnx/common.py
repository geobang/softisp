import json
from onnx import helper
from registry import get_block

def load_manifest(path):
    """Load a JSON manifest file."""
    with open(path, "r") as f:
        return json.load(f)

def instantiate_block(stage, block, version):
    """
    Create a block instance from the registry.
    """
    cls = get_block(block, version)
    if cls is None:
        raise ValueError(f"Unknown block {block} version {version}")
    return cls(stage, block, version)

def rewire_inputs(node, stage_in, prev_out):
    """
    Replace node input with previous stage output.
    """
    if node is None:
        return None
    inputs = [prev_out if i == stage_in else i for i in node.input]
    node.input[:] = inputs
    return node

def finalize_graph(nodes, final_out, which):
    """
    Wrap nodes into a full ONNX graph with a single input and output.
    """
    graph = helper.make_graph(
        nodes,
        f"{which}_graph",
        [helper.make_tensor_value_info("model_input", 1, None)],
        [helper.make_tensor_value_info(final_out, 1, None)]
    )
    return helper.make_model(graph)
