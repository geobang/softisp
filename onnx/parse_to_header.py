import json, os, onnx
from common import load_manifest

def load_manifest(manifest_path):
    with open(manifest_path, "r") as f: return json.load(f)

def collect_required_coeffs(manifest):
    req = {}
    for spec in manifest.get("pipeline", []):
        blk = spec["block"].lower()
        coeffs = spec.get("coeff_names", [])
        req.setdefault(blk, [])
        for c in coeffs:
            if c not in req[blk]:
                req[blk].append(c)
    return req

def onnx_to_header(onnx_file, header_file):
    model = onnx.load(onnx_file)
    inputs  = [i.name for i in model.graph.input]
    outputs = [o.name for o in model.graph.output]
    guard = os.path.basename(header_file).replace(".", "_").upper()
    with open(header_file, "w") as f:
        f.write(f"/* Auto-generated from {os.path.basename(onnx_file)} */\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("/* Inputs */\n")
        for name in inputs:  f.write(f"#define INPUT_{name} \"{name}\"\n")
        f.write("\n/* Outputs */\n")
        for name in outputs: f.write(f"#define OUTPUT_{name} \"{name}\"\n")
        f.write(f"\n#endif /* {guard} */\n")

def emit_coeff_struct(header_path, canonical, required):
    guard = f"{canonical}_COEFFS_BULK_H".upper()
    with open(header_path, "w") as f:
        f.write(f"/* Auto-generated coeffs bulk for {canonical} */\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("typedef float coeff_t;\n\n")
        f.write(f"typedef struct {canonical}_coeffs_bulk {{\n")
        for blk, coeffs in required.items():
            for c in coeffs:
                f.write(f"    coeff_t {blk}_{c};\n")
        f.write(f"}} {canonical}_coeffs_bulk;\n\n")
        f.write(f"#endif /* {guard} */\n")

def generate_headers_bundle(manifest_path, outdir, canonical):
    onnx_to_header(f"{outdir}/{canonical}_algo.onnx",  f"{outdir}/{canonical}_algo.h")
    onnx_to_header(f"{outdir}/{canonical}_isp.onnx",   f"{outdir}/{canonical}_isp.h")
    onnx_to_header(f"{outdir}/{canonical}_coord.onnx", f"{outdir}/{canonical}_coord.h")
    req = collect_required_coeffs(load_manifest(manifest_path))
    emit_coeff_struct(f"{outdir}/{canonical}_coeffs_bulk.h", canonical, req)
