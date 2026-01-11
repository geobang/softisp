#!/bin/bash
# Smart fixer for SoftISP microblocks:
# - Ensures @register_block + class skeleton exists
# - Enforces name/version from filename
# - Adds missing methods with valid bodies inside class
# - Repairs empty/half-written method definitions
# Usage: bash fix_methods.sh

set -euo pipefail

ROOT="microblocks"

echo "🔧 Scanning microblock files..."

find "$ROOT" -type f -name "*.py" ! -name "__init__.py" ! -name "base.py" | while read -r file; do
  cp "$file" "$file.bak"
  echo "• Fixing $file"

  python - <<'PY' "$file"
import re, sys, os

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# Infer names from filename
fname = os.path.basename(path)[:-3]  # strip .py
mver = re.search(r'_v(\d+)$', fname)
version = f"v{mver.group(1)}" if mver else "v1"
base = re.sub(r'_v\d+$', '', fname)
blockname = base.replace('_', '')
classname = ''.join(p.capitalize() for p in fname.split('_'))

required_methods = [
    "input_names",
    "output_names",
    "build_algo_node",
    "build_applier_node",
    "build_coordinator_node",
]

def stub_for(method, clsname):
    if method == "input_names":
        return [
            "    def input_names(self):",
            "        return [\"input\"]",
            "",
        ]
    if method == "output_names":
        return [
            "    def output_names(self):",
            "        return [\"output\"]",
            "",
        ]
    name_map = {
        "build_algo_node": "AlgoStub",
        "build_applier_node": "ApplierStub",
        "build_coordinator_node": "CoordinatorStub",
    }
    tag = name_map[method]
    return [
        f"    def {method}(self, prev_out=None):",
        "        from onnx import helper",
        "        node = helper.make_node(",
        "            \"Identity\",",
        "            inputs=[prev_out or \"input\"],",
        "            outputs=[\"output\"],",
        f"            name=\"{clsname}{tag}\"",
        "        )",
        "        return node",
        "",
    ]

# Ensure there is a @register_block + class; if not, create skeleton
has_decorator_class = re.search(
    r'^\s*@register_block\s*\n\s*class\s+\w+\(MicroBlock\)\s*:',
    src, re.M
)
has_class = re.search(r'^\s*class\s+\w+\(MicroBlock\)\s*:', src, re.M)

if not has_decorator_class and not has_class:
    print("  ↳ No class found; inserting skeleton")
    new_src = "\n".join([
        "from registry import register_block",
        "from microblocks.base import MicroBlock",
        "from onnx import helper",
        "",
        "@register_block",
        f"class {classname}(MicroBlock):",
        f"    name = \"{blockname}\"",
        f"    version = \"{version}\"",
        "",
        *stub_for("input_names", classname),
        *stub_for("output_names", classname),
        *stub_for("build_algo_node", classname),
        *stub_for("build_applier_node", classname),
        *stub_for("build_coordinator_node", classname),
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_src)
    sys.exit(0)

# Normalize: ensure @register_block decorator exists above the class
lines = src.splitlines()
class_line_idx = None
decorator_idx = None
for i, line in enumerate(lines):
    if re.match(r'^\s*class\s+\w+\(MicroBlock\)\s*:', line):
        class_line_idx = i
        break
for i in range(max(0, (class_line_idx or 0) - 3), (class_line_idx or 0) + 1):
    if i < 0 or i >= len(lines): continue
    if re.match(r'^\s*@register_block\s*$', lines[i]):
        decorator_idx = i
        break
if class_line_idx is not None and decorator_idx is None:
    print("  ↳ Adding @register_block decorator")
    lines.insert(class_line_idx, "@register_block")
    class_line_idx += 1  # class line moved down

# Enforce name/version attributes immediately after class line
class_indent = re.match(r'^(\s*)class', lines[class_line_idx]).group(1)
insert_pos = class_line_idx + 1

def upsert_attr(attr, value):
    # search within class block for attr; update if present, insert if missing
    found = False
    for j in range(class_line_idx + 1, len(lines)):
        if re.match(rf'^{class_indent}class\s+\w+\(MicroBlock\)\s*:', lines[j]):
            break
        m = re.match(rf'^{class_indent}\s+{attr}\s*=\s*["\']([^"\']+)["\']', lines[j])
        if m:
            found = True
            if m.group(1) != value:
                lines[j] = f'{class_indent}    {attr} = "{value}"'
            break
    if not found:
        lines.insert(insert_pos, f'{class_indent}    {attr} = "{value}"')

upsert_attr("name", blockname)
upsert_attr("version", version)

# Method repair: ensure methods exist and have bodies
text = "\n".join(lines)

def find_method_def(method):
    pat = re.compile(rf'^{class_indent}    def\s+{method}\s*\(.*\)\s*:', re.M)
    m = pat.search(text)
    if not m:
        return None
    # find exact line number
    start = text[:m.start()].count("\n")
    return start

def method_has_body(def_idx):
    # scan lines after def until next def/class or dedent
    for k in range(def_idx + 1, len(lines)):
        s = lines[k]
        if s.strip() == "":
            continue
        if re.match(rf'^{class_indent}class\s+\w+\(MicroBlock\)\s*:', s):
            return False
        if re.match(rf'^{class_indent}    def\s+\w+\s*\(', s):
            return False
        # body must be more indented than def (class_indent + 8 spaces or more)
        if re.match(rf'^{class_indent}        \S', s):
            return True
        # docstring or pass counts as body
        if re.match(rf'^{class_indent}        ("""|pass|#)', s):
            return True
        # if dedented to class level content appears, no body
        if re.match(rf'^{class_indent}    \S', s):
            return False
    return False

changed = False
for method in required_methods:
    idx = find_method_def(method)
    if idx is None:
        print(f"  ↳ Missing method: {method} → inserting stub")
        lines[insert_pos:insert_pos] = stub_for(method, classname)
        insert_pos += len(stub_for(method, classname))
        changed = True
    else:
        if not method_has_body(idx):
            print(f"  ↳ Empty method: {method} → repairing body")
            body = stub_for(method, classname)[1:]  # drop def line; keep body
            insert_at = idx + 1
            lines[insert_at:insert_at] = body
            changed = True

# Write back if changed
if changed or decorator_idx is None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
else:
    print("  ↳ Already compliant")

PY

done

echo "✅ Fix completed. Backups saved as .bak"
