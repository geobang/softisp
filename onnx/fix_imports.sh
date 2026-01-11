#!/bin/bash
# Fail-proof sanity fixer for SoftISP microblocks (Option B: cd onnx && python build_all.py)

set -e

ROOT="microblocks"

echo "🔧 Step 1: Ensure __init__.py exists in microblocks and subfolders..."
touch "$ROOT/__init__.py"
for d in "$ROOT"/*/; do
  [ -d "$d" ] && touch "$d/__init__.py"
done

echo "🔧 Step 2: Define desired import block..."
IMPORTS="from registry import register_block
from microblocks.base import MicroBlock
from onnx import helper

"

fixed=0
fine=0
issues=0

echo "🔧 Step 3: Process all microblock files..."
find "$ROOT" -type f -name "*.py" ! -name "__init__.py" ! -name "base.py" | while read -r file; do
  cp "$file" "$file.bak"

  # Remove ALL known bad imports
  sed -i -E '
    /from[[:space:]]+onnx\.microblocks\.base[[:space:]]+import[[:space:]]+MicroBlock/d;
    /from[[:space:]]+microblocks\.base[[:space:]]+import[[:space:]]+MicroBlock/d;
    /from[[:space:]]+base[[:space:]]+import[[:space:]]+MicroBlock/d;
    /from[[:space:]]+onnx\.registry[[:space:]]+import[[:space:]]+register_block/d;
    /from[[:space:]]+registry[[:space:]]+import[[:space:]]+register_block/d;
    /from[[:space:]]+onnx[[:space:]]+import[[:space:]]+helper/d;
    /^import[[:space:]]+onnx\.microblocks/d
  ' "$file"

  # Prepend correct imports
  tmpfile=$(mktemp)
  printf "%s" "$IMPORTS" > "$tmpfile"
  cat "$file" >> "$tmpfile"
  mv "$tmpfile" "$file"

  # Sanity check: ensure @register_block exists
  if ! grep -q "@register_block" "$file"; then
    echo "⚠️  $file missing @register_block decorator"
    issues=$((issues+1))
  fi

  # Log outcome
  if grep -q "from microblocks.base import MicroBlock" "$file" && \
     grep -q "from registry import register_block" "$file" && \
     grep -q "from onnx import helper" "$file"; then
    echo "🔧 $file fixed"
    fixed=$((fixed+1))
  else
    echo "✅ $file already fine"
    fine=$((fine+1))
  fi
done

echo "📊 Summary: $fixed files fixed, $fine already fine, $issues with issues"

echo "🔎 Step 4: Quick registry check..."
python - <<'EOF' || true
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import registry
try:
    registry.import_all_microblocks()
    print("Registered blocks:", sorted(registry.BLOCK_REGISTRY.keys()))
except Exception as e:
    print("⚠️ Registry import failed:", e)
EOF
