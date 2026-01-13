#!/bin/bash
# fix_import.sh
# Ensure every microblocks subfolder has an __init__.py
# and fix known bad imports inside .py files.

BASE_DIR="$(dirname "$0")/microblocks"

echo "[INFO] Scanning $BASE_DIR for subdirectories…"

# Add __init__.py to the top-level microblocks folder
if [ ! -f "$BASE_DIR/__init__.py" ]; then
  echo "# Marks microblocks as a Python package" > "$BASE_DIR/__init__.py"
  echo "[INFO] Created $BASE_DIR/__init__.py"
fi

# Recursively add __init__.py to all subfolders
find "$BASE_DIR" -type d | while read -r dir; do
  init_file="$dir/__init__.py"
  if [ ! -f "$init_file" ]; then
    echo "# Marks this directory as a Python package" > "$init_file"
    echo "[INFO] Created $init_file"
  fi
done

echo "[INFO] Fixing known import issues…"

# 1. Replace old registry import
find "$BASE_DIR" -type f -name "*.py" -print0 | xargs -0 sed -i \
  's/from[[:space:]]\+registry[[:space:]]\+import[[:space:]]\+register_block/from microblocks.base import MicroblockBase/g'

# 2. Replace bad microblock_base import
find "$BASE_DIR" -type f -name "*.py" -print0 | xargs -0 sed -i \
  's/from[[:space:]]\+microblocks\.microblock_base[[:space:]]\+import[[:space:]]\+MicroblockBase/from microblocks.base import MicroblockBase/g'

# 3. Replace bad subpackage base imports (e.g. microblocks.cropresize.base)
find "$BASE_DIR" -type f -name "*.py" -print0 | xargs -0 sed -i \
  's/from[[:space:]]\+microblocks\.[a-zA-Z0-9_]\+\.base[[:space:]]\+import[[:space:]]\+MicroblockBase/from microblocks.base import MicroblockBase/g'

# Replace any "from microblocks.<subpackage>.base import MicroblockBase"
# with "from microblocks.base import MicroblockBase"
find  "$BASE_DIR" -type f -name "*.py" -print0 | xargs -0 sed -i \
  's/from[[:space:]]\+microblocks\.[a-zA-Z0-9_]\+\.base[[:space:]]\+import[[:space:]]\+MicroblockBase/from microblocks.base import MicroblockBase/g'

# Remove bad MicroBlock import lines
find   "$BASE_DIR" -type f -name "*.py" -print0 | xargs -0 sed -i \
  '/from microblocks\.base import MicroBlock/d'


echo "[INFO] Done. __init__.py files created and imports fixed."
