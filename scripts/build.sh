#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
python -m build
echo "Build complete. Distributions are in dist/"
