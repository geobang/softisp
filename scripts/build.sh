#!/usr/bin/env bash
set -euo pipefail
# Simple build script for Phase A
echo "Installing Python dependencies (pip)"
python -m pip install --upgrade pip
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  pip install numpy pydantic pytest
fi
echo "Build step completed (Phase A has no compilation step)."