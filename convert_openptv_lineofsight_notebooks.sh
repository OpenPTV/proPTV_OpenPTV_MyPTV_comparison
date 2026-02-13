#!/usr/bin/env bash
set -euo pipefail

find OpenPTV_LineOfSight -name '*.ipynb' -print0 | while IFS= read -r -d '' file; do
  out="${file%.ipynb}_nb.py"
  uv run --python .venv/bin/python marimo -q -y convert "$file" -o "$out"
done
