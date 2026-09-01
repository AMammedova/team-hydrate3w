#!/usr/bin/env bash
# Downloads the 3W Dataset (Petrobras) into data/3W/.
#
# The dataset is large (multi-GB, Parquet files, CC BY 4.0). This clones
# the official petrobras/3W repository, which contains both the toolkit
# (Apache 2.0) and the dataset (CC BY 4.0) under 3W/dataset/.
#
# Run from the repo root: bash data/download_data.sh

set -euo pipefail

TARGET_DIR="data/3W"

if [ -d "$TARGET_DIR" ]; then
  echo "data/3W already exists, skipping clone. Delete it to re-download."
  exit 0
fi

echo "Cloning petrobras/3W (this may take a while, dataset is multi-GB)..."
git clone --depth 1 https://github.com/petrobras/3W.git "$TARGET_DIR"

echo ""
echo "Done. Dataset root for ThreeWDataset(root_dir=...) is:"
echo "  ${TARGET_DIR}/dataset"
echo ""
echo "Sanity-check the filename convention actually shipped in this clone"
echo "before running any CV fold on it (see src/data/inventory.py docstring):"
echo ""
echo "  python -m src.data.inventory --root ${TARGET_DIR}/dataset --event 9 --verify"
