#!/usr/bin/env bash
# Single-command reproduction entry point (team contract §0.3/§4.6, W4.6).
# The brief lists "no run_all entry point" as an automatic deduction --
# this is that entry point.
#
#   bash run_all.sh --root data/3W/dataset --event 9
#
# Currently wired through Module 1 (dataset summary) only. As Modules
# 2-8 land (see README.md's status table), extend this script to drive
# windowing -> caching -> the grouped-CV sweep -> results aggregation,
# writing results/fold_metrics.csv, results/summary.csv, and every
# figure/table the report references. No number in the paper should
# ever be typed by hand (team contract §0.3).

set -euo pipefail
ROOT="data/3W/dataset"
EVENT=9

python -m src.data.inventory --root "$ROOT" --event "$EVENT" "$@"
