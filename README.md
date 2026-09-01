# Early Warning for Hydrate Formation — 3W Dataset

Deep learning early-warning system for hydrate formation in offshore well
service lines (3W Dataset, Event 9). **Result 1 (required):** XGBoost vs.
TCN vs. GRU at a matched false-alarm budget, real-only vs. real+simulated
training for each. **Stretch goal (only after Result 1 is complete):**
self-supervised pretraining vs. random init. See
`team_responsibilities_all_members.md` §0.6 for why these are ordered,
not co-equal.

**Task split (5 members, current):** `TEAM_5_MEMBERS.md` — who owns which
file, the open-stub list per member, the day-by-day schedule to the 7 Sep
deadline, and the reduced scope (SSL pretraining is out; see its §0).
`team_responsibilities_all_members.md` remains the 4-member original and is
still authoritative on the shared contract (§0.1–§0.3).

Full spec: `DL_Project_Statement_Hydrate3W.docx` (repo root) — its
**Addendum** section (including A.9, added after a second review) is the
current source of truth wherever it disagrees with the body of the
document. Every module below implements one section of the statement /
one member's ownership area — read the matching section before touching
its TODOs.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Get real data, or start on fake data immediately

```bash
# Real data (multi-GB, takes a while):
bash data/download_data.sh
python -m src.data.inventory --root data/3W/dataset --event 9 --verify   # check filename convention first

# Fake data (seconds, unblocks everyone before the real cache is ready):
python src/data/make_fake_data.py --out data/fake/ --n_instances 40
```

Members 2, 3, and 4 develop against `data/fake/` until Member 1 ships the
real cache — same contract shapes, so nothing downstream needs to change
except the data path.

## Data & model contract (binding for every module)

- `X`, `mask`: `float32`/`uint8` `[N, C, W]` — **channels-first**.
- Shared constants (`NORMAL`/`TRANSIENT`/`ESTABLISHED`, 3W raw label
  codes, the results schema) live in `src/contract.py`. Import from
  there; don't redefine them locally.
- **Model contract:** `XGBoostBaseline` (classical estimator) implements
  `fit(X, mask, y, groups)` / `predict_proba(X, mask)` directly. `TCN`
  and `GRUClassifier` (PyTorch) implement the standard `forward(x, mask)`
  instead — `Trainer` (`src/models/train_loop.py`) is the adapter that
  wraps them into the same `fit`/`predict_proba` shape for Member 4's
  evaluation code to call. Nobody should try to make `TCN.fit()` a real
  method; that's what `Trainer` is for.
- `results/results.csv` columns: `model, fold, seed, condition, metric_name, value`.
  `condition` includes `real_only` / `real_plus_sim` (Result 1, required)
  and `pretrained` / `random_init` (stretch goal, only if attempted). No
  number in the paper is ever typed by hand — everything comes from `results/`.
- Window labels: `NORMAL` / `TRANSIENT` / `ESTABLISHED`, assigned by
  `src.data.windowing.label_window()`, default rule `most_severe` — the
  most severe state present anywhere in the window. Mathematically
  identical to `final_timestep` whenever an instance's severity is
  monotonic (the expected case — no reversion from Established back to
  Transient); provides a free safety margin if it isn't.
  `is_monotonic_severity()` checks this per-instance — Member 1 runs it
  during inventory (W1.3) and reports the result.
- Alarm smoothing (`src.eval.alarm.alarm_times`) is **causal (trailing)
  only** — never centered. `tests/test_alarm.py` has a regression test
  for this specifically.

## Repository layout (per team_responsibilities_all_members.md)

```
src/
|-- contract.py     # shared constants -- import, don't redefine
|-- data/           # Member 1 -- inventory, windowing, splits, availability, cache, fake data
|-- baselines/      # Member 2 -- rolling features, XGBoost, tuning, calibration
|-- models/         # Member 3 -- TCN, GRU, SSL pretraining (stretch), fine-tuning, train loop
`-- eval/           # Member 4 -- metrics, alarm definition, thresholds, aggregation, plots
run_all.sh           # Member 4 -- single reproduction entry point
results/             # results.csv, instance_inventory.csv -- generated, not hand-edited
report/, presentation/, contribution_report.pdf
```

## Status

| Module | File | Status |
|---|---|---|
| 1 — Instance inventory | `src/data/inventory.py` | implemented + tested (8/8) |
| 1 — Fake data generator | `src/data/make_fake_data.py` | implemented + verified against the contract |
| 1 — Download / availability / cache build / stats figures | `src/data/download.py`, `availability.py`, `build_cache.py`, `stats.py` | stubs, TODO |
| 1 — Windowing/labeling | `src/data/windowing.py` | `label_window()`, `is_monotonic_severity()` implemented + tested (12/12, incl. the any_transient bug fix and the most_severe/final_timestep equivalence proof); `build_windows()` TODO |
| 1 — Grouped CV (nested train/val/test) | `src/data/splits.py` | interface frozen, TODO |
| — — Shared contract | `src/contract.py` | implemented |
| 2 — Multi-timescale features | `src/baselines/features.py` | interface frozen (mask-aware, 3 scales), TODO |
| 2 — XGBoost baseline | `src/baselines/xgb_model.py` | implemented, mask-aware, `compute_sample_weight()` for imbalance (not `scale_pos_weight` — binary-only, no-op under multiclass) |
| 2 — Tuning / calibration / importance | `src/baselines/tune.py`, `calibrate.py`, `importance.py` | stubs, TODO |
| 3 — TCN | `src/models/tcn.py` | implemented, channels-first + mask, receptive field verified = 61 |
| 3 — GRU | `src/models/gru.py` | implemented, channels-first + mask |
| 3 — Heads | `src/models/heads.py` | `ClassificationHead` implemented; `ReconstructionHead` TODO |
| 3 — Trainer | `src/models/train_loop.py` | interface frozen (mask-aware), TODO |
| 3 — SSL pretraining / fine-tuning / profiling **(stretch goal — do not start before Result 1 works)** | `ssl_pretrain.py`, `finetune.py`, `profile.py` | stubs, TODO |
| 4 — Metrics | `src/eval/metrics.py` | `positive_score`, `pr_auc`, `pr_auc_per_class` implemented; `expected_calibration_error` TODO |
| 4 — Alarm definition + lead time | `src/eval/alarm.py` | implemented + tested (9/9, incl. a causality regression test) |
| 4 — Threshold selection | `src/eval/thresholds.py` | interface frozen (target_far=1/100h default), TODO |
| 4 — Aggregation | `src/eval/aggregate.py` | `summarize_folds` implemented; `to_latex_table` TODO |
| 4 — Plots | `src/eval/plots.py` | stubs for all four required figures, TODO |

## Run today

```bash
# Full test suite (29 tests, synthetic fixtures, no download needed)
pytest tests/ -v

# Fake data generator
python src/data/make_fake_data.py --out data/fake/ --n_instances 40

# Dataset summary once you've downloaded the real data -- fills in the
# floor-justification numbers and real-well-count needed by the report
bash run_all.sh --root data/3W/dataset --event 9
```

## Dataset & licensing

3W Dataset 2.0.0 (Petrobras), files under CC BY 4.0; toolkit code under
Apache 2.0. Source: https://github.com/petrobras/3W. Cite in `report/report.tex`.

## Tools & Acknowledgements (fill in before submission)

Disclose any substantial AI assistance here per the brief's academic
integrity requirement.
