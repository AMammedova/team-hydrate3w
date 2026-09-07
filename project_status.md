# Project Status — team-hydrate3w
**Early Warning of Hydrate Formation · 3W Dataset 2.0.0 · DLE-AI-202 Track 2**
**Last updated: 2026-09-07 · Deadline: 2026-09-07 23:59 Baku**

> **Test suite: 169 / 169 passing** (as of 2026-09-07 14:23)

---

## Legend
- `[x]` Completed — code exists, tests pass, findings documented
- `[~]` Partial — code exists but incomplete or awaiting real data
- `[ ]` Not started / blocked
- `[!]` BLOCKED — cannot proceed until another member delivers

---

## Global Blockers (affects ALL members)

| # | Blocker | Who delivers | Unblocks |
|---|---|---|---|
| 🔴 | **Real `data/cache/` not built** — `python -m src.data.build_cache` has not run | **M1** runs it | M3, M4, M5 training runs |
| 🔴 | **No `results/results.csv`** — training not executed on real data | **M3 + M4** run training | M5 table generation, report numbers |
| 🔴 | **No `results/model_outputs/*.npz`** — model probability files not written | **M3 + M4** train | M5 threshold selection, lead-time figures |
| 🟡 | **S3 Freeze** — test metrics must not be computed before freeze | All | Report conclusion numbers |

---

## Member 1 — Data Quality & Dataset Story

**Owns:** `src/data/availability.py`, `src/data/stats.py`, figures 1 & 2,
`report/` Dataset section + Introduction, slides 2–3.

| Task | File | Status |
|---|---|---|
| `variable_availability_table()` — channel × well table | [`availability.py`](file:///d:/final/team-hydrate3w/src/data/availability.py) | `[~]` stub exists, logic from `tools/channel_availability.py` not yet migrated |
| `transient_duration_histogram()` — 14-event histogram | [`stats.py`](file:///d:/final/team-hydrate3w/src/data/stats.py) | `[~]` helper `_shade_state_zones` done, histogram function pending |
| `annotated_trace_figure()` — Normal→Transient→Established trace | [`stats.py`](file:///d:/final/team-hydrate3w/src/data/stats.py) | `[~]` `_shade_state_zones` done; full figure function pending |
| `plot_annotated_trace()` wrapper in `plots.py` | [`plots.py`](file:///d:/final/team-hydrate3w/src/eval/plots.py) L30 | `[x]` implemented by M5; M1 must supply real instance data |
| Sensitivity sweep | [`m1_sensitivity_sweep.py`](file:///d:/final/team-hydrate3w/tools/m1_sensitivity_sweep.py) | `[~]` script exists; needs real cache to run |
| Dataset figures | [`m1_dataset_figures.py`](file:///d:/final/team-hydrate3w/tools/m1_dataset_figures.py) | `[~]` script exists; needs real cache to run |
| **Run `build_cache`** on real 3W data | `data/cache/` | `[!]` **CRITICAL — unblocks everyone** |
| Write Dataset section in `report.tex` §II | [`report.tex`](file:///d:/final/team-hydrate3w/report/report.tex) L103 | `[ ]` placeholder at line 103 |
| Write Introduction in `report.tex` §I | [`report.tex`](file:///d:/final/team-hydrate3w/report/report.tex) L87 | `[ ]` placeholder at line 87 |
| Slides 2–3 (Motivation, Dataset) | [`final_slides.md`](file:///d:/final/team-hydrate3w/presentation/final_slides.md) L9 | `[x]` written by M5 — M1 should review |
| NaN label policy (drop) implemented | [`windowing.py`](file:///d:/final/team-hydrate3w/src/data/windowing.py) | `[x]` done; mention in report needed |

> **M1 critical path:** `build_cache` must run before ANY model training can happen.

---

## Member 2 — Splits & Fold Design

**Owns:** `src/data/splits.py`, `fold_report()`, `report/` Experimental Setup +
Discussion sections, slides 4 & 10.

| Task | File | Status |
|---|---|---|
| `GroupedKFoldSplitter` — dual independent split (positive + normal wells) | [`splits.py`](file:///d:/final/team-hydrate3w/src/data/splits.py) | `[x]` fully implemented, 37 tests pass |
| `fold_report()` with `test_normal_hours` column | [`splits.py`](file:///d:/final/team-hydrate3w/src/data/splits.py) L630 | `[x]` column present |
| `CacheIndex` + `load_cache()` + `load_cache_index()` | [`splits.py`](file:///d:/final/team-hydrate3w/src/data/splits.py) L96 | `[x]` fully implemented |
| `val_mode="nested"` with `min_val_normal_hours` guard | [`splits.py`](file:///d:/final/team-hydrate3w/src/data/splits.py) L500 | `[x]` implemented |
| `n_splits=3` confirmed | config | `[x]` default is 3 |
| Guard `train_positive_wells >= 2` per fold (M3 §3 finding) | `splits.py` | `[ ]` **not yet** — M3 found fold 1 has single-well positives in training |
| **Run `fold_report()` on real cache** → `results/fold_report.csv` | CLI | `[!]` waiting on M1's `build_cache` |
| Write Experimental Setup in `report.tex` §IV-A/B | [`report.tex`](file:///d:/final/team-hydrate3w/report/report.tex) L314 | `[ ]` placeholder at line 314 |
| Write Discussion & Limitations in `report.tex` §VII | [`report.tex`](file:///d:/final/team-hydrate3w/report/report.tex) L552 | `[ ]` placeholder at line 552 |
| Slides 4 & 10 (CV setup, Limitations) | [`final_slides.md`](file:///d:/final/team-hydrate3w/presentation/final_slides.md) L19 | `[x]` written by M5 — M2 should review |

---

## Member 3 — Baseline (XGBoost)

**Owns:** `src/baselines/`, `tools/train_xgb.py`, `tools/ablate_features.py`,
calibration, `plot_reliability_diagram`, `report/results_xgboost.md`,
`report.tex` §VI-A, slide 5.

| Task | File | Status |
|---|---|---|
| `features.py` — multi-timescale rolling extractor (3 scales, 6 stats, presence fraction) | [`features.py`](file:///d:/final/team-hydrate3w/src/baselines/features.py) | `[x]` implemented |
| `xgb_model.py` | [`xgb_model.py`](file:///d:/final/team-hydrate3w/src/baselines/xgb_model.py) | `[x]` implemented |
| `tune.py` — inner CV + single-class guard | [`tune.py`](file:///d:/final/team-hydrate3w/src/baselines/tune.py) | `[x]` implemented |
| `calibrate.py` — Platt scaling | [`calibrate.py`](file:///d:/final/team-hydrate3w/src/baselines/calibrate.py) | `[x]` implemented |
| `importance.py` — gain + permutation importance | [`importance.py`](file:///d:/final/team-hydrate3w/src/baselines/importance.py) | `[x]` implemented |
| `plot_reliability_diagram()` | [`plots.py`](file:///d:/final/team-hydrate3w/src/eval/plots.py) L233 | `[x]` fully implemented |
| `tools/train_xgb.py` | [`train_xgb.py`](file:///d:/final/team-hydrate3w/tools/train_xgb.py) | `[x]` complete |
| `tools/ablate_features.py` (9-variant ablation) | [`ablate_features.py`](file:///d:/final/team-hydrate3w/tools/ablate_features.py) | `[x]` complete |
| `tools/probe_presence_shortcut.py` | [`probe_presence_shortcut.py`](file:///d:/final/team-hydrate3w/tools/probe_presence_shortcut.py) | `[x]` complete |
| **Run `train_xgb.py`** on real cache → `results/results.csv` + `model_outputs/` | needs real cache | `[!]` waiting on M1 |
| **Run with `--eval-test`** (AFTER S3 Freeze only) | | `[ ]` intentionally deferred |
| Write XGBoost results in `report.tex` §VI-A | [`report.tex`](file:///d:/final/team-hydrate3w/report/report.tex) L421 | `[x]` shortcut section written; numerical tables need real run |
| `results/results.csv` → `report/tables/` via `aggregate.py` | | `[!]` waiting on training run |
| Slide 5 (Baseline) | [`final_slides.md`](file:///d:/final/team-hydrate3w/presentation/final_slides.md) L24 | `[x]` written by M5 — M3 should review |
| M3 §7a threshold grid bug — fixed | [`thresholds.py`](file:///d:/final/team-hydrate3w/src/eval/thresholds.py) | `[x]` fixed by M5 today |

---

## Member 4 — Deep Models (TCN + GRU)

**Owns:** `src/models/tcn.py`, `src/models/gru.py`, `src/models/dataset.py`,
`src/models/train_loop.py`, `tools/train_deep_models.py`,
`report.tex` §III (Related Work), §IV-D (Architectures), §VI-B (Deep results), slide 6.

| Task | File | Status |
|---|---|---|
| `TCNModel` — 4-block causal dilations (1,2,4,8), receptive field 61 | [`tcn.py`](file:///d:/final/team-hydrate3w/src/models/tcn.py) | `[x]` implemented |
| `GRUModel` — unidirectional, causal | [`gru.py`](file:///d:/final/team-hydrate3w/src/models/gru.py) | `[x]` implemented |
| `HydrateDataset` (window → tensor, mask) | [`dataset.py`](file:///d:/final/team-hydrate3w/src/models/dataset.py) | `[x]` implemented |
| `Trainer` — AMP, grad accumulation, early stopping on PR-AUC, checkpoints | [`train_loop.py`](file:///d:/final/team-hydrate3w/src/models/train_loop.py) | `[x]` fully implemented, 14 tests pass |
| `losses.py` — class-weighted focal loss | [`losses.py`](file:///d:/final/team-hydrate3w/src/models/losses.py) | `[x]` implemented |
| `heads.py` — 3-class classification head | [`heads.py`](file:///d:/final/team-hydrate3w/src/models/heads.py) | `[x]` implemented |
| `profile.py` | [`profile.py`](file:///d:/final/team-hydrate3w/src/models/profile.py) | `[x]` implemented |
| `ssl_pretrain.py` / `finetune.py` | [`ssl_pretrain.py`](file:///d:/final/team-hydrate3w/src/models/ssl_pretrain.py) | `[x]` **OUT OF SCOPE** — `NotImplementedError` stub, frozen per §0.6 |
| `tools/train_deep_models.py` | [`train_deep_models.py`](file:///d:/final/team-hydrate3w/tools/train_deep_models.py) | `[x]` complete |
| **Run TCN + GRU training** (real_only + real_plus_sim) → `results/`, `model_outputs/` | needs cache + GPU | `[!]` waiting on M1's `build_cache` |
| Mask-zeroed ablation run (M3 §4 recommendation) | | `[ ]` optional — M4's call |
| Write Related Work in `report.tex` §III | [`report.tex`](file:///d:/final/team-hydrate3w/report/report.tex) L131 | `[ ]` placeholder at line 131 |
| Write Architecture descriptions in `report.tex` §IV-D | [`report.tex`](file:///d:/final/team-hydrate3w/report/report.tex) L242 | `[ ]` placeholder at line 242 |
| Write Deep model results in `report.tex` §VI-B | [`report.tex`](file:///d:/final/team-hydrate3w/report/report.tex) L505 | `[ ]` placeholder at line 505 |
| Slide 6 (Deep architectures) | [`final_slides.md`](file:///d:/final/team-hydrate3w/presentation/final_slides.md) L29 | `[x]` written by M5 — M4 should review |

---

## Member 5 — Evaluation Pipeline & Report Assembly (YOU)

**Owns:** `src/eval/`, threshold selection, lead-time figures, `report.tex`
(assembly + §IV-C Eval Protocol + Abstract + Conclusion), slides 7–9, 11–12,
`run_all.sh`, AI disclosure.

| Task | File | Status |
|---|---|---|
| `alarm.py` — causal rolling smoothing, `alarm_times()`, `lead_time()` | [`alarm.py`](file:///d:/final/team-hydrate3w/src/eval/alarm.py) | `[x]` done, causal verified by tests |
| `metrics.py` — `positive_score`, `pr_auc`, `event_recall`, `false_alarms_per_operating_hour`, `expected_calibration_error` | [`metrics.py`](file:///d:/final/team-hydrate3w/src/eval/metrics.py) | `[x]` done |
| `thresholds.py` — `select_threshold()` + `select_threshold_curve()` | [`thresholds.py`](file:///d:/final/team-hydrate3w/src/eval/thresholds.py) | `[x]` done |
| **Fix M3 §7a: `_adaptive_grid()` replaces fixed `linspace(0,1,200)`** | [`thresholds.py`](file:///d:/final/team-hydrate3w/src/eval/thresholds.py) | `[x]` **FIXED TODAY** — 4 new regression tests, all pass |
| `aggregate.py` — `load_results()`, `summarize_folds()`, `to_latex_table()`, `generate_all_tables()` | [`aggregate.py`](file:///d:/final/team-hydrate3w/src/eval/aggregate.py) | `[x]` done |
| `plot_lead_time_vs_false_alarm_rate()` | [`plots.py`](file:///d:/final/team-hydrate3w/src/eval/plots.py) L70 | `[x]` done |
| `plot_per_well_lead_time_box()` | [`plots.py`](file:///d:/final/team-hydrate3w/src/eval/plots.py) L151 | `[x]` done |
| `report.tex` structure, §IV-C Eval Protocol, §VI-C table shell, Tools & Acknowledgements | [`report.tex`](file:///d:/final/team-hydrate3w/report/report.tex) | `[x]` done |
| AI disclosure section | [`report.tex`](file:///d:/final/team-hydrate3w/report/report.tex) L568 | `[x]` done |
| `run_all.sh` — end-to-end pipeline entry point | [`run_all.sh`](file:///d:/final/team-hydrate3w/run_all.sh) | `[x]` done |
| `presentation/final_slides.md` — 12-slide deck | [`final_slides.md`](file:///d:/final/team-hydrate3w/presentation/final_slides.md) | `[x]` done (slides 8, 9, 11 have result placeholders) |
| **`report.tex` Abstract** — fill headline numbers | [`report.tex`](file:///d:/final/team-hydrate3w/report/report.tex) L63 | `[ ]` WAITING for real results |
| **`report.tex` Conclusion** — fill best model + sim verdict | [`report.tex`](file:///d:/final/team-hydrate3w/report/report.tex) L563 | `[ ]` WAITING for real results |
| Fill slide 8 (best model + headline) | [`final_slides.md`](file:///d:/final/team-hydrate3w/presentation/final_slides.md) L42 | `[ ]` WAITING |
| Fill slide 9 (sim data verdict) | [`final_slides.md`](file:///d:/final/team-hydrate3w/presentation/final_slides.md) L46 | `[ ]` WAITING |
| Fill slide 11 (recommendation) | [`final_slides.md`](file:///d:/final/team-hydrate3w/presentation/final_slides.md) L55 | `[ ]` WAITING |
| **Commit + push** today's `thresholds.py` fix | git | `[ ]` do now |
| Tag `v1.0-final` | git | `[ ]` after S3 Freeze |

---

## Open Placeholders in `report.tex`

| Line | Owner | What to write |
|---|---|---|
| 43–44 | **All** | University/institution name, member emails |
| 63 | **M5** | Abstract headline: best model, lead-time minutes, event recall % |
| 87 | **M1** | Introduction: motivation, 3W context, research gap |
| 103 | **M1** | Dataset section: statistics, channel decision, NaN policy |
| 131 | **M4** | Related Work: TCN/GRU in industrial time-series |
| 242 | **M4** | Architecture descriptions: TCN receptive field, GRU causality |
| 314 | **M2** | Experimental Setup: grouped CV, simulation protocol |
| 505 | **M4** | Deep model results: TCN and GRU performance both conditions |
| 552 | **M2** | Discussion & Limitations: sample size, generalization |
| 563 | **M5** | Conclusion: winner, simulated data verdict, future work |

---

## What Can Be Done RIGHT NOW (no cache needed)

| Who | Action |
|---|---|
| **M1** | Write Introduction + Dataset section in `report.tex` (numbers in `DATA_FINDINGS.md`) |
| **M1** | **Run `build_cache`** — single highest-priority action in the whole project |
| **M2** | Write Experimental Setup + Discussion in `report.tex` (design finalized) |
| **M4** | Write Related Work + Architecture descriptions in `report.tex` |
| **M4** | Review slide 6 in `final_slides.md` |
| **M5** | `git add src/eval/thresholds.py tests/test_thresholds.py && git commit && git push` |

## What Must Wait for Real Results

| Who | Action | Trigger |
|---|---|---|
| **M3** | Run `train_xgb.py` | after `data/cache/` built |
| **M4** | Run `train_deep_models.py` (TCN + GRU, both conditions) | after `data/cache/` built |
| **M5** | Run `generate_all_tables()` → `report/tables/*.tex` | after `results/results.csv` |
| **M5** | Fill Abstract + Conclusion in `report.tex` | after S3 Freeze + real numbers |
| **M5** | Fill slides 8, 9, 11 | after real numbers |
| **All** | Fill emails + institution name in `report.tex` L43–44 | now (no blocker) |
| **M5** | Tag `v1.0-final` + push | last, after everything above |

---

## Test Suite Summary

```
169 passed, 196 warnings  (2026-09-07 14:23)
```

All 196 warnings are harmless `PyparsingDeprecationWarning` from matplotlib internals.
No failures. `ssl_pretrain.py` and `finetune.py` correctly raise `NotImplementedError` (out-of-scope per §0.6).
