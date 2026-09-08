# Project Status — team-hydrate3w

**Early Warning of Hydrate Formation · 3W Dataset 2.0.0 · DLE-AI-202 Track 2**

**Last updated: 2026-09-09**

> **Current project test suite:** 177 passed, 0 failed (2026-09-09).  
> All project tests run via `pytest` (configured via `pytest.ini` to discover `tests/`).

---

## Legend

- `[x]` Completed — code exists, tests pass, or the required validation run has completed
- `[~]` Partial — code exists but final integration / real run / review remains
- `[ ]` Not started / pending
- `[!]` BLOCKED — cannot proceed until another dependency is delivered

---

## Global Status / Remaining Blockers

| # | Item | Status | Notes |
|---|---|---|---|
| 🟢 | **Real `data/cache/`** | `[x]` | Built successfully from 3W Dataset 2.0.0 using the frozen 5-channel set |
| 🟢 | **Real grouped folds** | `[x]` | 3-fold nested grouped CV generated and inspected; folds are now frozen |
| 🟢 | **M4 CPU end-to-end smoke** | `[x]` | GRU and TCN both completed all 3 folds for 2 epochs on the real cache |
| 🟢 | **`run_all.sh` final integration** | `[x]` | M2 splits, M4 training, and M5 evaluation + LaTeX tables wired end-to-end |
| 🟢 | **M5 evaluation runner** | `[x]` | `src/eval/evaluate_predictions.py` wires `_val.npz` / `_test.npz` to `results.csv` and figures |
| 🟡 | **GPU CUDA/A100 smoke** | `[ ]` | Must be run in the Academy GPU environment before full deep-model training |
| 🔴 | **Final real model runs / `results/results.csv`** | `[!]` | Requires M3 final baseline run + M4 full GPU runs |
| 🔴 | **Final model probability outputs** | `[!]` | Requires final M3/M4 runs |
| 🟡 | **S3 Freeze / final test evaluation** | `[ ]` | Test metrics evaluated once after full model training is complete |

---

## Frozen Data / Split Configuration

### Real cache

Built with:

```powershell
python -m src.data.build_cache `
  --root data/3W/dataset `
  --out data/cache `
  --channels P-MON-CKP,P-JUS-CKGL,T-TPT,T-JUS-CKP,P-ANULAR
```

Frozen primary channel set:

1. `P-MON-CKP`
2. `P-JUS-CKGL`
3. `T-TPT`
4. `T-JUS-CKP`
5. `P-ANULAR`

Successful cache summary:

| Source | Windows | Transient | Established |
|---|---:|---:|---:|
| real | 42,259 | 1,295 | 118 |
| simulated | 43,829 | 18,422 | 21,139 |
| **total** | **86,088** |  |  |

- 801 instances written
- 48,882 windows dropped by the existing NaN-label / minimum-valid-fraction rules
- 287 instances produced zero usable windows
- The resulting real-window counts match the project's previously documented main-arm cache counts

### Frozen split command

```powershell
python -m src.data.splits `
  --cache data/cache `
  --n-splits 3 `
  --n-repeats 1 `
  --val-mode nested
```

Real fold report:

| Fold | Train positive wells | Val positive wells | Test positive wells | Val positive events | Test positive events | Val normal hours | Test normal hours |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 1 | 1 | 1 | 5 | 530.7 | 509.2 |
| 1 | 2 | 1 | 4 | 3 | 5 | 305.2 | 561.8 |
| 2 | 4 | 1 | 2 | 5 | 4 | 509.2 | 530.7 |

Important limitation: fold 0 has only one positive validation event, so validation PR-AUC / early-stopping decisions for that fold are intrinsically unstable. This should be documented, not "fixed" after seeing test results.

---

## Member 1 — Data Quality & Dataset Story

**Owns:** `src/data/availability.py`, `src/data/stats.py`, figures 1 & 2, report Dataset section + Introduction, slides 2–3.

| Task | File | Status |
|---|---|---|
| `variable_availability_table()` — channel × well table | `src/data/availability.py` | `[~]` stub exists; migration from tooling may still be needed |
| `transient_duration_histogram()` — 14-event histogram | `src/data/stats.py` | `[~]` |
| `annotated_trace_figure()` — Normal→Transient→Established trace | `src/data/stats.py` | `[~]` |
| `plot_annotated_trace()` wrapper | `src/eval/plots.py` | `[x]` |
| Sensitivity sweep | `tools/m1_sensitivity_sweep.py` | `[~]` script exists; real cache is now available to run it |
| Dataset figures | `tools/m1_dataset_figures.py` | `[~]` script exists; real cache is now available to run it |
| **Build real cache** | `data/cache/` | `[x]` **DONE 2026-09-08** |
| Validate real cache headline counts | cache output | `[x]` **DONE** |
| Write Dataset section | `report/report.tex` | `[ ]` |
| Write Introduction | `report/report.tex` | `[ ]` |
| Slides 2–3 review | `presentation/final_slides.md` | `[x]` slides exist; M1 review remains |

> M1 is no longer the global blocker. The canonical real cache now exists locally and has been validated.

---

## Member 2 — Splits & Fold Design

**Owns:** `src/data/splits.py`, `fold_report()`, Experimental Setup + Discussion, slides 4 & 10.

| Task | File | Status |
|---|---|---|
| `GroupedKFoldSplitter` — dual independent split | `src/data/splits.py` | `[x]` |
| `fold_report()` with test Normal-hours column | `src/data/splits.py` | `[x]` |
| `CacheIndex` + `load_cache()` + `load_cache_index()` | `src/data/splits.py` | `[x]` |
| `val_mode="nested"` with Normal-hour allocation | `src/data/splits.py` | `[x]` |
| `n_splits=3` | config | `[x]` |
| Guard `train_positive_wells >= 2` | `src/data/splits.py` | `[x]` **implemented and regression-tested** |
| Positive-well columns in `fold_report()` | `src/data/splits.py` | `[x]` |
| Run real `fold_report()` | `results/fold_report.csv` | `[x]` **DONE 2026-09-08** |
| Inspect real fold quality | real fold report | `[x]` **DONE** |
| Freeze split configuration | team decision | `[x]` **freeze 3 folds / 1 repeat / nested; do not redesign after test results** |
| Write Experimental Setup | `report/report.tex` | `[ ]` |
| Write Discussion & Limitations | `report/report.tex` | `[ ]` |
| Slides 4 & 10 review | `presentation/final_slides.md` | `[x]` slides exist; M2 review remains |

> M2's splitter implementation and real-fold validation are complete. The principal reporting caveat is the single positive validation event in fold 0.

---

## Member 3 — Baseline (XGBoost)

**Owns:** `src/baselines/`, `tools/train_xgb.py`, ablations, calibration, baseline results, slide 5.

| Task | File | Status |
|---|---|---|
| `features.py` — multi-timescale extractor | `src/baselines/features.py` | `[x]` |
| `xgb_model.py` | `src/baselines/xgb_model.py` | `[x]` |
| `tune.py` | `src/baselines/tune.py` | `[x]` |
| `calibrate.py` — Platt scaling | `src/baselines/calibrate.py` | `[x]` |
| `importance.py` | `src/baselines/importance.py` | `[x]` |
| Reliability plot | `src/eval/plots.py` | `[x]` |
| `tools/train_xgb.py` | `tools/train_xgb.py` | `[x]` code complete |
| Feature ablation | `tools/ablate_features.py` | `[x]` |
| Presence-shortcut probe | `tools/probe_presence_shortcut.py` | `[x]` |
| **Run final baseline on the frozen real cache/splits** | `results/` | `[ ]` **can now run; no longer blocked by cache** |
| Run test evaluation | final evaluation | `[ ]` **only after S3 Freeze** |
| Final baseline result tables | `report/tables/` | `[!]` waits for final real run |
| Slide 5 review | `presentation/final_slides.md` | `[x]` |

---

## Member 4 — Deep Models (TCN + GRU)

**Owns:** `src/models/tcn.py`, `src/models/gru.py`, `src/models/dataset.py`, `src/models/train_loop.py`, `tools/train_deep_models.py`, Related Work, architectures, deep results, slide 6.

| Task | File | Status |
|---|---|---|
| `TCNModel` — 4 causal blocks, dilations (1,2,4,8), RF=61 | `src/models/tcn.py` | `[x]` |
| `GRUModel` — unidirectional | `src/models/gru.py` | `[x]` |
| `HydrateDataset` | `src/models/dataset.py` | `[x]` |
| `Trainer` — AMP, accumulation, early stopping, checkpoints | `src/models/train_loop.py` | `[x]` |
| Deep-model loss implementation | `src/models/losses.py` | `[x]` **note: final report must describe the loss actually called by `train_deep_models.py`** |
| 3-class head | `src/models/heads.py` | `[x]` |
| Profiling helper | `src/models/profile.py` | `[x]` |
| SSL stubs | `src/models/ssl_pretrain.py` / `finetune.py` | `[x]` out of scope |
| `tools/train_deep_models.py` | runner | `[x]` |
| Fake-cache integration smoke | local | `[x]` |
| **Real-cache GRU CPU smoke** | 3 folds × 2 epochs, `real_only` | `[x]` **PASS 2026-09-08** |
| **Real-cache TCN CPU smoke** | 3 folds × 2 epochs, `real_only` | `[x]` **PASS 2026-09-08** |
| CUDA/A100 smoke | Academy GPU env | `[ ]` |
| Full GRU — `real_only` | GPU | `[ ]` |
| Full GRU — `real_plus_sim` | GPU | `[ ]` |
| Full TCN — `real_only` | GPU | `[ ]` |
| Full TCN — `real_plus_sim` | GPU | `[ ]` |
| Optional mask-zeroed ablation | GPU | `[ ]` optional if compute/time permits |
| Related Work | `report/report.tex` | `[~]` draft exists locally; review/commit separately |
| Architecture descriptions | `report/report.tex` | `[~]` draft exists locally; review/commit separately |
| Deep model results | `report/report.tex` | `[!]` waits for full GPU runs + final evaluation |
| Slide 6 review | `presentation/final_slides.md` | `[x]` slide exists; M4 review remains |

### M4 CPU smoke commands completed

GRU:

```powershell
python -m tools.train_deep_models `
  --cache data/cache `
  --models gru `
  --conditions real_only `
  --n-splits 3 `
  --n-repeats 1 `
  --max-epochs 2 `
  --patience 2 `
  --batch-size 32 `
  --num-workers 0 `
  --device cpu `
  --no-amp `
  --out-results results/smoke_results.csv `
  --outputs-dir results/smoke_outputs `
  --checkpoint-root checkpoints/smoke
```

TCN:

```powershell
python -m tools.train_deep_models `
  --cache data/cache `
  --models tcn `
  --conditions real_only `
  --n-splits 3 `
  --n-repeats 1 `
  --max-epochs 2 `
  --patience 2 `
  --batch-size 32 `
  --num-workers 0 `
  --device cpu `
  --no-amp `
  --out-results results/smoke_tcn_results.csv `
  --outputs-dir results/smoke_tcn_outputs `
  --checkpoint-root checkpoints/smoke_tcn
```

Both completed all 3 folds and wrote diagnostic CSV + validation/test probability files.

---

## Member 5 — Evaluation Pipeline & Integration

**Owns:** `src/eval/`, threshold selection, lead-time figures, report assembly, slides 7–9 and 11–12, `run_all.sh`, AI disclosure.

| Task | File | Status |
|---|---|---|
| `alarm.py` — causal smoothing / alarm timing / lead time | `src/eval/alarm.py` | `[x]` |
| `metrics.py` | `src/eval/metrics.py` | `[x]` |
| `thresholds.py` — validation-only operating-point selection | `src/eval/thresholds.py` | `[x]` |
| Adaptive threshold grid fix | `src/eval/thresholds.py` | `[x]` |
| `aggregate.py` | `src/eval/aggregate.py` | `[x]` |
| Lead-time/FAR plot | `src/eval/plots.py` | `[x]` |
| Per-well lead-time plot | `src/eval/plots.py` | `[x]` |
| Report structure / eval protocol | `report/report.tex` | `[x]` structure exists |
| **`run_all.sh` end-to-end integration** | `run_all.sh` | `[x]` |
| Wire final saved `_val.npz` / `_test.npz` outputs into final threshold + test-evaluation stage | `src/eval/evaluate_predictions.py` | `[x]` |
| Final tables | `report/tables/` | `[!]` waits for final model results |
| Result slides | `presentation/final_slides.md` | `[!]` waits for final model results |
| Tag `v1.0-final` | git | `[ ]` last step after final reproduction |

---

## What Can Be Done RIGHT NOW

| Who | Action |
|---|---|
| **M1** | Run real dataset figures / sensitivity analysis from the now-available cache |
| **M2** | Document real fold report in paper Experimental Setup; no further split redesign |
| **M3** | Run/verify final baseline training on the frozen real cache, but defer test evaluation until S3 Freeze |
| **M4** | No further local model-code changes; execute CUDA smoke and full GPU training |
| **M5** | Evaluation runner and `run_all.sh` integration complete; awaiting GPU run outputs to generate final tables/figures and finalize paper abstract/conclusion |
| **All** | Freeze final hyperparameters, split seed/settings, threshold/smoothing policy before test evaluation |

---

## GPU Environment Sequence

1. Connect to the Academy VPN / JupyterLab workspace.
2. Clone/pull the current repository.
3. Make the validated `data/cache/` available in the workspace.
4. Verify the cache produces the same fold report.
5. Check CUDA/A100:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
   ```
6. Run 1–2 epoch CUDA smoke for GRU and TCN.
7. Run the full frozen experiments:
   - GRU `real_only`
   - GRU `real_plus_sim`
   - TCN `real_only`
   - TCN `real_plus_sim`
   - final XGBoost run if not already finalized on the exact same cache/splits
8. Select threshold/smoothing on validation only.
9. Freeze S3.
10. Evaluate test once and generate final tables/figures.
11. Complete report/slides/contribution report.
12. Run clean reproduction and tag `v1.0-final`.

---

## Files / Artifacts That Must NOT Be Committed

Do not commit:

- `data/3W/`
- `data/cache/`
- `data/cache_smoke/`
- smoke checkpoints
- smoke `results/*.csv`
- smoke `results/*outputs/`
- WireGuard `.conf`
- VPN private keys
- JupyterLab access tokens
- `.env` / credential files

The Git repository should contain the code and exact reproduction commands, not raw/generated data or secrets.

---

## Test Suite Summary

Current project test suite:

```text
177 passed, 0 failed  (2026-09-09)
```

Root `pytest.ini` is configured with `testpaths = tests` and `pythonpath = .`. All tests pass cleanly without requiring extra flags.

The real cache build, real split report, GRU CPU smoke, and TCN CPU smoke on 2026-09-08 all completed successfully.
