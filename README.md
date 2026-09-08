# Early Warning for Hydrate Formation — 3W Dataset

Deep learning early-warning system for hydrate formation in offshore well
service lines using **3W Dataset 2.0.0, Event 9**.

**Result 1 (required):** XGBoost vs. TCN vs. GRU at a matched false-alarm
budget, with `real_only` and `real_plus_sim` training conditions.

**Stretch goal:** self-supervised pretraining vs. random initialization, only
after Result 1 is complete. SSL remains out of scope for the current frozen
submission path unless time remains after the required experiments.

## Read before changing the pipeline

- `DATA_FINDINGS.md` — measured properties of the real Event-9 data and the
  rationale for the final labeling, channel, normalization, and window choices.
- `TEAM_5_MEMBERS.md` — current 5-member ownership and hand-off plan.
- `team_responsibilities_all_members.md` — original shared contract; still
  useful for the binding interfaces.
- `DL_Project_Statement_Hydrate3W.docx` — project specification; the Addendum
  remains the source of truth where it overrides the main body.
- `project_status.md` — current completion state, frozen real-cache counts,
  fold report, CPU smoke status, and remaining GPU/integration work.

The real data changed several parts of the original plan:

- only **14** real Event-9 instances contain a transient phase;
- only **3** reach blockage;
- `failure_time` therefore means **transient onset**;
- hydrate wells and Normal wells are disjoint;
- per-instance normalization is required;
- the 1 Hz signals are decimated by 30 and the model window is
  **60 samples = 30 minutes**, not 60 seconds.

---

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / JupyterLab terminal

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the Academy GPU environment already provides a CUDA-enabled PyTorch build,
verify it before reinstalling anything.

---

## Get the real 3W data

The real dataset is downloaded locally under `data/3W/` and is intentionally
ignored by Git.

Windows-compatible download used successfully in this project:

```powershell
python -c "from src.data.download import download; download('data/3W')"
```

Verify the dataset:

```powershell
python -m src.data.inventory `
  --root data/3W/dataset `
  --event 9 `
  --verify
```

The inventory should detect real Normal-operation instances and both real and
simulated Event-9 instances.

Fake data remains available for fast regression/integration tests:

```powershell
python src/data/make_fake_data.py --out data/fake/ --n_instances 40
```

`data/fake/` is not a substitute for the real cache; it exists only for
development/testing.

---

## Frozen primary channel set

The final main experiment uses exactly **5 channels**:

```text
P-MON-CKP
P-JUS-CKGL
T-TPT
T-JUS-CKP
P-ANULAR
```

Do **not** use the cache builder's automatic channel selection for the final
experiment. A smoke build with the automatic selector produced a different
6-channel set, so the final cache is built with the 5-channel list explicitly.

---

## Build the real cache

The cache builder is:

```text
src/data/build_cache.py
```

The generated real cache is:

```text
data/cache/
```

Build the canonical cache with the frozen 5-channel set:

```powershell
python -m src.data.build_cache `
  --root data/3W/dataset `
  --out data/cache `
  --channels P-MON-CKP,P-JUS-CKGL,T-TPT,T-JUS-CKP,P-ANULAR
```

Successful build on 2026-09-08:

| Source | Windows | Transient | Established |
|---|---:|---:|---:|
| real | 42,259 | 1,295 | 118 |
| simulated | 43,829 | 18,422 | 21,139 |
| **total** | **86,088** |  |  |

Additional build summary:

- 801 instances written
- 48,882 windows dropped by the existing NaN-label / validity rules
- 287 instances produced zero usable windows
- final deep-model smoke runs confirmed `C=5`, `W=60`

The long list of "0 usable windows" warnings during the build is expected for
recordings that fail the frozen validity rules; the build itself completed
successfully.

---

## Frozen grouped cross-validation

Generate / verify the frozen real folds with:

```powershell
python -m src.data.splits `
  --cache data/cache `
  --n-splits 3 `
  --n-repeats 1 `
  --val-mode nested
```

Validated real fold report:

| Fold | Train positive wells | Val positive wells | Test positive wells | Val positive events | Test positive events | Val Normal h | Test Normal h |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 1 | 1 | 1 | 5 | 530.7 | 509.2 |
| 1 | 2 | 1 | 4 | 3 | 5 | 305.2 | 561.8 |
| 2 | 4 | 1 | 2 | 5 | 4 | 509.2 | 530.7 |

The test folds contain all 14 positive events in total (`5 + 5 + 4 = 14`).
All validation and test folds clear the 300 Normal-hour floor.

**Important limitation:** fold 0 contains only one positive validation event.
Validation PR-AUC and early-stopping decisions for that fold are therefore
intrinsically unstable. This is a reporting limitation, not a reason to
redesign the folds after seeing test results.

The split configuration is now frozen. Do not change folds after final test
performance is observed.

---

## Data & model contract

- `X`, `mask`: `float32` / `uint8`, shape `[N, C, W]`, channels-first.
- Shared constants live in `src/contract.py`.
- `XGBoostBaseline` exposes estimator-style
  `fit(X, mask, y, groups)` / `predict_proba(X, mask)`.
- TCN and GRU expose `forward(x, mask)` and are trained through
  `src/models/train_loop.py`.
- Final primary input shape is `C=5`, `W=60`.
- `results/results.csv` schema:
  `model, fold, seed, condition, metric_name, value`.
- Required conditions:
  `real_only`, `real_plus_sim`.
- `failure_time` = transient onset.
- `blockage_time` = Established onset, retained as secondary metadata.
- `nan_label_policy="drop"` is the default.
- window labels are
  `NORMAL`, `TRANSIENT`, `ESTABLISHED`.
- alarm smoothing is causal / trailing only.

### Mask caveat

The deep models concatenate the presence mask with the sensor values. This is
deliberate because it distinguishes an observed value from a missing one.
However, sensor availability can also encode well identity. The XGBoost
analysis found evidence of this shortcut, so mask-aware deep-model results
should be interpreted cautiously. A mask-zeroed deep ablation is optional if
GPU time permits.

---

## Repository layout

```text
src/
|-- contract.py
|-- data/              # inventory, windowing, cache, splits, fake data
|-- baselines/         # XGBoost baseline, features, tuning, calibration
|-- models/            # TCN, GRU, dataset, training loop, profiling
`-- eval/              # metrics, alarms, thresholds, aggregation, plots

tools/
|-- train_xgb.py
|-- train_deep_models.py
`-- ...

run_all.sh              # end-to-end integration entry point
results/                # generated outputs; not hand-edited
report/
presentation/
```

The current 5-member ownership is defined in `TEAM_5_MEMBERS.md`; older
4-member comments in legacy files should not be treated as the current
responsibility map.

---

## Current implementation status

| Area | Status |
|---|---|
| Real 3W download + inventory | ✅ complete |
| Frozen 5-channel real cache | ✅ complete |
| Real grouped split + fold report | ✅ complete |
| `train_positive_wells >= 2` guard | ✅ complete |
| XGBoost implementation | ✅ complete |
| TCN implementation | ✅ complete |
| GRU implementation | ✅ complete |
| Deep training loop | ✅ complete |
| Real-cache GRU CPU smoke | ✅ passed |
| Real-cache TCN CPU smoke | ✅ passed |
| Evaluation primitives | ✅ implemented |
| `run_all.sh` final integration | ⚠️ must be reviewed / brought fully in sync before GPU |
| CUDA/A100 smoke | ⏳ pending |
| Full TCN/GRU real experiments | ⏳ pending |
| Final test evaluation | ⏳ pending until S3 Freeze |
| Final report/slides numbers | ⏳ pending final results |

---

## Project tests

Run only this repository's tests:

```powershell
pytest tests -q
```

Current result:

```text
150 passed, 1 skipped, 2 warnings
```

The 2 warnings are harmless seaborn `PendingDeprecationWarning`s from
`tests/test_eval_plots.py`.

After downloading the 3W repository, bare:

```powershell
pytest -q
```

also discovers `data/3W/tests`, which depends on the external
`ThreeWToolkit` package and is not part of this project's test suite.

Recommended permanent root `pytest.ini` setting:

```ini
[pytest]
testpaths = tests
```

---

## CPU smoke tests completed on the real cache

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

Both completed all 3 folds and wrote validation/test probability files.

These are integration checks only. Do not report their metrics as final
results.

---

## Academy GPU environment

The Academy workspace is accessed through the provided WireGuard VPN and
JupyterLab URL.

Never commit:

- WireGuard `.conf`
- VPN private keys
- JupyterLab token
- endpoint credentials
- `.env` credential files

In JupyterLab, open a Terminal and first verify the GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Do not reinstall PyTorch before checking whether a compatible CUDA build is
already present.

The validated real cache must also be available in the GPU workspace. After
copying/uploading it, rerun the split command and verify the fold report is
identical before training.

---

## GPU execution order

1. Pull/clone the final pre-GPU repository state.
2. Make `data/cache/` available in the workspace.
3. Verify the real fold report is identical.
4. Check CUDA/A100.
5. Run a 1–2 epoch CUDA smoke for GRU.
6. Run a 1–2 epoch CUDA smoke for TCN.
7. Run full frozen experiments:
   - GRU `real_only`
   - GRU `real_plus_sim`
   - TCN `real_only`
   - TCN `real_plus_sim`
8. Run/finalize XGBoost on the exact same frozen cache/splits.
9. Select threshold/smoothing on validation only.
10. Freeze S3.
11. Evaluate the test set once.
12. Generate final tables/figures.
13. Complete report, slides, and contribution report.
14. Run clean reproduction and tag `v1.0-final`.

---

## `run_all.sh`

`run_all.sh` is the intended single-entry reproduction path, but it must be
reviewed before the GPU window to ensure that:

- it no longer contains stale "waiting for M2/M4" comments;
- it uses the frozen 5-channel cache;
- it uses the frozen 3-fold split;
- it invokes the current XGBoost and deep-model runners;
- it collects saved validation/test probability files;
- threshold and smoothing selection use validation only;
- the frozen policy is then applied unchanged to test;
- final tables and figures are generated from recorded results rather than
  manually typed numbers.

Do not rely on a stale `run_all.sh` for the final GPU run.

---

## Generated artifacts that must not be committed

Keep these out of Git:

```text
data/3W/
data/cache/
data/cache_smoke/
data/fake/
checkpoints/smoke/
checkpoints/smoke_tcn/
results/smoke_results.csv
results/smoke_tcn_results.csv
results/smoke_outputs/
results/smoke_tcn_outputs/
```

The repository should contain code and exact reproduction commands, not the raw
dataset, generated cache, smoke outputs, or credentials.

---

## Dataset & licensing

3W Dataset 2.0.0 (Petrobras). Dataset files are distributed under CC BY 4.0;
toolkit code is under Apache 2.0.

Source repository: `https://github.com/petrobras/3W`

Use the formal references already maintained in `report/report.tex`.

---

## Tools & Acknowledgements

Before submission, disclose substantial AI assistance in the report according
to the course brief and team policy. All generated code/text must be reviewed
and validated by the team.
