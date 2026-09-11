# Early Warning for Hydrate Formation — 3W Dataset
**Team Enigma** · AI Academy Baku · DLE-AI-202, Cohort I 2026 · Track 2

Deep learning early-warning system for hydrate formation in offshore well
service lines using **3W Dataset 2.0.0, Event 9**.

### Headline Results (v1.0-final)

| Model | Condition | Event Recall | Lead Time | FAR |
|---|---|---|---|---|
| **XGBoost** | **real\_only** | **23.3% ± 25.2%** | **43.4 min** | 7.9%/h |
| TCN | real\_only | 8.3% ± 14.4% | 89.3 min (1 fold) | 3.2%/h |
| GRU | real\_only | 0.0% | — | 3.7%/h |
| XGBoost | real\_plus\_sim | 16.7% ± 28.9% | 209.9 min (1 fold) | 1.8%/h |

> **XGBoost trained on real data only is the recommended early-warning model** at the target FAR of ≤ 1 alarm per 100 operating hours.

**One-command reproduction:**
```bash
bash run_all.sh
```
Outputs: `results/summary.csv`, `results/tables/*.tex`, `figures/*.png`.

**Stretch goal (not attempted):** self-supervised pretraining — deferred due to timeline; see Discussion in the report.

### Team Members (Team Enigma)

| Member | Name | Email | Primary Responsibilities |
|---|---|---|---|
| **M1** | Aisel Mamedova | `aysel.mamedova25@aiacademy.az` | Data quality & availability analysis, sensor trace figure, Sec.~I & III |
| **M2** | Shamistan Huseynov | `semistan.huseynov25@aiacademy.az` | Two-population grouped CV design, fold report, Sec.~V |
| **M3** | Rahima Karimova | `rahime.karimova25@aiacademy.az` | 95-feature extractor, XGBoost baseline, calibration, ablation, Sec.~IV-A & VI-A |
| **M4** | Gulnur Mammadova | `gulnur.mammedova25@aiacademy.edu.az` | TCN & GRU architectures, training loop, A100 GPU experiments, Sec.~II & VI-B |
| **M5** | Saida Arabova | `saida.erebova25@aiacademy.az` | Evaluation pipeline, alarm logic, `run_all.sh`, synthesis, Abstract, IV-C, VI-C, VII, VIII |

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
| Evaluation primitives | ✅ complete |
| `run_all.sh` end-to-end reproduction | ✅ complete & verified |
| CUDA/A100 GPU execution | ✅ complete (12 runs on A100 GPU) |
| Full TCN/GRU real experiments | ✅ complete |
| Final test evaluation (matched FAR) | ✅ complete |
| Final deliverables (Report, Slides, Contribution Report) | ✅ complete |

---

## Project tests

Run only this repository's tests:

```powershell
pytest tests -q
```

Current result:

```text
177 passed, 1 skipped, 2 warnings
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

## GPU execution workflow (completed)

The full experiments were completed following this protocol on the Academy A100 GPU (80 GB):

1. Cloned the frozen repository state.
2. Verified `data/cache/` was identical and fold report matched.
3. Verified CUDA / PyTorch environment on the NVIDIA A100-SXM4-80GB.
4. Ran deep models in FP32 (`--no-amp`) to guarantee numerical stability.
5. Trained all 12 deep model configurations:
   - GRU `real_only` (3 folds)
   - GRU `real_plus_sim` (3 folds)
   - TCN `real_only` (3 folds)
   - TCN `real_plus_sim` (3 folds)
6. Trained XGBoost baseline across all 3 folds on the identical splits.
7. Selected thresholds and alarm persistence on validation data only.
8. Froze policies and evaluated the test sets once.
9. Generated publication figures and summary tables.
10. Assembled the final IEEE report, presentation slides, and contribution report.

---

## `run_all.sh`

`run_all.sh` is the single-command reproduction script that executes the complete end-to-end pipeline:

1. **Cache Verification**: Ensures the frozen 5-channel cache exists (or builds it if absent).
2. **Split Verification**: Generates and checks the 3-fold leak-free grouped split.
3. **Model Execution**: Runs the XGBoost baseline and deep models (TCN and GRU) across folds.
4. **Validation-Only Tuning**: Performs causal alarm threshold search and smoothing on validation sets only.
5. **Frozen Test Evaluation**: Applies the tuned thresholds to test folds under the matched FAR budget ($\le 1.0\%$/h).
6. **Artifact Generation**: Produces the final results summary (`results/summary.csv`), LaTeX tables (`results/tables/*.tex`), and publication figures (`figures/*.png`).

Run full reproduction:
```bash
bash run_all.sh
```

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

---

## Submission Checklist

### GitHub
- [ ] All code committed and pushed to `origin/m5-eval-pipeline-and-report`
- [ ] Branch merged into `main`
- [ ] Tag `v1.0-final` created and pushed (`git tag v1.0-final && git push origin v1.0-final`)
- [ ] `contribution_report.tex` (compile to `contribution_report.pdf` in repository root)
- [ ] `README.md` up to date (this file)
- [ ] All 177 tests passing (`pytest -q --tb=no`)

### Report
- [ ] `report/report.tex` compiled to `report/report.pdf` (run `pdflatex` twice)
- [ ] No `\placeholder{}` commands remaining in the PDF
- [ ] Abstract ends with GitHub URL tagged `v1.0-final`
- [ ] All author names ordered from Member 1 to Member 5
- [ ] All team member emails correct

### Slides
- [ ] `report/slides.tex` compiled to `presentation/presentation.pdf`
- [ ] Slides reviewed by all members

### Moodle (ONE member submits)
- [ ] `report/report.pdf` uploaded
- [ ] `presentation/presentation.pdf` uploaded
- [ ] Other members do NOT submit duplicates
