# M3 findings — things the rest of the team needs to know

Member 3 (XGBoost baseline). Measured on the real cache built from the full
3W download, not assumed. Every number regenerates from:

```bash
python -m src.data.build_cache --root data/3W/dataset --out data/cache \
    --channels P-MON-CKP,P-JUS-CKGL,T-TPT,T-JUS-CKP,P-ANULAR
python -m tools.train_xgb            --cache data/cache      # results/results.csv
python -m tools.ablate_features      --cache data/cache      # results/ablation_features.csv
python -m tools.probe_presence_shortcut --cache data/cache   # results/probe_presence_shortcut.csv
python -m tools.analyse_ablation                             # every stat quoted below
```

The full write-up is in `report/results_xgboost.md`. This file is only the
part that changes what **other members** should do.

---

## 1. The cache confirms the channel decision (for M1)

The 5-channel main arm was built and measured: **14/14 transient events and
3/3 blockage events survive**, exactly as `DATA_FINDINGS.md` §9 predicted
from the instance-level scan. 364 of 651 real instances yield usable
windows; 287 produce zero (all of WELL-00002 and WELL-00008, as expected).

Real cache: **42,259 real windows**, 1,295 Transient + 118 Established =
**3.34% positive**. Simulated: 43,829 windows, **90.3% positive**.

---

## 2. PR-AUC is not comparable across folds — this affects everyone (for M5)

**This is the single most important methodological point I found.**

Each fold's validation split has a very different positive base rate, and
PR-AUC rises mechanically with the base rate:

| fold | validation windows | positive windows | base rate | XGB PR-AUC | lift |
|---|---|---|---|---|---|
| 0 | 11,260 | **1** | 0.0001 | 0.0005 | 3.5× |
| 1 | 7,327 | 463 | 0.0632 | 0.7755 | 12.3× |
| 2 | 11,592 | 638 | 0.0550 | 0.0751 | 1.4× |

Fold 0's validation set has **one positive window**. Its PR-AUC is that
row's reciprocal rank, not a performance estimate.

I nearly published a wrong conclusion from this. On raw PR-AUC, the number
of positive validation events correlates strongly with score
(Spearman ρ = 0.638, p = 0.004) — which looks like "sparse folds can't
measure anything". After base-rate correction the correlation **vanishes**
(ρ = −0.14, p = 0.58). It was an artefact.

**Action for M5:** the head-to-head table must not average raw PR-AUC across
these folds, or compare models on it fold-by-fold, without reporting each
fold's base rate. Lift over base rate, or per-fold reporting with base rates
attached, is the honest form.

---

## 3. Fold 1 cannot be tuned at all (for M2)

In fold 1's training split, **every positive window belongs to a single
well**. Grouped inner CV therefore cannot produce a split with positives on
both sides — all 3 inner splits are unusable, and that fold falls back to a
default configuration (`tuning_skipped=1` in `results/results.csv`).

Fold 0's validation has 1 positive window; two of the nine (split-seed,
fold) cells I tested have the same problem.

**Action for M2:** this is input to the `n_splits` decision. With 7 positive
wells, 3 folds leaves training folds whose positives sit in one well. Worth
checking whether leave-one-well-out gives better-conditioned training folds,
and worth adding a `train_positive_wells` column to `fold_report()` — the
existing `n_val_positive_events` guard does not catch this, because the
problem is on the *training* side.

Related: XGBoost does **not** raise when fitted on a single-class training
split. It sets `n_classes_=1`, ignores `num_class=3`, and `predict_proba`
returns a transposed `(3, 2n)` array that fails much later with an opaque
length mismatch. `src/baselines/tune.py` now guards this and
`tests/test_baselines.py` pins the upstream behaviour.

---

## 4. The model partly reads instrumentation, not physics (for M4 and M5)

**This is the finding most likely to affect the paper's headline claim.**

Gain importance put **40.8% of all gain on the 5 presence-fraction columns**
(out of 95). So I trained on presence features **only** — no sensor values
at all — over 9 fold-cells:

| arm | features | mean validation PR-AUC |
|---|---|---|
| all features | 95 | 0.233 |
| values only (presence removed) | 90 | 0.198 |
| **presence only** | **5** | **0.172** |

A model that never sees a pressure or temperature reading reaches **74% of
full performance**, and in **2 of 9 cells it beats all 90 value features**.

Why: missingness here is *well-specific* (`DATA_FINDINGS.md` §8) and the
hydrate wells and Normal wells are **disjoint populations** (§2). So "which
sensors exist" is a proxy for well identity, which is nearly the label.
Per-instance normalisation removes offset and scale — it **cannot** remove
*"this channel is absent"*, because that lives in the mask, not the values.

Consistently, permutation importance does not reproduce the physical
signature and does not reproduce itself across folds (Kendall τ = +0.20,
+0.40, **−0.40**; all p > 0.48). In the only fold with real signal the top
channel is **P-ANULAR** — the annulus, which does not communicate with the
service line — while **P-MON-CKP**, the channel the restriction mechanism
predicts, ranks **last** (mean share 0.002).

**Action for M4:** the TCN and GRU receive the mask as input, so they can
read the same shortcut. Worth one run with the mask channel zeroed or
removed to see how much of the deep models' performance survives.

**Action for M5:** consider reporting the `values_only` arm alongside the
headline so a reader can see how much of the number survives without the
shortcut. This belongs in Limitations either way.

---

## 5. Feature design does not matter at this sample size (for everyone)

9 feature-extractor variants over 18 paired cells, validation only. Fold
composition accounts for ~**47×** more variance than the entire
feature-design space (448× on the uncorrected scale). No arm survives a
Bonferroni correction; only the presence-fraction block even reaches
nominal significance (1.175× lift, 7/10 wins, p = 0.027) — and §4 explains
why it helps, which makes it a liability rather than a win.

**Nobody should spend the remaining time tuning features.** The measurement
apparatus cannot resolve the difference.

---

## 6. Simulated data did not help the baseline

Paired difference (`real_plus_sim − real_only`) = **−0.160 ± 0.325**,
negative in 2 of 3 folds. Simulated windows are 90.3% positive against a
3.34% real rate and roughly triple the training set, pulling the model into
a regime the real wells never occupy. `real_only` stands as the headline
condition — with the honest caveat that n = 3 folds cannot establish this,
only fail to contradict it.

---

## 7. Calibration works (for M5)

Platt scaling fit on validation, applied frozen to test:

| condition | ECE before → after | Brier before → after |
|---|---|---|
| real_only | 0.0409 → **0.0106** | 0.0371 → **0.0264** |
| real_plus_sim | 0.0538 → **0.0121** | 0.0490 → **0.0350** |

Improves in **all 6** fold × condition cells (74–78% ECE reduction).

`results/model_outputs/*.npz` carries both raw and calibrated arrays:
`probs` (raw 3-class, byte-identical in contract to M4's files),
`probs_calibrated`, `pos_score_raw`, `pos_score_calibrated`.

### 7a. Calibration compresses the score range — `select_threshold`'s fixed grid breaks on it

Calibration is monotone, so it cannot change *which* alarm sets are
achievable. But `select_threshold()` sweeps `np.linspace(0, 1, 200)`, and
after calibration the scores no longer span [0, 1]:

| fold | calibrated score range | sweep thresholds inside that range (of 200) |
|---|---|---|
| 0 | [0.0000, **0.0021**] | **0** |
| 1 | [0.0007, 1.0000] | 198 |
| 2 | [0.0382, **0.1673**] | 26 |

In fold 0 the selected threshold comes back as **0.0050, which is above the
entire score range** — the model can then never fire, on validation or test.
Fold 2 has 26 usable grid points instead of 200, so the FAR/lead-time
trade-off is sampled very coarsely.

**Suggested fix (M5's call):** sweep over score **quantiles** rather than a
fixed 0–1 grid, or sweep on `pos_score_raw`. Because Platt scaling is
monotone the two give identical alarm decisions where the grid can resolve
them — I verified raw and calibrated produce the same alarm outcome in all
three folds — so this is purely about the sweep having resolution where the
scores actually live.

### 7b. Indicative event recall at the target FAR is low

Running `select_threshold` + `alarm_times` on my validation outputs at
1 false alarm / 100 h (with `smooth_window=5`, `min_duration=0`, which are
**M5's parameters to choose**, so treat this as indicative only):

| fold | positive validation instances alarmed |
|---|---|
| 0 | 0 / 1 |
| 1 | 2 / 3 |
| 2 | 0 / 5 |
| **total** | **2 / 9** |

That is the operationally meaningful number and it is far more informative
than PR-AUC: at the project's false-alarm budget the baseline catches about
a fifth of the events on validation. M5 should regenerate this properly once
smoothing and `min_duration` are selected.

---

## 8. Test set is untouched

`tools/train_xgb.py` writes test **probabilities** every run (M5 needs them)
but computes **no test metrics** unless `--eval-test` is passed. Run that
flag only after the S3 freeze.
