<!--
Member 3 draft for report.tex's Results section VI-A (XGBoost Baseline).
Markdown for now, following M1's convention in dataset.md.

Sources, all regenerable, none typed by hand:
  results/ablation_features.csv        python -m tools.ablate_features --cache data/cache \
                                           --seeds 42,43 --split-seeds 42,7,2024
  results/results.csv                  python -m tools.train_xgb --cache data/cache
  results/probe_presence_shortcut.csv  python -m tools.probe_presence_shortcut --cache data/cache
  results/tables/*                     from the train_xgb run
  figures/reliability_xgboost.png      from the train_xgb run
-->

# Results — XGBoost baseline

## VI-A.1 Headline: the baseline works, and the fold noise is larger than everything else

Validation PR-AUC, 3 grouped folds, seed 42. The positive-window rate over
the whole real cache is **0.0334**, but each fold's *validation* rate
differs sharply — see below, it changes how this table must be read.

| fold | positive validation events | real\_only | real\_plus\_sim |
|---|---|---|---|
| 0 | 1 | 0.0005 | 0.0003 |
| 1 | 3 | **0.7755** | 0.2420 |
| 2 | 5 | 0.0751 | 0.1288 |
| **mean ± sd** | | **0.284 ± 0.428** | **0.124 ± 0.121** |

The mean is close to meaningless on its own: the standard deviation exceeds
it. But the spread is **not** what it first looks like, and the correction is
worth stating carefully because it is easy to get wrong.

**PR-AUC is not comparable across these folds.** Each fold's validation
split has a different positive base rate, and PR-AUC rises mechanically with
the base rate. The three folds differ by more than two orders of magnitude:

| fold | validation windows | positive windows | base rate | PR-AUC | lift over base |
|---|---|---|---|---|---|
| 0 | 11,260 | **1** | 0.0001 | 0.0005 | 3.5× |
| 1 | 7,327 | 463 | 0.0632 | 0.7755 | 12.3× |
| 2 | 11,592 | 638 | 0.0550 | 0.0751 | 1.4× |

Fold 0's validation set contains **exactly one positive window** out of
11,260. A PR-AUC computed from a single positive row is not a performance
estimate at all — it is that row's reciprocal rank. Read as lift over each
fold's own base rate, the baseline is above chance in all three folds
(1.4×–12.3×) rather than "at chance in two of them", which is what the raw
column suggests. Any per-fold PR-AUC quoted from this dataset must carry its
base rate alongside it.

**Simulated data did not help.** The paired difference
(`real_plus_sim − real_only`) is **−0.160 ± 0.325**, negative in 2 of 3
folds and driven by fold 1 (−0.53). This is consistent with a distribution
argument rather than a sample-size one: the simulated windows are
**90.3% positive** (39,561 of 43,829) against a **3.34%** real positive
rate, and they inflate the training set from ~19k to ~63k rows. The model is
therefore pulled toward a regime that does not exist in the real wells it is
scored on. `real_only` remains the headline condition.

## VI-A.2 Fold composition dominates every other source of variance

The feature ablation ran 9 extractor variants over 18 paired cells (3 CV
split seeds × 3 folds × 2 model seeds), on validation folds only.
Decomposing the variance across those 162 runs, on both the raw scale and
the base-rate-corrected one (log lift):

| source of spread | var. of group means (raw PR-AUC) | var. of group means (log lift) |
|---|---|---|
| feature-design arm | 0.000149 | 0.0209 |
| fold cell (which wells land in validation) | 0.0669 | 0.988 |
| **ratio** | **448×** | **47×** |

On the corrected scale fold composition still accounts for roughly **47×**
more spread than the entire feature-design space searched. The raw-scale
figure of 448× overstates it, because part of that spread is the base-rate
artifact above rather than genuine instability.

**A relationship we checked and had to discard.** On raw PR-AUC, the number
of positive validation events correlates strongly with score
(Spearman ρ = 0.638, p = 0.0044), which invites the conclusion that folds
with few events cannot measure anything. That correlation **does not
survive base-rate correction**: on lift, ρ = −0.14 (p = 0.58). Folds with
more positive events also have higher base rates, and PR-AUC rises with the
base rate on its own. We report this because the uncorrected version is the
natural thing to compute and it is misleading — the same trap applies to any
per-fold PR-AUC comparison in this project, including the head-to-head
model table.

What remains true is that fold-to-fold variability is large on any scale,
and that two cells have a validation split containing a *single positive
window*, where no metric is estimable.

Two consequences that reach beyond this section:

* **Anything selected on validation PR-AUC in a single-positive-window fold
  is selected on noise** — including the deep models' early stopping and
  Module 8's threshold selection.
* **One training fold could not be tuned at all.** In fold 1 every positive
  window in the training split belongs to a *single well*, so grouped inner
  cross-validation cannot produce a split with positives on both sides; all
  3 inner splits were unusable and that fold fell back to the default
  configuration (recorded as `tuning_skipped=1` in `results/results.csv`).
  With 7 positive wells and 3 outer folds this is a structural property of
  the dataset, not an accident of one seed.

## VI-A.3 Feature-design ablation: honest null results

Arms were compared as **paired** differences (identical folds and seeds) and
tested with a Wilcoxon signed-rank test, restricted to the 10 cells where
validation PR-AUC is estimable at all (≥ 2 positive validation events).
Because raw PR-AUC is not comparable across folds (§VI-A.1), differences are
reported on the base-rate-corrected scale as a **lift ratio**; the raw-scale
figures are in `results/ablation_features.csv`.

| comparison | lift ratio | wins | Wilcoxon p |
|---|---|---|---|
| **presence-fraction block vs without** | **1.175×** | **7/10** | **0.027** |
| 1 timescale vs 3 | 1.027× | 4/10 | 0.92 |
| NaN vs zero encoding of a dead sensor | 1.025× | 6/10 | 0.77 |
| slope on true index vs on rank | 0.977× | 5/10 | 0.63 |
| 2 timescales vs 3 | 0.914× | 2/10 | 0.23 |

Only the presence-fraction block separates from noise, and it does so on
both scales (raw: +0.066 PR-AUC, same p = 0.027). Everything else is within
3% of parity. Even the presence result would not survive a Bonferroni
correction for the six comparisons made (threshold 0.0083), so it should be
read as suggestive rather than established — but it is the only arm where
the direction, the win count and the p-value agree, and §VI-A.4 gives an
independent mechanism for why it helps.

We therefore justify the two changed defaults on **correctness** grounds and
say so rather than implying the numbers backed them: `0.0` is not neutral
filler after per-instance normalisation but the value a *healthy* sensor at
its own baseline reports, and a slope regressed on sample rank is not a
rate. Three timescales are retained because the module contract specifies
them, not because they were shown to help — one timescale (35 features
instead of 95) performs the same within noise.

The most useful output of this ablation is the null result itself: it tells
the team not to spend the remaining days tuning features, because the
measurement apparatus cannot resolve the difference.

## VI-A.4 The most important finding: the model partly reads instrumentation, not physics

Two results pointed the same way. The ablation found the per-channel
presence-fraction block to be the only component with consistent supporting
evidence, and gain importance on fold 1 — the only fold with substantial
validation signal — placed **40.8% of the total gain on those 5 columns out
of 95**.

We tested the implication directly (`tools/probe_presence_shortcut.py`):
train on the presence-fraction columns **only**, with no sensor values of any
kind, over 9 fold-cells.

| arm | features | mean val PR-AUC | median | mean lift over base rate |
|---|---|---|---|---|
| all features | 95 | 0.233 | 0.106 | 17.4× |
| values only (presence block removed) | 90 | 0.198 | 0.077 | 30.4× |
| **presence only** | **5** | **0.172** | 0.051 | **7.7×** |

A model that cannot see a single pressure or temperature reading reaches
**74% of the full model's mean PR-AUC**. In individual cells the effect is
starker still:

| cell | base rate | presence only | values only | all features |
|---|---|---|---|---|
| split 42, fold 1 | 0.063 | 0.602 | 0.752 | 0.887 |
| split 7, fold 2 | 0.042 | **0.493** | 0.354 | 0.483 |
| split 2024, fold 0 | 0.054 | **0.284** | 0.259 | 0.290 |

In 2 of 9 cells, **sensor availability alone outperforms all 90 value
features.**

The explanation is in the dataset, not the model. `DATA_FINDINGS.md` §8
records that missingness here is *well-specific* — different wells
instrument different sensors, and some sensors are dead for an entire
instance — while §2 records that the hydrate wells and the
Normal-operation wells are **disjoint populations**. Sensor availability is
therefore a proxy for well identity, and well identity is very nearly the
label. Per-instance normalisation, which is the project's main defence
against well-identity confounding, removes each recording's offset and
scale; it cannot remove *"this channel is absent"*, because that information
lives in the mask rather than in the values.

This reframes the importance analysis. Permutation importance does **not**
reproduce the expected physical hydrate signature, and does not reproduce
itself across folds:

| channel (permutation share) | fold 0 | fold 1 | fold 2 | mean |
|---|---|---|---|---|
| P-JUS-CKGL | 0.645 | 0.408 | 0.467 | 0.507 |
| T-JUS-CKP | 0.332 | 0.000 | 0.533 | 0.289 |
| P-ANULAR | 0.023 | 0.489 | 0.000 | 0.170 |
| T-TPT | 0.000 | 0.099 | 0.000 | 0.033 |
| P-MON-CKP | 0.000 | 0.004 | 0.000 | 0.002 |

Pairwise rank agreement between folds is nil (Kendall τ = +0.20, +0.40,
−0.40; all p > 0.48). In fold 1 — the only fold with real signal — the
top-ranked channel is **P-ANULAR**, the annulus, which does not communicate
with the service line and has no mechanistic reason to lead. Meanwhile
`P-MON-CKP`, the pressure upstream of the production choke and the channel
the 3W descriptor's restriction mechanism predicts should dominate, ranks
last with a mean share of 0.002.

**We therefore do not claim that the baseline recovered the physical hydrate
signature, because it did not.** Reporting the opposite would have required
quoting a single fold and ignoring the other two. The honest reading is that
at 14 positive events across 7 disjoint wells, the model has both the
opportunity and the incentive to identify wells rather than physics, and the
importance analysis shows it partly took it.

Recommended follow-ups, in priority order, for whoever has time:

1. Re-run the headline with the presence block removed (`values_only`
   above is exactly that arm) and report both, so the reader can see how
   much of the number survives without the shortcut.
2. Check whether the deep models inherit the same shortcut — the TCN and
   GRU receive the mask as input, so they can read the same signal.
3. Restrict evaluation to a channel set instrumented in every well, at the
   cost of events (`DATA_FINDINGS.md` §9 quantifies the trade-off).

## VI-A.5 Calibration

Calibration is the one component that behaved exactly as intended, in every
fold and both conditions:

| condition | ECE before | ECE after | Brier before | Brier after |
|---|---|---|---|---|
| real\_only | 0.0409 | **0.0106** | 0.0371 | **0.0264** |
| real\_plus\_sim | 0.0538 | **0.0121** | 0.0490 | **0.0350** |

Platt scaling fit on validation reduced expected calibration error by
**74–78%** and the Brier score by **29%**, and improved both in all 6
fold × condition cells. Because Module 8 selects the alarm threshold by
sweeping a probability axis, this is a prerequisite for the threshold chosen
on validation to mean anything on test, not a cosmetic step.

`figures/reliability_xgboost.png` shows the before/after curves with a
per-bin count panel beneath. The panel is not decoration: at a 3% positive
rate the high-probability bins hold very few windows, and a reliability
curve that hides bin populations makes three unlucky windows look like a
systematic miscalibration.

## VI-A.6 Tuning budget and cost

The search evaluated the full **16-configuration** grid per fold with
grouped inner CV, and both conditions of a fold share the configuration
selected on that fold's real training wells, so `real_only` and
`real_plus_sim` differ only in training data. Selected configurations were
consistent: `max_depth=3`, `n_estimators=300` in both tunable folds, i.e.
the search preferred the shallowest and smallest models offered — the
expected outcome with ~1,400 positive windows. Fold 1 could not be tuned
(§VI-A.2) and used the default configuration.

The feature extractor was rewritten to vectorised masked arithmetic,
verified **bit-identical** to the original loop under the legacy settings
(max absolute difference 1.19e-7, i.e. float32 rounding) and measured
**20.7× faster** (2.59 s → 0.125 s for 4,000 windows). The full
86,088 × 95 feature matrix builds in **1.6 s**, which is what made a
162-run paired ablation plus a 27-run shortcut probe affordable on CPU.
