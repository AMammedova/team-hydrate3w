<!--
Member 3's content for the baseline slides (TEAM_5_MEMBERS.md §4, §6:
"Slayd: baseline + importance"). Bullet-point outline only -- M5 assembles
and formats the final deck.

Figures referenced here are produced by the pipeline, not drawn by hand:
  figures/reliability_xgboost.png      tools/train_xgb.py
  results/tables/*_importance_by_channel.csv  tools/train_xgb.py
  results/ablation_features.csv        tools/ablate_features.py
-->

# Slide A — Baseline: what XGBoost actually sees

- The baseline is **not** a strawman: same folds, same imbalance handling,
  same calibration step and a recorded tuning budget as the deep models.
  If the TCN beats an untuned XGBoost, the result says nothing.
- Input is the identical cache the deep models use: channels-first
  `[N, 5, 60]` windows + an aligned presence mask. One window = **30 min**
  of real time (60 samples at 1 per 30 s), stride 150 s.
- **95 features** = 5 channels x 6 statistics x 3 timescales + 5 presence
  fractions.
  - 3 timescales taken **from the end of the window** (full / last half /
    last quarter) -> every feature is causal by construction.
  - 6 statistics: mean, std, min, max, slope, last_diff.
  - All computed over **present samples only**; the presence fraction is
    passed as its own feature so the model is told how much it is trusting.

# Slide B — Two decisions we measured instead of assuming

- **A dead sensor must not look like a healthy one.**
  Per-instance normalisation centres each recording near zero, so encoding
  an absent channel as `0.0` tells the model "this sensor is calmly at its
  baseline" -- the worst possible reading. We emit `NaN` and let XGBoost
  learn a default branch direction. This matters because 3 real instances
  have sensors frozen for the whole recording, and one of them is a
  blockage event.
- **The slope must be a rate, not a rank.**
  Regressing on the rank of present samples silently closes every gap: two
  points 9 steps apart rising 9 units report slope 9.0 instead of 1.0.
  Rate of change is the physical hydrate signature, so we regress on true
  position.
- Both legacy behaviours are still reachable, so the ablation is a real
  A/B. The rewrite was first verified **bit-identical** to the original
  implementation (max diff 1.19e-7) and is **20.7x faster**.
- *Table: results/ablation_features.csv -- paired validation PR-AUC per arm*

# Slide C — Class imbalance and calibration

- Positive windows are **3.34%** of the real cache (1,295 Transient + 118
  Established out of 42,259).
- Imbalance handled with **inverse-frequency `sample_weight`** from each
  training fold's own counts -- **not** `scale_pos_weight`, which is
  binary-only in XGBoost and silently does nothing under
  `multi:softprob` with `num_class=3`.
- Weighted training pushes probabilities away from the base rate, so the
  raw scores are miscalibrated **by construction**, not by accident.
  That matters because Module 8 picks the alarm threshold by sweeping a
  probability axis: if 0.7 does not mean 0.7, the threshold chosen on
  validation transfers to test as an arbitrary number.
- **Platt scaling fit on validation, applied frozen to test.** Platt over
  isotonic because a validation fold holds only a handful of positive
  *events* and isotonic will fit a step function to noise.
- *Figure: `figures/reliability_xgboost.png` -- before/after, with the
  per-bin count panel underneath. At a 3% positive rate the high-probability
  bins hold very few windows, and a reliability curve that hides that is
  misleading.*

# Slide D — Results, and what the numbers really say

- Validation PR-AUC, `real_only`: fold 0 **0.0005**, fold 1 **0.776**,
  fold 2 **0.075** -> mean **0.284 +/- 0.428**. *The sd exceeds the mean.*
- **But PR-AUC is not comparable across these folds** -- each has a
  different base rate, and PR-AUC rises with it:

  | fold | positive windows | base rate | PR-AUC | lift |
  |---|---|---|---|---|
  | 0 | **1** of 11,260 | 0.0001 | 0.0005 | 3.5x |
  | 1 | 463 of 7,327 | 0.0632 | 0.7755 | 12.3x |
  | 2 | 638 of 11,592 | 0.0550 | 0.0751 | 1.4x |

  Fold 0 has **one positive window** -- its PR-AUC is that row's reciprocal
  rank, not a performance estimate. On lift the model is above chance in
  **all three** folds, not two.
- **A correlation we had to discard:** raw PR-AUC tracks the number of
  positive validation events (rho = 0.638, p = 0.004) -- but that vanishes
  after base-rate correction (rho = **-0.14**, p = 0.58). It was an
  artefact. The same trap applies to the head-to-head model table.
- **Fold composition still beats feature design by ~47x** (corrected scale;
  448x raw).
- **Simulated data did not help:** paired diff **-0.160 +/- 0.325**,
  negative in 2 of 3 folds. Simulated windows are **90.3% positive** vs a
  **3.34%** real rate, so they pull the model into a regime the real wells
  never occupy. `real_only` is the headline. *(Defence question 5.)*
- **One training fold could not be tuned at all:** in fold 1 every positive
  training window belongs to a single well, so grouped inner CV has no split
  with positives on both sides. Structural, not bad luck.
- **Calibration is the one clean win:** ECE 0.041 -> 0.011 (`real_only`),
  0.054 -> 0.012 (`real_plus_sim`); Brier -29%. Improves in **all 6**
  fold x condition cells.

# Slide E — The finding we did not want: instrumentation, not physics

- Gain importance put **40.8% of all gain on the 5 presence-fraction
  columns** (of 95). So we tested it directly.
- **Trained on presence features ONLY -- no sensor values at all:**

  | arm | features | mean val PR-AUC |
  |---|---|---|
  | all features | 95 | 0.233 |
  | values only | 90 | 0.198 |
  | **presence only** | **5** | **0.172** |

  A model that never sees a pressure or temperature reaches **74% of full
  performance**, and in **2 of 9 folds it beats all 90 value features.**
- **Why:** missingness is *well-specific* (wells instrument different
  sensors), and hydrate wells and Normal wells are **disjoint populations**.
  So "which sensors exist" is a proxy for well identity, which is nearly the
  label. Per-instance normalisation removes offset and scale -- it **cannot**
  remove *"this channel is absent"*, which lives in the mask.
- Consistently, permutation importance **does not** reproduce the physical
  signature and does not reproduce itself: rank agreement across folds is nil
  (Kendall tau = +0.20, +0.40, **-0.40**; all p > 0.48). In the only fold
  with real signal the top channel is **P-ANULAR** (the annulus, which does
  not see the service line), while **P-MON-CKP** -- the channel the
  restriction mechanism predicts -- ranks **last** (mean share 0.002).
- **So we do not claim the baseline recovered the hydrate signature.**
  Claiming it would have meant quoting one fold and ignoring two.
- Recommended: report `values_only` alongside the headline; check whether
  the TCN/GRU inherit the same shortcut (they receive the mask too).

# Defence answers this slide set owns

- **"Why isn't accuracy the headline?"** A model predicting Normal for
  every window scores 96.7% accuracy on this cache and never fires an
  alarm. PR-AUC on `P(Transient)+P(Established)`, event recall and
  false alarms per operating **hour** are the metrics that can distinguish
  it from a useful model.
- **"Why `P(Transient)+P(Established)` and not `P(Transient)` alone?"**
  Only 3 real instances ever reach the Established phase. Dropping that
  mass would discard the most severe evidence the model has, and an alarm
  is equally warranted in either positive state.
