<!--
Member 3 draft for report.tex's Method section IV-A (Baseline Feature
Engineering) and the baseline half of IV-C. Markdown for now, following the
convention M1 set in introduction.md / dataset.md.

Every number below is reproducible:
  cache statistics      data/cache/cache_summary.csv, written by
                        `python -m src.data.build_cache --root data/3W/dataset
                         --out data/cache --channels P-MON-CKP,P-JUS-CKGL,T-TPT,T-JUS-CKP,P-ANULAR`
  ablation numbers      results/ablation_features.csv, written by
                        `python -m tools.ablate_features --cache data/cache`
  baseline results      results/results.csv, written by
                        `python -m tools.train_xgb --cache data/cache`
None is typed by hand (team contract §0.3).
-->

# Method — XGBoost baseline

## Why the baseline is treated as a real model

Track 2 does not waive the baseline requirement, and a weak baseline would
make the deep-model comparison meaningless: if the TCN beats an XGBoost that
was never tuned, the result says nothing about temporal convolution. The
baseline therefore gets the same fold structure, the same class-imbalance
treatment, the same calibration step and a comparable tuning budget as the
deep models, and its design decisions are ablated rather than asserted.

## Input contract

The baseline consumes the same cache the deep models do: channels-first
`float32 [N, C, W]` windows with an aligned `uint8 [N, C, W]` presence mask,
where `mask == 1` marks a genuinely observed sample. With the main-arm
channel set (`P-MON-CKP, P-JUS-CKGL, T-TPT, T-JUS-CKP, P-ANULAR`), `C = 5`
and `W = 60` decimated samples, one sample per 30 s, so each window spans
**30 minutes** of real time with a stride of 150 s.

On the built cache this yields **42,259 real windows** from 364 usable real
instances, of which **1,295 are Transient and 118 Established** — a positive
rate of **3.34%**. The 14 transient events and all 3 blockage events survive
the channel choice, confirming on the built cache what `DATA_FINDINGS.md` §9
predicted from the instance-level trade-off scan.

## Multi-timescale, mask-aware features

For each window and channel we take three nested slices measured **from the
end of the window** — the full 60 samples, the last 30, and the last 15 —
and compute six statistics on each: `mean`, `std`, `min`, `max`, `slope`
(least-squares trend) and `last_diff` (net change between the first and last
present sample). Taking slices from the end is what keeps the features
causal: a spike early in the window can only ever affect the full-window
statistics, never the shorter ones, which is verified by a unit test rather
than assumed.

Every statistic is computed over **present samples only**. The per-channel
**presence fraction** over the full window is then appended as its own
feature block, so the model is told how much of each channel it is actually
trusting rather than having to infer it. With `C = 5` this gives
`5 × 6 × 3 + 5 = 95` features.

The extractor has no fitted state: every column is a function of one window
and its own mask, with no statistic pooled across rows, no scaler and no
vocabulary. Extracting once for the whole cache is therefore bit-identical
to extracting per fold, and the runner asserts the absence of a `fit()` at
startup so the property cannot silently lapse.

## Two design decisions that were measured, not assumed

**Encoding of a dead sensor.** Three real instances carry sensors that are
frozen for the entire recording (`DATA_FINDINGS.md` §9), and one of them is
a blockage event, so this is not a corner case. When a channel observes
nothing in a slice, its six statistics are undefined. The obvious fallback —
emit `0.0` — is actively wrong here: `windowing.normalize_instance()`
standardises each instance to roughly zero mean, so `0.0` is precisely the
value a *healthy* sensor sitting at its own baseline reports. That encoding
tells the model that a dead sensor looks perfectly normal. We instead emit
`NaN` and let XGBoost learn a default branch direction at each split, which
turns "this sensor was dead" into usable evidence. The presence fraction
stays a real measurement of exactly `0.0` and is never `NaN`.

**Time axis of the slope.** Rate of change is the physical hydrate
signature, so a biased slope attacks the feature the domain argument leans
on hardest. Regressing on the *rank* of the present samples silently closes
every gap: two samples nine steps apart with a rise of nine units report a
slope of 9.0 instead of 1.0. We regress on the sample's true position inside
the slice, so a gap keeps its width and the slope stays a rate per sample.

Both legacy behaviours remain reachable (`missing_policy="zero"`,
`slope_time="rank"`) so the ablation in Section~\ref{sec:results_xgb} is a
real comparison rather than a claim, and the vectorised implementation was
verified bit-identical to the original loop under the legacy settings before
either default was changed (max absolute difference `1.19e-7`, i.e. float32
rounding), while running **20.7× faster**.

**What the ablation actually showed.** Neither change is separable from fold
noise on this cache: NaN encoding is worth `+0.037 ± 0.059` validation
PR-AUC (6 of 10 paired wins, `p = 0.28`) and the slope axis `−0.003 ± 0.012`
(`p = 0.77`). We keep both defaults anyway, but on **correctness** grounds
rather than measured gains — `0.0` states something false about a dead
sensor, and a slope regressed on rank is not a rate — and the report says so
rather than implying the numbers backed them. The one component with
consistent supporting evidence is the presence-fraction block
(`+0.066`, 7 of 10 wins, nominal `p = 0.027`). Section~\ref{sec:results_xgb}
gives the noise floor that makes all of these hard to resolve.

## Class imbalance

Positive windows are 3.34% of the real cache. Imbalance is handled with
**inverse-frequency `sample_weight`** computed from each *training fold's own*
class counts and passed to `fit()`. It is deliberately **not** handled with
`scale_pos_weight`: that parameter is binary-classification-only in XGBoost,
and under `objective="multi:softprob"` with `num_class=3` it has no defined
effect and silently does nothing.

`num_class=3` is set explicitly rather than inferred, because a training
fold can legitimately contain no Established windows — only 118 exist in the
entire real cache. Without it the model would emit two columns and
`positive_score()`, which indexes column 2, would fail or silently mis-read.
A regression test pins this.

## Hyperparameter search

Hyperparameters are searched on the **training fold only**, scored by
grouped inner cross-validation over that fold's wells, so no well appears on
both sides of an inner split. The search space is a 16-point grid over
`n_estimators`, `max_depth`, `learning_rate` and `subsample`; the number of
configurations evaluated is recorded in `results/results.csv` so the report
can show the baseline and the deep models received comparable budgets.

The scored quantity is validation **PR-AUC on
`positive_score() = P(Transient) + P(Established)`** — the same
pre-registered reduction the alarm logic and threshold selection consume.
Tuning on accuracy or on multiclass log-loss would optimise a quantity the
paper never reports. Sample weights are applied inside the search exactly as
at final fit time; tuning an unweighted model and shipping a weighted one
would select hyperparameters for a different problem.

## Calibration

Module 8 selects the alarm threshold by sweeping a probability axis until
the one-alarm-per-100-hours budget is spent, so the probabilities must mean
what they say or the threshold chosen on validation transfers to test as an
arbitrary number. Weighted training on a 3% positive rate pushes scores away
from the base rate by construction, so miscalibration is expected rather
than incidental.

A calibrator is fit on the **validation fold** and applied **unchanged** to
that fold's test wells. Platt scaling (two parameters) is the default rather
than isotonic regression: with a handful of positive *events* per validation
fold, a non-parametric isotonic fit will happily fit a step function to
noise. The runner writes raw and calibrated probabilities side by side, so
the choice remains auditable and Module 8 can consume either.

Folds whose validation split contains a single class are a real occurrence
in this dataset; there the calibrator degrades to the identity rather than
raising, because collapsing to the observed base rate would destroy the
ranking threshold selection depends on.

## Feature importance

Two measures are reported. **Gain** falls out of the fitted booster but is
computed on training data and is biased toward high-cardinality features, so
it describes what the trees leaned on rather than what generalises.
**Permutation importance** shuffles one validation column at a time and
records the drop in validation PR-AUC; it measures out-of-sample
contribution to the metric actually reported and inherits none of gain's
bias. The physical argument in the Discussion cites the permutation
ranking.

Importances are aggregated to (channel, statistic, timescale) through the
feature-name grammar the extractor emits, so a split can always be traced to
a physical sensor. The naming is generated from the same nested loop that
emits the columns, because a name list that drifts from the column order is
a silent failure: the model still trains, the figure still renders, and
every physical attribution in the paper names the wrong sensor.
