# Team Responsibilities — All Four Members

**Project:** Early Warning of Hydrate Formation in Offshore Well Service Lines (Option A, Track 2)
**Course:** DLE-AI-202, Cohort I 2026

> **Reconciled with `DL_Project_Statement_Hydrate3W.docx` (see its Addendum).** This document has now been through two rounds of correction after external review — here's the current state, in order of how much it changed:
> 1. **Window labeling now defaults to `most_severe`** (§1.3, W1.6), not `any_transient`. A real bug was found in naive `any_transient`: a window that has already progressed to Established but also contains an earlier transient sample would get called "Transient," silently reporting a *less* advanced state than the window actually has. `most_severe` (label = the most severe state present anywhere in the window) fixes this while keeping the early-flagging benefit `any_transient` was for.
> 2. **SSL pretraining is now a stretch goal, not a mandatory co-headline result** — see the rewritten §0.6. The guaranteed project is XGBoost vs. TCN vs. GRU, real-only vs. real+simulated, at a matched false-alarm budget. Pretraining is attempted only once that full pipeline is working end-to-end with populated results — this is the brief's own advice ("get the pipeline right first... add stretch goals later"), and it means the project's core claim doesn't depend on the riskiest, least-tested component succeeding on a deadline.
> 3. `src/contract.py` now exists — shared constants (`NORMAL`/`TRANSIENT`/`ESTABLISHED`, the 3W label codes, the results-schema columns) live there, imported everywhere, not redefined per-file.
> 4. A handful of process wording changes: "nobody edits another directory" softened to allow reviewed cross-edits; the commit-count-parity language softened so it doesn't incentivize meaningless micro-commits; some purely administrative ownership (README/requirements/.gitignore) is now shared rather than solely Member 4's.
>
> Nothing about who owns which directory changed beyond the addition of `src/contract.py`.

**How this document works.** Each member has one module, one directory, and one clear deliverable. The modules connect only through a data contract that is frozen on day one, which means all four of you can write and test code at the same time without waiting for each other. Nobody edits another person's directory.

**Read your own section in full, and skim the other three.** At the defense you can be asked about any part of the project, so you need to know roughly what the others are doing even though you are not writing their code.

---

# Part 0 — The shared contract (agreed by all four, before anyone writes real code)

This is the only thing that blocks parallel work. Spend two hours on it together, write it down, and then do not change it unilaterally.

## 0.1 The data contract

Put this in `src/contract.py` as documentation and as helper functions.

```python
# A cached, windowed dataset. One .npz file per source instance.
X:        float32  [N, C, W]   # N windows, C channels, W timesteps
mask:     uint8    [N, C, W]   # 1 = value present and not frozen, 0 = missing
y:        int64    [N]         # 0 = normal, 1 = transient, 2 = blocked
group:    int64    [N]         # well id — the key used for grouped splitting
inst_id:  int64    [N]         # source instance id
t_end:    float64  [N]         # timestamp of the window's last sample (seconds)
is_sim:   uint8    [N]         # 0 = real instance, 1 = simulated instance
```

Two additional arrays, one per **instance** rather than per window, saved alongside:

```python
failure_time: float64          # timestamp when the blocked (steady-state) period begins
                               # NaN if the instance never reaches blockage
normal_hours: float64          # total hours of normal-operation data in this instance
```

`failure_time` is what lead time is measured against. `normal_hours` is the denominator for false alarms per hour. Without these two, Member 4 cannot compute the headline metrics, so they are part of the contract, not an afterthought.

## 0.2 The model contract

Every model, including the XGBoost baseline, exposes exactly this:

```python
class Model:
    def fit(self, X, mask, y, groups) -> None: ...
    def predict_proba(self, X, mask) -> np.ndarray:  # float32 [N, 3], rows sum to 1
```

Member 4's evaluation code only ever sees the output of `predict_proba`. It never knows or cares whether a TCN or a decision tree produced it.

**One clarification, so nobody reads this as contradicting Part 3:** `XGBoostBaseline` satisfies this directly. The PyTorch models (`TCN`, `GRUClassifier`) do **not** implement `fit`/`predict_proba` themselves — they expose `forward(x, mask)`, the standard `nn.Module` interface, and `Trainer` (Member 3, W3.6) is the adapter that wraps a PyTorch model into something exposing `fit`/`predict_proba` for Member 4 to call. So: **classical estimators implement this contract directly; PyTorch models + Trainer implement it together.** Nobody should try to make `TCN.fit()` a real method — that's what `Trainer` is for.

Shared constants for all of the above (the 3-class labels, the 3W raw label codes, the results-schema columns) live in `src/contract.py`, imported everywhere — nobody redefines `NORMAL = 0` locally in their own file. Whoever notices a missing constant adds it there and tells the group, rather than inventing a local one.

## 0.3 The results contract

Every experiment appends rows to `results/results.csv` with exactly these columns:

```
model, fold, seed, condition, metric_name, value
```

Where `condition` is one of `pretrained`, `random_init`, `real_only`, `real_plus_sim`, `no_mask`, `mask_ratio_0.15`, and so on. Every table and figure in the paper is generated from this file by a script. **No number in the paper is ever typed by hand.**

## 0.4 The fake data generator

Member 1 writes `src/data/make_fake_data.py` in the **first hour of the project**. It produces synthetic arrays that match the contract exactly: correct shapes, correct dtypes, a handful of fake wells, and a planted "transient" signal (for example, a slow ramp in two channels before the label flips to 2).

This is the single most important unblocking artefact in the project. Members 2, 3 and 4 develop entirely against fake data until the real cache is ready. When it lands, they change one path and rerun.

## 0.5 Git workflow

- One primary owner per directory — that person is who you check with and who reviews changes there. It's not a lock: if you find a real problem in someone else's module (a wrong constant, a units mismatch), fix it or pair with the owner rather than working around it in your own code. This project is coupled enough that rigid ownership walls would slow everyone down for no real safety benefit.
- Branch per feature, small commits, descriptive messages. The instructor cross-checks Git history against the contribution report, and code dumped hours before the deadline does not count as a contribution.
- Everyone commits from day one, every day — timely, substantive commits that clearly document what you actually did. Don't chase a specific commit count for its own sake; a handful of large, well-described commits that clearly show real work is better evidence than dozens of trivial ones padding the count. The instructor looks at Git history, the contribution report, and defense performance together, not commit-count parity alone.
- Nobody force-pushes to main.

## 0.6 The core project (guaranteed) and the stretch goal (attempted only if the core is done)

This project has **one guaranteed result, required for a passing submission, and one stretch goal, attempted only after the guaranteed result is fully working end-to-end.** They are not co-equal. Ordering them any other way risks the entire project on its riskiest, least-tested component.

**The guaranteed result (Result 1 — do this first, this is the project):** XGBoost vs. TCN vs. GRU, compared at a matched false-alarm operating point, **and** real-only vs. real+simulated training for each of those three models. This does not need a new module beyond what's already specced in Parts 1-4. It reuses the `condition` column that already exists in the results contract (§0.3): whoever trains a model — Member 2 for XGBoost, Member 3 for TCN/GRU — just runs their existing pipeline **twice**, once per condition (`real_only`, `real_plus_sim`), and logs both. Simulated instances stay training-side only, per W1.7's grouped-split guarantees — never in a val or test fold, in either condition. Even a "simulated data didn't help" finding is a fully valid, defensible result under the brief's own rules — negative results are explicitly acceptable.

**The stretch goal (only after Result 1 is complete, `results.csv` is populated, and the core pipeline runs end to end):** self-supervised masked-reconstruction pretraining vs. random init (Member 3's design in Part 3 — `ssl_pretrain.py`, `finetune.py`, `ReconstructionHead`). This is intellectually the most interesting piece of the project and Member 3's design for it is genuinely good — it isn't being cut, it's being reordered. The reason: your own team doc calls the fold-safe pretraining corpus "the most likely leakage bug in the whole project," and it requires the pretrained-vs-random-init runs to be identical in every other respect to mean anything. That's real risk on a real deadline. If it succeeds, it becomes a genuinely strong addition to the Results section and the abstract. If it runs out of time or something in it breaks, the project's actual scientific claim — does deep learning beat a strong baseline, does augmentation help — is already complete and defensible without it. Do not start building `ssl_pretrain.py` before Result 1 is fully working; that ordering is not optional.

If the stretch goal is attempted, it logs `condition=pretrained` / `condition=random_init` into the same `results.csv`, using the same schema as Result 1 — no separate reporting pipeline.

Result 1 goes in the paper as its own Results subsection and is required. If the stretch goal succeeds in time, it gets a second Results subsection and a mention in the abstract; if it doesn't get attempted or doesn't work out, the paper is still complete without it — say so plainly in Discussion rather than padding around a missing result.

## 0.7 The three sync points

| Sync | When | What happens |
|---|---|---|
| **S1** | End of day 1 | Contract frozen, fake generator working, all four directories created, everyone has run something end to end on fake data |
| **S2** | When the real cache lands | Member 1 announces it and reports the well count behind the real hydrate instances. Everyone swaps the data path and runs on real data for the first time |
| **S3** | Freeze | All tuning finished on validation folds. Everything frozen. Only after this does Member 4 run the test folds, exactly once |

**Rule for S3:** after the freeze, if anybody wants to change a hyperparameter, the answer is no. Changing anything after seeing test numbers is tuning on the test set, which the brief lists as an automatic deduction.

---

# Part 1 — Member 1: Data Module

## 1.1 Mission

Turn 3W Parquet files into a cached, windowed, leakage-safe tensor dataset that the other three can load in one line. You own everything from the raw download to `X, mask, y, group`.

You are on the critical path for two things only: the fake generator (hour one) and the real cache (as early as possible). Everything else you do is analysis that improves the paper.

## 1.2 Files you own

```
src/data/
|-- download.py            # fetch from Figshare, verify checksums
|-- inventory.py           # build the instance catalogue
|-- availability.py        # missingness and frozen-value analysis
|-- windowing.py           # windowing + label assignment (label_window(), see W1.6)
|-- splits.py              # grouped k-fold by well
|-- build_cache.py         # the main entry point: raw -> .npz cache
|-- make_fake_data.py      # synthetic data matching the contract
`-- stats.py               # dataset statistics tables and figures
src/contract.py            # shared constants (NORMAL/TRANSIENT/ESTABLISHED, 3W
                            # label codes, results schema) -- you create this in
                            # week one since you're the first to need these
                            # enums; everyone else imports from it, nobody
                            # redefines it locally (§0.2)
README.md                  # first draft -- you're the one hitting missing
requirements.txt           # dependencies and setup friction first; Member 4
.gitignore                 # does the final consistency pass before submission
```

## 1.3 Work items in order

### W1.1 — Fake data generator (hour one, highest priority)
Generate `X, mask, y, group, inst_id, t_end, is_sim, failure_time, normal_hours` with realistic shapes: say 8 fake wells, 40 fake instances, 20 channels, 300-step windows. Plant a detectable ramp before each transition to label 2. Include some missing values in the mask so downstream code is forced to handle them.

**Done when:** Members 2, 3 and 4 can each load it and run their code end to end.

### W1.2 — Download and read the Parquet correctly
The 3W 2.0.0 data is in Apache Parquet, and the files must be read with the Pyarrow engine alongside Brotli compression. This is a common first-hour blocker. Pin `pyarrow` in requirements and confirm you can read one file before anything else.

Also read `dataset.ini` at the dataset root, which specifies dataset properties, and record the version string. Version matters: comparisons across 3W versions require care because of substantial differences between them, so the paper must state 2.0.0 explicitly.

### W1.3 — Build the instance inventory (**the most important early deliverable**)
Produce a table with one row per instance: instance id, event type, source type (real / simulated / hand-drawn), **well id**, duration, number of samples, which label periods it contains and their durations.

From this table, answer and report to the team at S2:

1. **How many distinct real wells lie behind the Hydrate in Service Line instances?** This determines whether grouped cross-validation is viable and how many folds you can have. Everything in the experimental design depends on it.
2. What are the transient period durations? If transients are very short, the lead-time question needs reframing, and the team needs to know immediately.
3. How many instances contain a complete normal → transient → blocked progression?
4. How many simulated Hydrate instances exist (expect 150), and confirm none of them share a `well_id`/group with a real well — the inventory's source column plus your group-assignment logic is what Member 2 and Member 3 rely on for Result 1 (§0.6). Report this count alongside the real well count at S2.

**Deliverable:** `results/instance_inventory.csv` plus a short written summary posted to the team.

### W1.4 — Variable availability and frozen-value analysis
For each of the 27 variables, compute across the relevant instances: percentage of samples present, percentage frozen, percentage of instances where the variable is entirely absent.

Frozen-value detection: a value is frozen if it is bit-identical for longer than a threshold (start with 60 seconds at 1 Hz, and justify the choice). Mark frozen stretches as missing in the mask.

**Deliverable:** a table for the paper, one row per variable, with availability, frozen rate, and a keep/drop decision.

### W1.5 — Freeze the variable list
Select the subset of variables to use, based only on availability and physical relevance. **Freeze this list before any modelling.** Never revisit it based on test results. Record the decision and the reason in the config.

### W1.6 — Windowing
Implement window extraction with window length `W` and stride `S`. Provide `W` and `S` as config parameters so Members 2 and 3 can sweep them on validation folds.

**Label rule: default to `most_severe`, not `any_transient` or final-timestep-only.** A window's label is the most severe state present anywhere in it (`Normal < Transient < Established`), implemented once in `src.data.windowing.label_window()` — call that function, don't reimplement the logic inline.

This fixes a real bug in the naive `any_transient` rule this document used to recommend: a window shaped `[Normal, Transient, Transient, Established, Established]` would get called "Transient" under `any_transient`, even though the window has already progressed to Established — silently reporting a *less* advanced state than the window actually contains. `most_severe` can't make that mistake, since Established always outranks Transient.

**Correction to an earlier version of this section:** this document used to also claim `most_severe` flags onset earlier than `final_timestep`. That claim was wrong and has been removed. Under a causal sliding window, `final_timestep` also flags a window as Transient the moment its last sample first becomes Transient — it doesn't need to wait for the window to fully advance past onset. In fact, whenever an instance's severity never decreases over time (Normal → Transient → Established, no reversion — the expected case for one failure progression), `most_severe` and `final_timestep` are **mathematically identical** for every window: the maximum of a non-decreasing sequence always equals its last element. This is now a checkable fact, not an assumption — `src.data.windowing.is_monotonic_severity()` is implemented and tested. **Add this to Member 1's inventory step (W1.3):** run it on every real Event-9 instance and report the fraction that's monotonic. If it's always True, the label_rule choice provably doesn't change any result for this event, and `most_severe` stays the default purely as a free safety margin against the untested edge case, not because of any onset-timing advantage. If any real instance turns out non-monotonic, that's a genuinely interesting data-quality finding worth reporting on its own, and the two rules will actually diverge for that instance — worth a closer look before trusting either blindly.

`label_rule` is a config flag (`"most_severe"` | `"final_timestep"` | `"majority"`) — run all three once as a cheap labeling-rule sensitivity sweep (report event recall and lead time under each). `label_window()` is already implemented and unit-tested (`tests/test_windowing.py`, including the exact bug scenario above) — you're wiring it into `build_windows()`, not writing the labeling logic from scratch.

Consider downsampling from 1 Hz (for example to 0.1 Hz by averaging) to shorten sequences. If you do, make it a config flag, not a hard-coded choice, and report it.

### W1.7 — Grouped splits
```python
def get_folds(group, k, seed) -> list[(train_idx, val_idx, test_idx)]
```
Guarantees, which you must **write as unit tests**:

- No well id appears in more than one of train / val / test in any fold.
- No instance is split across sets.
- Every fold's test set contains at least one hydrate event, or the fold is rejected and reported.

If the well count is too low for clean k-fold, implement repeated leave-one-well-out instead and say so.

### W1.8 — The normalisation trap
**This is the leakage risk that most teams miss.** Feature scaling statistics must be computed on the training fold only, never on the whole dataset. Two acceptable options:

- Fit the scaler inside each fold on training data, and expose it so Members 2 and 3 apply the same fitted scaler to val and test.
- Use per-instance robust scaling that uses no cross-instance information at all.

Pick one, document it, and put a unit test around it. Be ready to explain at the defense why fitting a scaler on all data before splitting would be leakage.

### W1.9 — Build and cache
`build_cache.py` writes the cached arrays to NVMe scratch. Deterministic given a seed. Records the config used in a JSON sidecar file.

### W1.10 — Dataset figures for the paper
- Transient duration histogram.
- A raw multi-variable trace of one hydrate instance with the three label periods shaded. This becomes the paper's opening figure and the anchor of the presentation.
- Variable availability bar chart.

## 1.4 What you must be able to defend

- Why splits are grouped by well, and exactly what leakage would occur if windows were split randomly.
- Why the scaler is fitted inside the fold.
- How frozen values are detected and why they are treated as missing rather than as real values.
- How many wells the results rest on, and what that means for the confidence in the numbers.

## 1.5 Paper sections you write

The Dataset section, the variable availability table, the data-preparation part of Experimental Setup, the data figures, **and the Introduction** (you understand the domain problem and its practical stakes better than anyone at this point — write it from there, not from a generic template). Draft your own slide(s) for these too (§5.3).

## 1.6 Pitfalls

- Spending three days perfecting imputation. Do not. Mask channels are enough for this project.
- Deciding the variable list after seeing which variables help the model. That is leakage of a subtle kind.
- Delivering the real cache late. Everything downstream is fine on fake data, but the real numbers cannot start until you ship. Prioritise a *working* cache over a *perfect* one, and refine after.

---

# Part 2 — Member 2: Baseline Module

## 2.1 Mission

Build an XGBoost baseline strong enough that beating it means something. The brief weights experimental rigour at 20%, and a large part of that is whether your baseline is fair or a strawman. Your job is to make the deep models work for their win, or to honestly demonstrate that they do not get one.

Take this framing seriously: **you are not building the thing that loses.** If XGBoost wins, that is a genuine and interesting result, and the paper says so.

## 2.2 Files you own

```
src/baselines/
|-- features.py            # rolling-window feature extraction
|-- xgb_model.py           # the Model class satisfying the contract
|-- tune.py                # hyperparameter search on validation folds
|-- calibrate.py           # probability calibration
`-- importance.py          # feature importance analysis
```

## 2.3 Work items in order

### W2.1 — Feature extraction
For each window and each kept variable, compute at minimum:

- mean, standard deviation, min, max, range
- slope from a least-squares linear fit over the window
- difference between the last value and the value at the window start
- mean over the most recent quarter of the window, minus mean over the earliest quarter
- fraction of the window where the mask says the value was present

Compute these at **two or three time scales** (for example the full window, the last half, the last quarter). Multi-scale features are what make a tabular baseline genuinely competitive with a temporal model, and skipping them is what makes a baseline weak.

**Mask awareness matters:** compute statistics only over present values, and pass the presence fraction as its own feature so the model knows how much data a statistic was based on.

### W2.2 — The model
Multiclass XGBoost over the three classes. Handle class imbalance explicitly through class weights or sampling, as the brief requires, and record which approach was used.

Wrap it in the contract's `Model` interface so Member 4's evaluation code treats it identically to the deep models.

### W2.3 — Hyperparameter search, budget-matched
Search over depth, learning rate, number of estimators, subsample, colsample, and minimum child weight. Use validation folds only.

**Record how many configurations you tried.** Member 3 records the same for the deep models. The paper states both, so a reader can see the baseline got a comparable tuning budget. This single sentence in the paper is worth a lot on the rigour criterion.

### W2.4 — Calibration
Fit isotonic regression or Platt scaling on validation folds and report reliability diagrams before and after. The brief names calibration explicitly as an example of domain-appropriate evaluation, so this is directly rewarded.

### W2.5 — Feature importance analysis
Report which features and which physical variables carry the signal. This is one of the few places the project generates genuine domain insight: if pressure slope upstream dominates, that is a physically interpretable finding worth a paragraph and a figure.

### W2.6 — Window-length sensitivity
Run the baseline across the candidate window lengths on validation folds. This helps Member 3 pick a sensible starting point without burning GPU time, and it is a cheap extra analysis for the paper.

### W2.7 — Real-vs-simulated augmentation (Result 1, required, see §0.6)
Once W2.2-W2.4 are working, train the finalized baseline **twice** per fold: once on real Event-9 instances only (`condition=real_only`), once on real + the 150 simulated instances (`condition=real_plus_sim`). Simulated instances go in the training side only — never val or test, in either run (Member 1's group assignment already guarantees no simulated instance can land in a val/test fold; you don't need to filter anything extra). Log both to `results.csv` under those condition tags. This is the baseline's contribution to Result 1 — no new module, just two training runs instead of one.

## 2.4 What you must be able to defend

- Why this baseline is fair, with the tuning-budget numbers to prove it.
- Why PR-AUC rather than ROC-AUC or accuracy under this class imbalance.
- What the feature importances say physically about how a hydrate announces itself.
- What would have to be true for a deep model to beat this baseline, in terms of the structure in the signal that features cannot capture.

## 2.5 Paper sections you write

The baseline subsection of Method, Related Work, the feature importance figure, and the calibration discussion. Draft your own slide(s) for these too (§5.3). You also draft the limitations paragraph for the baseline (what it structurally can't capture) — Member 4 folds it into Discussion.

## 2.6 Pitfalls

- Building a deliberately weak baseline so the deep models look good. The defense will find it, and it is the fastest way to lose rigour marks.
- Single-scale features only. This is the main reason tabular baselines underperform unfairly.
- Forgetting that XGBoost also needs the mask information. A NaN and a real zero are not the same thing.

---

# Part 3 — Member 3: Deep Models Module

## 3.1 Mission

Build the TCN and GRU and run the deep-model side of Result 1, the guaranteed comparison (see §0.6): TCN vs. GRU vs. XGBoost at a matched false-alarm budget, and real-only vs. real+simulated training for both TCN and GRU (W3.7's A1 below — cheap, reuses your existing training pipeline, no new module). Get this fully working end to end before touching anything else.

Once Result 1 is complete and `results.csv` is populated, implement the self-supervised pretraining stage as the project's stretch goal: **does pretraining on abundant normal data help when labelled failures are rare?** This is genuinely the most interesting technical question in the project, and your design for it (below) is solid — it's just not the thing the project's success depends on. Don't start `ssl_pretrain.py` before Result 1 works.

You own the 30% technical-execution criterion more directly than anyone else.

## 3.2 Files you own

```
src/models/
|-- tcn.py                 # dilated causal conv network
|-- gru.py                 # recurrent alternative
|-- heads.py               # classification head, reconstruction head
|-- ssl_pretrain.py        # masked reconstruction pretraining
|-- finetune.py            # supervised fine-tuning
|-- train_loop.py          # AMP, checkpointing, early stopping, seeds
`-- profile.py             # parameter counts and inference latency
```

## 3.3 Work items in order

### W3.1 — The TCN
Dilated causal convolutions in residual blocks, with dilation doubling per block. **Compute the receptive field explicitly** and make sure it covers the intended window length. Write it in the code as a comment and in the paper as a sentence, because it is a very likely defense question.

Input channels are the kept variables **concatenated with their mask channels**, so a model with 20 variables has 40 input channels.

Target size: on the order of 10⁵ to 10⁶ parameters. Small is correct here, not a compromise.

### W3.2 — The GRU
A comparable-parameter-count recurrent model. Its purpose is to show that any pretraining finding is not specific to one architecture.

### W3.3 — The sanity check that must pass before anything else
The brief advises overfitting a tiny subset to prove the code learns before launching the full run. Take 20 windows, turn off regularisation, and train until the model reaches near-zero training loss. **If it cannot overfit 20 windows, the pipeline is broken and no amount of training will fix it.**

Do not skip this. It catches label misalignment, mask sign errors, and dead gradients in ten minutes rather than two days.

### W3.4 — Masked reconstruction pretraining
1. Build the pretraining corpus from data not used as labelled hydrate examples in the current fold: normal operation instances and non-event portions of other event types. No labels are used.
2. Mask random **contiguous spans** rather than isolated points. Span masking is harder and forces the model to learn temporal structure rather than local interpolation. Start with a masking ratio around 15–25%.
3. Loss is computed only on masked positions **and** only where the availability mask says the true value exists. Reconstructing a value that was never recorded is meaningless.
4. Save the encoder weights. Discard the reconstruction head.

**The pretraining corpus must respect fold boundaries.** If you pretrain on data from a well that appears in the test fold, you have leaked. Coordinate with Member 1 so the pretraining corpus is built per fold from training wells only. Write a test for this.

### W3.5 — Fine-tuning
Attach a classification head to the encoder and fine-tune. Try both full fine-tuning and a short frozen-encoder warm-up followed by unfreezing, and select on validation.

**The critical constraint:** M1 (random init) and M2 (pretrained) must use the **identical** architecture, identical fine-tuning recipe, identical data, identical schedule, identical seeds. The only difference is where the weights started. If anything else differs, the ablation proves nothing.

### W3.6 — Training infrastructure
Mixed precision, checkpoint every epoch with resume-from-checkpoint, early stopping on a validation metric, seeds controlled for weight init and shuffling, metrics logged to disk not just stdout. Your booked window can end mid-training, so resumption is not optional.

### W3.7 — Ablations you run
- **A1 (required, Result 1, do this first — §0.6):** real-only vs. real+simulated training, TCN and GRU, all three folds, three seeds. This is the deep-model side of the guaranteed comparison; Member 2 runs the equivalent for the baseline. Log both to `results.csv` as `condition=real_only` / `condition=real_plus_sim`.
- **A2 (required):** with vs. without mask channels.
- **A3 (stretch goal, only after A1/A2 and the full pipeline are working — §0.6):** pretrained vs. random init, all three folds, three seeds. If attempted, log as `condition=pretrained` / `condition=random_init`.
- **A4 (stretch goal, only if A3 happens):** masking ratio sweep (for example 0.10, 0.15, 0.25, 0.40), so the pretraining result is not one lucky setting.

### W3.8 — Profiling
Parameter count and inference latency per window for each model. Needed for the practical-implications discussion the Track 2 requirements ask for.

### W3.9 — Record your tuning budget
How many configurations you tried, so it can be compared with Member 2's. This goes in the paper.

## 3.4 What you must be able to defend

- The receptive field of your TCN and why it is adequate for the window length.
- Why masked reconstruction is a sensible pretraining task for process time series, and what the encoder plausibly learns from it.
- Why the pretraining corpus must be built per fold.
- What it would mean if pretraining did **not** help, and why that would still be a real finding rather than a failure.

## 3.5 Paper sections you write

The deep model subsection of Method, Experimental Setup for the deep models, the architecture diagram, **and the Conclusion** (the guaranteed result — TCN/GRU vs. baseline, real-vs-sim — is deep-model territory; write the closing claim from having lived inside it). If the pretraining stretch goal is attempted and succeeds in time, you also write its description as an addition to Method and a paragraph in Results. Draft your own slide(s) for these too (§5.3). You also draft the limitations paragraph for the deep models (and pretraining, if attempted) — Member 4 folds it into Discussion.

## 3.6 Pitfalls

- Making the model bigger to "improve" it. With 57 events, a larger model overfits harder. Small is the right call and you should be able to say why.
- Letting M1 and M2 differ in more than initialisation. This silently destroys the headline claim.
- Pretraining on data that includes test wells. The most likely leakage bug in the whole project.
- Skipping the tiny-subset overfit check because you are in a hurry.

---

# Part 4 — Member 4: Evaluation, Reproducibility and Reporting

## 4.1 Mission

You own the metrics, the plots, the repository, the paper skeleton, and the test-set gate. This is not the lightest role. You are the one who turns three working modules into one submission, and several automatic deductions in the brief land squarely in your territory.

You are also the **test-set gatekeeper.** Nobody else runs the test folds. Ever.

## 4.2 Files you own

```
src/eval/
|-- metrics.py             # all metric implementations
|-- alarm.py               # alarm definition, lead time (causal/trailing smoothing only)
|-- thresholds.py          # threshold selection at fixed false-alarm budget
|-- aggregate.py           # results.csv -> tables
|-- plots.py               # every figure
run_all.sh                 # the one-command reproduction -- yours alone
README.md                  # drafted by Member 1, you do the final pass (§0.5/W4.6)
requirements.txt           # drafted by Member 1, you do the final pass
.gitignore                 # drafted by Member 1, you do the final pass
report/                    # report.tex, generated tables
presentation/
contribution_report.pdf
```

## 4.3 Work items in order

### W4.1 — Metrics, developed against random arrays
You do not need anyone else's code to start. Generate random `[N, 3]` probability arrays and fake labels, and implement:

- **Event recall:** fraction of real hydrate events flagged at any point during their transient phase.
- **PR-AUC** per class.
- **False alarms per operating hour:** count alarm *onsets* on normal-operation data, divide by `normal_hours`. Count onsets, not samples, or one long alarm counts as thousands.
- **Median warning lead time** and the full lead-time distribution.
- **Expected calibration error** and reliability diagrams.
- **Per-well breakdown** of all of the above.

Write unit tests with hand-constructed cases where you know the right answer.

### W4.2 — The alarm definition (write this before any results exist)
```python
def alarm_times(proba, t, smooth_window, threshold, min_duration) -> list[float]:
    """Smooth the positive-class probability with a rolling average.
    Declare an alarm when the smoothed value stays above threshold
    for at least min_duration continuous seconds.
    Return the onset timestamps."""
```
Lead time is `failure_time - first_alarm_time` for alarms occurring before failure.

**This definition is frozen before results are seen.** Write it down, date it, commit it. Being able to say "we fixed this definition before we saw any results" is worth real credit at the defense, and changing it afterwards to make numbers look better is exactly the behaviour the rubric penalises.

### W4.3 — Threshold selection
Select thresholds on validation folds to hit a **fixed false-alarm budget** (for example one false alarm per 100 operating hours), then apply them unchanged to test folds. Every model is compared at the same budget, which is what makes the lead-time comparison meaningful.

Also produce the **lead time versus false-alarm rate curve** by sweeping the budget. This is the paper's most informative figure, because it shows the whole trade-off rather than one operating point.

### W4.4 — Aggregation
Read `results/results.csv`, group by model / condition / metric, produce mean ± standard deviation across folds and seeds, and emit LaTeX tables directly into `report/tables/`. The paper `\input`s them. No hand-typed numbers, ever.

### W4.5 — The four figures
1. Raw sensor trace with the transient shaded and each model's alarm marked. (Built with Member 1's plotting code.)
2. Lead time versus false-alarm rate, one curve per model.
3. Per-well lead-time box plot.
4. Reliability diagram before and after calibration.

The brief says one clear comparison plot and one honest results table beat ten decorative ones. Resist adding more.

### W4.6 — Reproducibility infrastructure
- `run_all.sh` reproducing every headline number from a clean checkout. The brief lists "no `run_all` entry point" as an **automatic deduction**. This one is yours alone — it's the evaluation gatekeeper's job by nature, since it has to know the full pipeline end to end.
- `requirements.txt`, `README.md`, `.gitignore` — **drafted by Member 1** in week one (they're the ones actually running the setup steps and hitting missing dependencies first), **kept up to date by whoever's change breaks them**. You do a final consistency pass before submission, but these three files are not yours to solo-own — see the note in §0.5. This is a deliberate small shift from the first draft: three tiny files shouldn't add to an already-heavy integration role when moving them costs nothing.
- Test the whole thing by cloning into a fresh directory and running it. Untested reproduction instructions are usually broken instructions.

### W4.7 — The paper
Set up the IEEE two-column template in week one and fill Method and Setup while they are fresh, as the brief advises. You write: **Abstract, Results, Tools and Acknowledgements**, and you assemble **Discussion and Limitations** from the paragraph each member drafts for their own module (per §5.3 — you are the editor there, not the sole author). Introduction is Member 1's and Conclusion is Member 3's — chase them for a draft in week one, don't write it yourself to save time; the rebalance only holds if it holds.

Format requirements you own:
- Authors listed as Surname, Given name(s), sorted **alphabetically by surname**.
- The Abstract **ends with the GitHub repository link**, tagged `v1.0-final`, per IEEE convention as the brief specifies.
- **No leftover template text anywhere.** This is an automatic deduction and it is entirely avoidable. Search the document for template boilerplate before submitting.
- The AI-assistance disclosure in Tools and Acknowledgements.

### W4.8 — Slides and contribution report
Assemble the deck from each member's drafted slides, enforce the 15–20 minute budget by timing a rehearsal, and keep it figure-driven.

Prepare `contribution_report.pdf` in the repository root with columns for Member Name, Code, Report Sections, Experiments, Slides, Other, plus a short description per member, agreed by all four. Cross-check it against Git history yourself before submitting, because the instructor will.

### W4.9 — The final tag
`git tag v1.0-final` and push. Confirm the tagged commit is the one that reproduces the paper's numbers.

## 4.4 What you must be able to defend

- The exact alarm definition and why it was chosen over a simpler rule, including why it applies identically to XGBoost and the deep models.
- Why thresholds are chosen at a fixed false-alarm budget rather than by maximising F1.
- What the per-well variance says about how much to trust the headline number.
- Why event recall and false alarms per hour are the right metrics for an operator, and accuracy is not.

## 4.5 Pitfalls

- Starting the paper in the last two days. Start it in week one.
- Counting alarm samples instead of alarm onsets, which inflates the false-alarm rate by orders of magnitude.
- Letting someone run the test folds "just to check". Say no.
- Writing reproduction instructions without testing them from a clean clone.

---

# Part 5 — Shared obligations

## 5.1 Cross-teaching (schedule this, do not leave it to chance)

Every member must be able to explain any part of the work at the defense, and repeated "I don't know" answers lower your individual score. So rotate 30-minute walkthroughs:

| Session | Teacher | Student |
|---|---|---|
| 1 | Member 1 | Member 2 |
| 2 | Member 2 | Member 3 |
| 3 | Member 3 | Member 4 |
| 4 | Member 4 | Member 1 |

Then swap directions so all pairs are covered.

## 5.2 The five questions everyone answers, regardless of module

1. Why are splits grouped by well, and what would leakage look like otherwise?
2. Why PR-AUC and false alarms per hour instead of accuracy?
3. What exactly counts as the first reliable alarm, and why that definition — and why must the smoothing be causal (trailing), not centered?
4. Why does real-vs-simulated augmentation (Result 1, guaranteed) matter, and what would it mean if simulated data showed no benefit?
5. Why is pretraining a stretch goal rather than a guaranteed result, and what specific risk does that ordering protect the project against?

## 5.3 Paper section ownership

Rebalanced from the first draft: Member 4 already carries the heaviest integration load (aggregation scripts, every plot, reproducibility infra, slide assembly, contribution report, final tag, sole test-set access) — stacking most of the paper's prose on top of that as well would make Member 4's contribution look disproportionate against Git history even if the *work* were fairly divided, and risks the individual-imbalance penalty in the rubric. Introduction moves to Member 1 (natural extension of owning the Dataset section) and Conclusion moves to Member 3 (natural extension of owning the deep-model results). Discussion is jointly drafted, not solo-written.

| Section | Owner |
|---|---|
| Abstract | Member 4 |
| Introduction | Member 1 |
| Related Work | Member 2 |
| Method — data preparation | Member 1 |
| Method — baseline | Member 2 |
| Method — deep models (+ pretraining, if the stretch goal is attempted) | Member 3 |
| Experimental Setup | Members 1 and 3 |
| Results (Result 1 — required; pretraining stretch goal — only if it happened) | Member 4 |
| Discussion and Limitations | Member 4 assembles; **each member drafts the limitations paragraph for their own module** (a real cross-check discipline, not busywork — the person who built something knows its failure modes best) |
| Conclusion | Member 3 |
| Tools and Acknowledgements | Member 4 |

**Slides follow the same split.** Per W4.8, each member drafts the slide(s) for the section(s) they wrote above; Member 4 assembles the deck, enforces the 15–20 minute budget, and keeps it figure-driven. Nobody hands their slide content to someone else to write.

## 5.4 Definition of done, for the whole project

- [ ] Dataset licence (CC BY 4.0) and toolkit licence (Apache 2.0) both stated, dataset version 2.0.0 stated, Figshare DOI cited
- [ ] Splits fixed up front, split sizes reported, test folds touched exactly once
- [ ] Baseline present and fair, with both tuning budgets reported
- [ ] Result 1 present and complete: XGBoost vs. TCN vs. GRU at a matched false-alarm budget, AND real-only vs. real+simulated for each model, each isolating a single changed variable
- [ ] Pretraining stretch goal: either fully reported (if attempted and finished) or explicitly and honestly noted as not attempted / not completed in time — never silently absent
- [ ] Metrics go beyond accuracy and match what the domain cares about
- [ ] Seeds fixed, environment pinned, checkpoints and logs saved
- [ ] Headline result reproduced from a clean checkout by one command
- [ ] IEEE two-column format, abstract ends with the repository link, authors alphabetical by surname, no leftover template text
- [ ] AI assistance disclosed
- [ ] Every claim backed by a table or figure
- [ ] Slides figure-driven, timed at 15–20 minutes, submitted before the defense
- [ ] `contribution_report.pdf` in the repository root, agreed by all four, consistent with Git history
- [ ] Repository tagged `v1.0-final`
- [ ] Every member can explain every part of the project
