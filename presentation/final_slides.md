# Hydrate Formation Early Warning — Final Presentation

## 1. Title Slide
**Early Warning of Hydrate Formation in Offshore Well Service Lines**
**(3W Dataset 2.0.0, Event 9)**
**Team:** Members 1–5
**Track:** Deep Learning Track 2

## 2. Motivation & Problem Statement (M1)
- **Economic & Safety Impact:** Hydrate formation in offshore well service lines can cause catastrophic blockages, leading to prolonged well shut-ins, lost production, and safety risks.
- **The Early Warning Opportunity:** Rather than reacting to an established blockage, early detection of the transient phase enables proactive intervention.
- **Goal:** Classify sensor data windows into Normal, Transient, or Established states early enough to give operators actionable lead time, while maintaining a strictly controlled false-alarm rate.

## 3. Dataset & Real Data Findings (M1)
- **3W Dataset (Event 9):** Comprises 57 real, 150 simulated, and 594 normal-operation instances.
- **The Positive Class is Rare:** Of the 57 real instances, only 14 exhibit the transient phase, and merely 3 reach full blockage. Lead time is therefore measured against transient onset.
- **Strict Disjointness:** Hydrate wells (15) and Normal wells (9) have zero overlap. Per-instance normalization is mandatory to prevent models from simply memorizing well identities.

## 4. Cross-Validation & Experimental Setup (M2)
- **Why grouping is mandatory here:** the 15 hydrate wells and the 9 Normal wells are *disjoint* — well identity alone predicts the label, and it is recoverable from sensor offsets. Every split is by well.
- **Two independent splits, paired fold by fold:** positive wells balanced on **events**, Normal wells balanced on **hours**. A single stratified split leaves fold composition to chance — and Normal hours range from 1,220 h (WELL-00002) to 6 h (WELL-00007).
- **Why 3 folds, not leave-one-well-out:** LOWO puts **5 of 7** test folds below 300 Normal hours, the thinnest at **49.9 h** — a budget of 1 alarm / 100 h is not measurable there. At k=3 every test fold holds **4–5 events and 509–562 Normal hours**.

| Fold | Val events | Val normal h | Test events | Test normal h |
|---|---|---|---|---|
| 0 | 1 | 530.7 | 5 | 509.2 |
| 1 | 3 | 305.2 | 5 | 561.8 |
| 2 | 5 | 509.2 | 4 | 530.7 |

- **Simulation protocol:** each simulated instance is its own pseudo-well and is never eligible for validation or test — `real_only` vs `real_plus_sim` differ *only* in the training side, so the comparison is paired on wells.
- **Frozen before test:** k=3, 1 repeat, nested validation, seed 42 — unchanged after any test number was seen.

## 5. Baseline: XGBoost & Feature Engineering (M3)
- **Feature Extraction:** A multi-timescale rolling feature extractor operating on a 30-minute causal window (scales: 1.0, 0.5, 0.25).
- **Extracted Stats:** Computes mean, std, min, max, slope, and last\_diff, and incorporates a mask-aware presence fraction feature for each channel.
- **Handling Imbalance:** Optimized thresholds on validation folds to maximize recall while adhering strictly to the FAR budget.

## 6. Deep Learning Architectures (M4)
- **TCN (Temporal Convolutional Network):** Features causal dilations (1, 2, 4, 8) across 4 blocks with a receptive field of 61. It operates directly on raw channels and explicitly takes masks into account.
- **GRU (Gated Recurrent Unit):** Unidirectional architecture to strictly enforce causality.
- **Input Strategy:** Both deep models process the selected 5-channel subset alongside missingness masks to learn from sensor drops.

## 7. Evaluation Protocol (M5)
- **Lead Time:** Defined as the time from the first alarm onset to the annotated *transient* onset.
- **Matched FAR Budget:** All models are compared at a fixed false-alarm budget of 1 alarm per 100 operating hours.
- **Strict Causality & No Peeking:** We used trailing (causal) smoothing. Thresholds were selected exclusively on validation folds (the S3 Freeze protocol) and applied to test folds without modification.

## 8. Results: Model Comparison (M5)
- **Methodology:** We compared the feature-engineered XGBoost baseline against TCN and GRU.
- **Conditions:** Each model was evaluated under two training conditions: Real-only data and Real + Simulated data.
- **[PLACEHOLDER] Outcome:** The [Model Name] model trained on [Condition] achieved the highest median lead time of [X] minutes and an event recall of [Y]%. 

## 9. Results: Trade-offs (M5)
- **Lead Time vs. FAR:** As false-alarm tolerance increases, models can detect the transient onset earlier, but at the cost of operator trust.
- **Simulated Data Impact:** [PLACEHOLDER] Adding simulated data [helped / hindered / did not affect] the model's ability to identify real transient phases.
- **Per-Well Distribution:** Lead times varied significantly across the 7 real positive wells due to differing sensor configurations and noise profiles.

## 10. Discussion & Limitations (M2)
- **The operating point does not transfer across wells.** Thresholds tuned on validation to ≤ 1 alarm / 100 h deliver **1.8–7.9 / 100 h** on test — while event recall stays at **0.00–0.23**. Too permissive for the budget *and* too strict to catch events means the alarms land on Normal data: the score scale shifts from well to well.
- **Why that was predictable:** every validation fold contains **exactly one positive well** (Table 1). The operating point is always fitted to one well's score distribution, then applied to 1–4 unseen wells.
- **Fold 0 cannot validate anything:** **1 positive window out of 11,260**. Early stopping, calibration and threshold selection are all fitted there; a calibrator cannot even be fitted. Fold 1 has the mirror problem on the training side — all training positives in one well, so inner CV could not tune and fell back to defaults.
- **Small sample:** 14 transient events / 7 wells; 3 blockage instances only, so the headline metric was redefined to transient onset. Lead time is computed from **0–2 flagged events** per cell — reported, but not a performance estimate.
- **Simulation bias:** simulated windows are **90.3% positive** vs a **3.34%** real rate and all 150 reach blockage. Mixed effect — for XGBoost, PR-AUC 0.11→0.21 and FAR 7.9→1.8, but recall 0.23→0.17. `real_only` is the headline.
- **Our own asymmetries:** the baseline is calibrated per fold, the deep models are not — so ECE is not like-for-like. (A monotone calibrator would *not* change any alarm decision, so this does not explain the recall gap.) SSL pretraining was **not attempted within the timeframe**.
- **We did not re-cut the folds after seeing test results** — that is the selection this protocol exists to prevent.

## 11. Conclusion & Future Work (M5)
- **Recommendation:** [PLACEHOLDER] The [Model Name] architecture is currently recommended for early warning due to its robust lead time at the target FAR.
- **Future Directions:** 
  - Investigate self-supervised learning (SSL) pre-training on the vast unlabeled normal sequences (deferred in this sprint).
  - Expand the dataset to include more real positive events.
  - Explore multi-well domain adaptation techniques.

## 12. Tools & Acknowledgements (M5)
- **Stack:** Python 3.x, PyTorch 2.3.1, XGBoost 2.0.3, Scikit-Learn 1.5.0, Pandas, Matplotlib, and the 3W Dataset 2.0.0 toolkit by Petrobras.
- **Acknowledgements:** During development, AI assistants (Large Language Models) were used to structure boilerplate evaluation code, format LaTeX tables, and assemble this presentation. All AI-generated code and content were rigorously reviewed and validated by the team.
