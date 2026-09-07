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
- **Preventing Leakage:** We implemented a Grouped K-Fold cross-validation strategy, grouped by well ID.
- **Independent Splits:** Normal wells and positive-event wells were split independently to ensure diverse representation across all folds.
- **Simulation Protocol:** Simulated instances were exclusively restricted to training folds to prevent them from inflating test metrics.

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
- **Simulation Bias:** Simulated data systematically over-represents the blockage phase, which is rarely reached in reality (only 3 real instances).
- **Small Sample Size:** The most critical limitation is having only 14 real transient instances across 7 wells, restricting the statistical power of the evaluation.
- **Generalization Challenges:** Disjoint well populations mean that generalizing to entirely unseen wells remains difficult.

## 11. Conclusion & Future Work (M5)
- **Recommendation:** [PLACEHOLDER] The [Model Name] architecture is currently recommended for early warning due to its robust lead time at the target FAR.
- **Future Directions:** 
  - Investigate self-supervised learning (SSL) pre-training on the vast unlabeled normal sequences (deferred in this sprint).
  - Expand the dataset to include more real positive events.
  - Explore multi-well domain adaptation techniques.

## 12. Tools & Acknowledgements (M5)
- **Stack:** Python 3.x, PyTorch 2.3.1, XGBoost 2.0.3, Scikit-Learn 1.5.0, Pandas, Matplotlib, and the 3W Dataset 2.0.0 toolkit by Petrobras.
- **Acknowledgements:** During development, AI assistants (Large Language Models) were used to structure boilerplate evaluation code, format LaTeX tables, and assemble this presentation. All AI-generated code and content were rigorously reviewed and validated by the team.
