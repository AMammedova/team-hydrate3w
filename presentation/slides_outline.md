# Hydrate Formation Early Warning — Final Presentation

**Template & Assembly by: Member 5**

> **Note for the team:** M5 is responsible for assembling and formatting the final slide deck. Each member is responsible for their assigned slides. Please drop your content (bullet points, figures) in this document, and M5 will build the final `.pptx` or LaTeX Beamer deck for submission.

---

## 1. Title Slide (M5)
- **Title:** Early Warning of Hydrate Formation in Offshore Well Service Lines (3W Dataset 2.0.0, Event 9)
- **Team:** Members 1–5
- **Track:** Deep Learning Track 2

## 2. Motivation & Problem Statement (M1)
- Hydrate formation: economic and safety impacts in offshore oil & gas
- The need for early warning (vs. reactive response to blockage)
- **M1 TODO:** Add 3-4 bullet points

## 3. Dataset & Real Data Findings (M1)
- The 3W Dataset (Event 9): Normal, Simulated, Real events
- **Key Findings:** Only 14 real transient events, disjoint well populations, monotonic severity
- *Figure: Annotated raw sensor trace (M1 to provide from `plot_annotated_trace`)*

## 4. Cross-Validation & Experimental Setup (M2)
- Preventing leakage: Grouped K-Fold by well
- Independent splits for positive and normal wells
- *Table: Fold report summary (M2 to provide)*

## 5. Baseline: XGBoost & Feature Engineering (M3)
- Multi-timescale rolling feature extractor (30 min window)
- Handling class imbalance
- *Figure: Feature importance vs. Physical hydrate signature (M3 to provide)*

## 6. Deep Learning Architectures (M4)
- **TCN (Temporal Convolutional Network):** Causal dilations, receptive field = 61
- **GRU (Gated Recurrent Unit):** Unidirectional
- Both operating on (channels-first) raw data + mask
- *Figure: Architecture Diagram (M4 to provide)*

## 7. Evaluation Protocol (M5)
- **Lead time:** Time from first alarm to annotated *transient* onset
- **Matched FAR budget:** 1 false alarm per 100 operating hours
- **Strict causality:** Trailing smoothing, thresholds selected on validation folds only

## 8. Results: Model Comparison (M5)
- *Table: Head-to-Head Comparison (from `aggregate.py`)*
- Real-only vs. Real + Simulated conditions
- Which model achieved the best lead time at the target FAR?

## 9. Results: Trade-offs (M5)
- *Figure: Lead time vs. False Alarm Rate (M5 `plot_lead_time_vs_false_alarm_rate`)*
- *Figure: Per-well lead time distribution box plot (M5 `plot_per_well_lead_time_box`)*

## 10. Discussion & Limitations (M2)
- Simulated data systematically over-represents the blockage phase
- Small $N$ for real positive events (14 instances across 7 wells)
- **M2 TODO:** Add 2-3 bullet points

## 11. Conclusion & Future Work (M5)
- Summary of which architecture is recommended for early warning
- The impact of simulated data augmentation
- Future work: SSL pre-training (deferred in this sprint), larger datasets

## 12. Tools & Acknowledgements (M5)
- PyTorch, XGBoost, Scikit-Learn, Petrobras 3W toolkit
- Disclosure of AI assistance (per academic integrity requirements)
