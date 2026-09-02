<!--
Member 1's content for slides 1-3 (motivation + dataset), TEAM_5_MEMBERS.md
sec 6/8. Bullet-point outline only -- M5 assembles/formats the final deck
(sec 6). No .pptx tooling in scope here.
-->

# Slide 1 — Motivation

- Hydrate formation in offshore well service lines: gas hydrates crystallize
  under pressure/temperature/water-gas conditions common in production;
  fully-formed blockage can shut in a well for days.
- The failure has a detectable transient phase before it fully establishes
  -> genuine early-warning opportunity if caught in time to intervene.
- Goal: classify sensor windows as Normal / Transient / Established, early
  enough to give operators real lead time, at a controlled false-alarm rate.

# Slide 2 — Dataset (3W Event 9)

- Petrobras 3W Dataset 2.0.0, event 9 ("Hydrate in Production Line"):
  57 real + 150 simulated + 594 Normal-operation instances (all verified
  against the dataset's own documentation).
- Each instance: full-duration 1 Hz multivariate recording, median 32,135
  timesteps (real).
- **The positive class is rare and mostly transient-only:** of 57 real
  instances, only 14 ever show the transient phase and only 3 ever reach
  blockage -> lead time is measured against transient onset, not blockage.
- **Hydrate wells and Normal wells are completely disjoint** (15 vs. 9
  wells, zero overlap) -> per-instance normalization is mandatory, not
  optional, or a model can just learn "which well is this."

# Slide 3 — Data quality: the channel trade-off

- Different wells instrument different sensors; some sensors are frozen
  (stuck reading) for an instance's entire duration.
- Selecting channels by "which wells keep this channel" picks the wrong
  set: the criterion that matters is "which *events* survive it."
- Primary channel set (`P-MON-CKP, P-JUS-CKGL, T-TPT, T-JUS-CKP,
  P-ANULAR`, 5 channels) keeps **14/14 transient events, 3/3 blockage
  events, 7/7 positive wells**, with 1,467 normal hours for false-alarm
  calibration (14x the 100 h budget).
- A narrower, physically-intuitive-looking set (`P-TPT, T-TPT`) loses a
  third of the events -- including one of the three blockage cases -- and
  is used only as a Discussion sensitivity arm, not the headline result.
