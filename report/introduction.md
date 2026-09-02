<!--
Member 1 draft for report.tex's Introduction section. Markdown for now --
no IEEE/.tex template exists in this repo yet (report/ was empty except
.gitkeep); M5 folds this into report.tex during final assembly (TEAM_5_MEMBERS.md
sec 6). Content only, not final prose/citations.
-->

# Introduction

Hydrate formation in offshore well service lines is a slow-onset failure
mode: gas hydrates crystallize inside flowlines under the combination of
pressure, temperature, and water/gas content that offshore production
routinely produces, and a fully formed blockage can shut in a well for days.
Because the physical process passes through a detectable transient phase
before it fully establishes, there is a genuine early-warning opportunity —
if a model can recognize the transient signature in the multivariate sensor
stream early enough, an operator gets lead time to intervene (chemical
injection, pigging, rate changes) before the line is fully blocked.

This project builds and evaluates an early-warning classifier for hydrate
formation (3W Dataset event 9, "Hydrate in Production Line") using the
Petrobras 3W Dataset 2.0.0 — the largest public, labeled dataset of real and
simulated offshore well instrumentation for this class of problem. We frame
the task as three-class window classification (Normal / Transient /
Established) over a sliding window of multivariate sensor readings, and
evaluate models at a matched false-alarm budget so that lead time and
false-alarm rate can be compared fairly across architectures.

**Track 2 scope.** The required result (Result 1) is a head-to-head
comparison of a gradient-boosted baseline (XGBoost) against two deep
sequence models (TCN, GRU), each trained under two data conditions —
`real_only` and `real_plus_sim` — to quantify what the 3W dataset's
simulated instances actually add over real, sparse, positive-class data. A
stretch goal (self-supervised pretraining) is explicitly out of scope until
Result 1 is complete (see `TEAM_5_MEMBERS.md` §0.6) and is not attempted in
this submission; that decision is recorded in the Discussion section.

**What makes this dataset hard, in one paragraph.** Unlike many
early-warning benchmarks, the positive class here is genuinely rare (Section
Dataset) and the wells that ever exhibit the event share no wells with the
Normal-operation set — meaning a model that shortcuts to "which well is
this" can look deceptively strong under a naive evaluation. Sections
Dataset and Experimental Setup describe, with measured numbers rather than
assumptions, exactly how this shapes the windowing, channel selection, and
cross-validation design used throughout the rest of the paper.
