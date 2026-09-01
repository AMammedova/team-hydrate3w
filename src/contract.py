"""
Shared constants and types for this project. Every module imports its
core enums from here rather than redefining them locally -- see
team_responsibilities_all_members.md §0.2. This file existing and being
imported everywhere is what prevents different members from silently
inventing slightly different conventions for the same thing.
"""

from __future__ import annotations

# 3-class window/instance state label, used everywhere (windowing, models,
# eval). Order matters: it is also the severity ranking used by
# windowing.label_window(rule="most_severe") -- higher value = more severe.
NORMAL = 0
TRANSIENT = 1
ESTABLISHED = 2
CLASS_NAMES = ("normal", "transient", "established")

# 3W dataset labeling convention (dataset.ini): an undesirable event's
# steady-state label is its event code; its transient label is
# 100 + event code.
EVENT_CODE = 9                              # Hydrate in Service Line
TRANSIENT_OFFSET = 100
TRANSIENT_RAW_LABEL = TRANSIENT_OFFSET + EVENT_CODE   # 109

# results/results.csv schema (team contract §0.3) -- every aggregation
# and plotting function assumes exactly these columns, in this order.
RESULTS_COLUMNS = ["model", "fold", "seed", "condition", "metric_name", "value"]

# Valid values for the `condition` column. Result 1 (guaranteed headline,
# §0.6 of the team doc) uses real_only/real_plus_sim. pretrained/random_init
# are reserved for the SSL-pretraining stretch goal -- only logged if that
# stretch goal is actually attempted, see team doc §0.6.
CONDITION_REAL_ONLY = "real_only"
CONDITION_REAL_PLUS_SIM = "real_plus_sim"
CONDITION_PRETRAINED = "pretrained"          # stretch goal only
CONDITION_RANDOM_INIT = "random_init"        # stretch goal only

# Tensor axis convention (Addendum A.2): channels-first, everywhere.
# X, mask: [N, C, W] = [n_windows, n_channels, window_size]
