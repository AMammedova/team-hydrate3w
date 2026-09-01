"""
Module 2 — Windowing, Labeling, and Missingness Handling.
See DL_Project_Statement_Hydrate3W.docx, section 5 (DL2.1-DL2.5), and
its Addendum A.5 for the label_rule default.

Interfaces are frozen; build_windows() is still TODO. Whoever picks this
up: read DL2.1b before hardcoding window_size=60 -- it's an unvalidated
default that must be checked against real transient durations once
Module 1's summary() has run. label_window() below IS implemented and
tested -- use it, don't reinvent it inline in build_windows().
"""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from src.contract import ESTABLISHED, EVENT_CODE, NORMAL, TRANSIENT, TRANSIENT_OFFSET
from src.data.inventory import InstanceRecord

_SEVERITY_ORDER = (NORMAL, TRANSIENT, ESTABLISHED)  # low -> high; index IS the severity


def _raw_label_to_state(raw_label: int, event_code: int = EVENT_CODE) -> int:
    """Map a raw 3W class code to one of NORMAL/TRANSIENT/ESTABLISHED for this event."""
    if raw_label == event_code:
        return ESTABLISHED
    if raw_label == TRANSIENT_OFFSET + event_code:
        return TRANSIENT
    return NORMAL  # 0 (true normal) and any unrelated event code both collapse to Normal here


def label_window(
    raw_labels: np.ndarray, event_code: int = EVENT_CODE, rule: str = "most_severe"
) -> int:
    """
    raw_labels: per-timestep raw 3W class codes within one window.
    Returns one of NORMAL / TRANSIENT / ESTABLISHED.

    rule="most_severe" (DEFAULT): label = the most severe state present
    anywhere in the window (Normal < Transient < Established). Fixes a
    real bug in naive "any_transient" labeling: a window that has
    already progressed to Established but also contains an earlier
    transient sample would get called "Transient" under any_transient,
    silently reporting a LESS advanced state than the window actually
    contains. most_severe can't make that mistake, since Established
    always outranks Transient in the max.

    IMPORTANT, corrected reasoning (an earlier version of this docstring
    claimed most_severe "flags onset immediately, unlike final_timestep"
    -- that claim was wrong and has been removed): under a causal
    sliding window, final_timestep ALSO flags a window as Transient the
    moment the window's last sample first becomes Transient -- it does
    not need to wait for the window to "fully advance" past onset. In
    fact, whenever a single instance's severity never decreases over
    time (Normal -> Transient -> Established, no reversion -- the
    expected case for one failure progression), most_severe and
    final_timestep are PROVABLY IDENTICAL for every window: the max of a
    non-decreasing sequence always equals its last element. See
    is_monotonic_severity() below -- run it during Module 1's inventory
    step on real Event-9 instances. If it's ever False, most_severe and
    final_timestep genuinely diverge for that instance and the
    difference is worth a closer look; if it's always True (the expected
    case), the choice between the two doesn't actually change any
    result, and most_severe stays the implemented default purely because
    it's a strict, free safety margin against that unverified edge case,
    not because of any onset-timing advantage.

    rule="final_timestep": label = state at the window's last sample.
    Clean causal "nowcast" framing; mathematically identical to
    most_severe whenever is_monotonic_severity() holds for the instance
    (see above). Kept as the standard sensitivity-sweep alternative.

    rule="majority": label = the state that occupies more than half the
    window. A middle ground; also kept as a sensitivity-sweep option.
    """
    if len(raw_labels) == 0:
        raise ValueError("label_window() called on an empty window")

    states = np.array([_raw_label_to_state(int(l), event_code) for l in raw_labels])

    if rule == "most_severe":
        return int(states.max())
    if rule == "final_timestep":
        return int(states[-1])
    if rule == "majority":
        counts = np.bincount(states, minlength=len(_SEVERITY_ORDER))
        return int(counts.argmax())
    raise ValueError(f"unknown label_rule {rule!r} -- expected 'most_severe', 'final_timestep', or 'majority'")


def is_monotonic_severity(raw_labels: np.ndarray, event_code: int = EVENT_CODE) -> bool:
    """
    True if this instance's per-timestep severity never decreases over
    time (Normal -> Transient -> Established, no reversion). Module 1
    should call this on every real Event-9 instance during the inventory
    step (W1.3) and report the fraction that pass. Under monotonic
    severity, label_window(rule="most_severe") and
    label_window(rule="final_timestep") are mathematically identical for
    every window drawn from that instance -- this is a decidable,
    checkable fact about the real data, not a design opinion, and it's
    what actually settles whether the label_rule choice matters at all
    for this event.
    """
    states = np.array([_raw_label_to_state(int(l), event_code) for l in raw_labels])
    return bool(np.all(np.diff(states) >= 0))


# ---------------------------------------------------------------------------
# Frozen-value detection (DL2.1 / §2.3): a run of >= N identical consecutive
# readings is a stuck sensor, not signal. Detected on the RAW 1 Hz series,
# before any decimation, so `frozen_run_seconds` really is seconds.
# ---------------------------------------------------------------------------
LABEL_COLUMNS = ("class", "state")


def frozen_run_mask(values: np.ndarray, min_run: int) -> np.ndarray:
    """
    values: 1-D array, NaNs allowed. Returns a bool mask, True where the
    sample belongs to a run of >= min_run identical consecutive values.

    NaN != NaN, so a NaN breaks a run rather than extending it -- which is
    what we want: a gap is already missing, it is not a frozen sensor.
    """
    n = len(values)
    out = np.zeros(n, dtype=bool)
    if n == 0 or min_run <= 1:
        return out
    same = np.empty(n, dtype=bool)
    same[0] = False
    same[1:] = values[1:] == values[:-1]      # NaN comparisons are False
    starts = np.flatnonzero(~same)             # index where each equal-value run begins
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        if e - s >= min_run:
            out[s:e] = True
    return out


class VariableSelector:
    """
    Drops channels too incomplete to be worth imputing, and turns stuck-sensor
    runs into explicit NaN.

    LEAKAGE RULE (DL2.1): fit() must see TRAINING instances only. It is called
    once per CV fold with that fold's training instances, and the resulting
    column list is applied unchanged to val/test. Fitting it on everything
    leaks the distribution of the held-out wells into the split.
    """

    def __init__(self, max_missing_frac: float = 0.5, frozen_run_seconds: int = 60) -> None:
        self.max_missing_frac = max_missing_frac
        self.frozen_run_seconds = frozen_run_seconds
        self._kept_columns: list[str] | None = None
        self.missing_frac_: dict[str, float] = {}
        self.channel_means_: dict[str, float] = {}

    @property
    def kept_columns(self) -> list[str]:
        if self._kept_columns is None:
            raise RuntimeError("VariableSelector.fit() has not been called yet")
        return list(self._kept_columns)

    def fit(self, instances: Iterable[InstanceRecord]) -> "VariableSelector":
        """
        `instances` may be a list OR a generator -- build_cache streams the 594
        Normal instances through here one at a time rather than holding ~2.7 GB
        of DataFrames in memory at once.

        Missing fraction is computed AFTER frozen-run conversion, since
        transform() turns those runs into NaN: a channel that is 60% frozen is
        60% missing for our purposes, and counting it as present would keep a
        dead sensor in the model's input.
        """
        nan_weighted: dict[str, float] = {}
        n_weighted: dict[str, int] = {}
        sum_present: dict[str, float] = {}
        cnt_present: dict[str, int] = {}

        for inst in instances:
            for col in inst.variable_columns():
                v = inst.df[col].to_numpy(dtype="float64")
                v = np.where(frozen_run_mask(v, self.frozen_run_seconds), np.nan, v)
                present = ~np.isnan(v)
                nan_weighted[col] = nan_weighted.get(col, 0.0) + float((~present).sum())
                n_weighted[col] = n_weighted.get(col, 0) + len(v)
                sum_present[col] = sum_present.get(col, 0.0) + float(np.nansum(v))
                cnt_present[col] = cnt_present.get(col, 0) + int(present.sum())

        if not n_weighted:
            raise ValueError("VariableSelector.fit() received no instances")

        self.missing_frac_ = {c: nan_weighted[c] / n_weighted[c] for c in n_weighted}
        self.channel_means_ = {
            c: (sum_present[c] / cnt_present[c]) if cnt_present[c] else 0.0
            for c in n_weighted
        }
        self._kept_columns = sorted(
            c for c, f in self.missing_frac_.items() if f <= self.max_missing_frac
        )
        if not self._kept_columns:
            raise ValueError(
                f"every channel exceeded max_missing_frac={self.max_missing_frac}; "
                f"observed missing fractions: {self.missing_frac_}"
            )
        return self

    def means_for(self, columns: list[str]) -> np.ndarray:
        """channel_means_ as a [C] array ordered like `columns` -- this is what
        mask_missing()'s `channel_means` expects."""
        return np.array([self.channel_means_.get(c, 0.0) for c in columns], dtype="float64")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subsets to the fitted column list and converts frozen runs to NaN.

        The label columns ("class", and "state" when present) are carried
        through untouched -- build_windows() needs `class` to label windows, so
        stripping them here would just force every caller to re-attach them.
        """
        kept = self.kept_columns
        missing = [c for c in kept if c not in df.columns]
        if missing:
            raise ValueError(
                f"instance is missing fitted channels {missing}; the variable "
                f"list must be frozen across all instances (W1.5)"
            )
        out = df[kept].copy()
        for col in kept:
            v = out[col].to_numpy(dtype="float64")
            v[frozen_run_mask(v, self.frozen_run_seconds)] = np.nan
            out[col] = v
        for label_col in LABEL_COLUMNS:
            if label_col in df.columns:
                out[label_col] = df[label_col].to_numpy()
        return out


def mask_missing(
    X: np.ndarray,
    fill: str = "ffill_then_mean",
    *,
    channel_means: np.ndarray | None = None,
    max_gap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    X: float array [N, C, W] -- CHANNELS-FIRST (Addendum A.2). Time is the LAST
    axis. NaNs mark absent or frozen samples.

    Returns (X_filled, mask):
        X_filled  float32 [N, C, W]  -- no NaNs left
        mask      uint8   [N, C, W]  -- 1 = value was really observed

    CAUSALITY: filling is forward only. A NaN is filled from the most recent
    EARLIER observation, never a later one. Back-filling would import a future
    sample into the present, which for a lead-time metric is the same class of
    bug as centered alarm smoothing (Addendum A.9.1) -- it just happens one
    module earlier and is much harder to see.

    fill="ffill_then_mean": causal forward-fill (optionally capped at `max_gap`
        samples), then anything still unfilled -- leading NaNs with no earlier
        observation at all -- takes `channel_means` if given, else 0.0. Either
        way `mask` is 0 there, so the model is told the value was invented.
    fill="zero": no forward-fill; every absent sample becomes 0.0.

    `channel_means` (shape [C]) must come from TRAINING instances only --
    VariableSelector.means_for() provides them. Using the instance's own future
    mean here would be a second, quieter leak.
    """
    if X.ndim != 3:
        raise ValueError(f"expected X with shape [N, C, W], got {X.shape}")
    if fill not in ("ffill_then_mean", "zero"):
        raise ValueError(f"unknown fill {fill!r} -- expected 'ffill_then_mean' or 'zero'")

    Xf = np.array(X, dtype="float64", copy=True)
    mask = (~np.isnan(Xf)).astype("uint8")

    if fill == "zero":
        return np.nan_to_num(Xf, nan=0.0).astype("float32"), mask

    n, c, w = Xf.shape
    obs = mask.astype(bool)
    # Index of the most recent observed sample at or before each position.
    idx = np.where(obs, np.arange(w)[None, None, :], -1)
    idx = np.maximum.accumulate(idx, axis=-1)
    if max_gap is not None:
        too_old = (np.arange(w)[None, None, :] - idx) > max_gap
        idx = np.where(too_old, -1, idx)
    safe = np.where(idx < 0, 0, idx)
    Xf = np.take_along_axis(np.nan_to_num(Xf, nan=0.0), safe, axis=-1)
    unfilled = idx < 0
    if channel_means is not None:
        cm = np.asarray(channel_means, dtype="float64").reshape(1, -1, 1)
        if cm.shape[1] != c:
            raise ValueError(f"channel_means has {cm.shape[1]} entries, expected {c}")
        Xf = np.where(unfilled, np.broadcast_to(cm, Xf.shape), Xf)
    else:
        Xf = np.where(unfilled, 0.0, Xf)
    return Xf.astype("float32"), mask


def _masked_stat(X: np.ndarray, obs: np.ndarray, fn) -> np.ndarray:
    """Apply `fn` per channel over observed samples only; 0.0 where a channel
    observed nothing. Returns [N, C, 1] so it broadcasts against [N, C, W]."""
    n, c, _ = X.shape
    out = np.zeros((n, c, 1), dtype="float64")
    for i in range(n):
        for j in range(c):
            vals = X[i, j][obs[i, j]]
            out[i, j, 0] = float(fn(vals)) if vals.size else 0.0
    return out


def normalize_instance(
    X: np.ndarray,
    mask: np.ndarray,
    method: str = "warmup",
    warmup_samples: int = 120,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Per-instance standardisation of a [N, C, T] (or [C, T]) decimated series.

    WHY THIS IS NOT OPTIONAL HERE (W1.8, and DATA_FINDINGS.md §2): in this
    dataset the 15 hydrate wells and the 9 Normal-Operation wells are DISJOINT
    -- no well appears in both. So absolute sensor level is a perfect proxy for
    the label: a model can score well by recognising which well it is looking
    at and never learn anything about hydrates. Removing each instance's own
    offset and scale is what forces it to use shape instead of identity.

    method="warmup" (default): statistics from the first `warmup_samples`
        decimated samples of this instance only. Causal and deployment-honest
        -- an online system really can calibrate on the first stretch of a
        recording. Caveat to report: if an instance's transient begins inside
        the warm-up span those statistics are contaminated; at decimate=30 and
        warmup_samples=120 that span is the first hour, and the measured onsets
        in this dataset sit past it for all but the shortest instances.
    method="instance_robust": median / IQR over the WHOLE instance.
        Transductive -- it reads later samples of the same recording, so it is
        not deployable as-is. Kept as the sensitivity-sweep arm.
    method="none": passthrough, for the ablation that shows what the confound
        above is actually worth.
    """
    if method == "none":
        return X
    squeeze = X.ndim == 2
    Xw = X[None, ...] if squeeze else X
    Mw = mask[None, ...] if squeeze else mask
    obs = Mw.astype(bool)
    if method == "warmup":
        k = min(warmup_samples, Xw.shape[-1])
        seg, seg_obs = Xw[..., :k], obs[..., :k]
        center = _masked_stat(seg, seg_obs, np.mean)
        scale = _masked_stat(seg, seg_obs, np.std)
    elif method == "instance_robust":
        center = _masked_stat(Xw, obs, np.median)
        scale = _masked_stat(Xw, obs, lambda a: np.percentile(a, 75)) - _masked_stat(
            Xw, obs, lambda a: np.percentile(a, 25)
        )
    else:
        raise ValueError(
            f"unknown normalize method {method!r} -- expected 'warmup', "
            f"'instance_robust' or 'none'"
        )
    out = np.where(obs, (Xw - center) / np.maximum(scale, eps), 0.0).astype("float32")
    return out[0] if squeeze else out


class WindowBuilder:
    """
    Turns one variable-length instance into fixed-length, model-ready windows.

    WINDOW LENGTH -- the DL2.1b check has now been RUN (DATA_FINDINGS.md §4).
    Measured on the 14 real Event-9 instances that actually contain a transient
    phase: median transient duration 12,332 s (~3.4 h), min 182 s, max
    49,161 s. The old default of 60 samples at 1 Hz covered ~0.5% of a typical
    transient -- the model was asked to judge a 3-hour process from a
    one-minute snapshot.

    Fix, chosen because it leaves every model untouched: decimate the 1 Hz
    signal by `decimate` (default 30 -> one sample per 30 s, each the mean of
    the observed raw samples in that block) and keep `window_size` at 60
    SAMPLES. A window is then 30 minutes of real time while W stays 60, so the
    TCN's receptive field of 61 still covers the whole window and no
    architecture changes.

    Units, so nobody has to guess:
        decimate         raw 1 Hz samples per decimated sample
        window_size      decimated SAMPLES per window (not seconds)
        stride           decimated samples between window starts
        seconds covered  window_size * decimate
    """

    def __init__(
        self,
        window_size: int = 60,           # decimated samples -- see class docstring
        stride: int = 5,                 # decimated samples
        label_rule: str = "most_severe",  # "most_severe" | "final_timestep" | "majority"
        min_valid_frac: float = 0.5,
        decimate: int = 30,              # raw 1 Hz samples per decimated sample
        nan_label_policy: str = "drop",   # "drop" | "normal"
    ) -> None:
        if window_size < 1 or stride < 1 or decimate < 1:
            raise ValueError("window_size, stride and decimate must all be >= 1")
        if nan_label_policy not in ("drop", "normal"):
            raise ValueError(
                f"unknown nan_label_policy {nan_label_policy!r} -- expected 'drop' or 'normal'"
            )
        self.window_size = window_size
        self.stride = stride
        self.label_rule = label_rule
        self.min_valid_frac = min_valid_frac
        self.decimate = decimate
        self.nan_label_policy = nan_label_policy

    @property
    def window_seconds(self) -> int:
        return self.window_size * self.decimate

    def _decimate(self, instance: InstanceRecord, columns: list[str]):
        """
        Returns (values [C, Tb], raw_block_labels [Tb], block_has_nan_label [Tb]).

        A block's value per channel is the mean of its OBSERVED raw samples, or
        NaN if the block observed nothing. A block's label is the most severe
        raw code present in it, re-encoded back to a raw 3W code so
        label_window() still receives raw codes and keeps working for every
        label_rule -- the labeling logic is not reimplemented here.
        """
        d = self.decimate
        df = instance.df
        n_full = (len(df) // d) * d
        if n_full == 0:
            return (
                np.zeros((len(columns), 0)),
                np.zeros(0, dtype="int64"),
                np.zeros(0, dtype=bool),
            )

        V = df[columns].to_numpy(dtype="float64")[:n_full].reshape(-1, d, len(columns))
        observed = ~np.isnan(V)
        counts = observed.sum(axis=1)
        sums = np.nansum(V, axis=1)
        values = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)   # [Tb, C]

        raw = df["class"].to_numpy(dtype="float64")[:n_full].reshape(-1, d)
        has_nan = np.isnan(raw).any(axis=1)
        states = np.full(raw.shape, -1, dtype="int64")
        defined = ~np.isnan(raw)
        if defined.any():
            states[defined] = np.array(
                [_raw_label_to_state(int(v), instance.event_code) for v in raw[defined]],
                dtype="int64",
            )
        block_state = states.max(axis=1)
        block_state = np.where(block_state < 0, NORMAL, block_state)
        raw_block = np.select(
            [block_state == ESTABLISHED, block_state == TRANSIENT],
            [instance.event_code, TRANSIENT_OFFSET + instance.event_code],
            default=0,
        ).astype("int64")
        return values.T, raw_block, has_nan

    def _keep_indices(self, mask: np.ndarray, has_nan: np.ndarray, n_blocks: int) -> list[int]:
        """Window start indices surviving both drop rules. One implementation,
        so build_windows() and window_masks() can never disagree about which
        windows exist."""
        W, S = self.window_size, self.stride
        keep = []
        for s in range(0, n_blocks - W + 1, S):
            sl = slice(s, s + W)
            if self.nan_label_policy == "drop" and has_nan[sl].any():
                continue
            if mask[:, sl].mean() < self.min_valid_frac:
                continue
            keep.append(s)
        return keep

    def _prepare(self, instance: InstanceRecord, channel_means, normalize: str):
        columns = [c for c in instance.df.columns if c not in LABEL_COLUMNS]
        if not columns:
            raise ValueError(f"instance {instance.instance_id} has no variable columns")
        if "class" not in instance.df.columns:
            raise ValueError(
                f"instance {instance.instance_id} has no 'class' column to label from"
            )
        values, raw_block, has_nan = self._decimate(instance, columns)
        n_channels, n_blocks = values.shape
        if n_blocks < self.window_size:
            return None, None, None, None, n_channels
        filled, mask = mask_missing(
            values[None, ...], "ffill_then_mean", channel_means=channel_means
        )
        filled = normalize_instance(filled, mask, method=normalize)
        return filled[0], mask[0], raw_block, has_nan, n_channels

    def build_windows(
        self,
        instance: InstanceRecord,
        *,
        channel_means: np.ndarray | None = None,
        normalize: str = "warmup",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            X:               float32 [N, C, W]  -- CHANNELS-FIRST (Addendum A.2)
            y:               int64   [N]        -- NORMAL / TRANSIENT / ESTABLISHED
            well_ids:        [N]                -- instance.well_id, repeated
            window_end_time: float64 [N]        -- SECONDS from the instance's
                                                  first sample to the window's
                                                  last sample

        Two deliberate deviations from this function's original docstring, both
        to match code that already exists elsewhere:

        1. X is channels-first [N, C, W]. The original said
           (n_windows, window_size, n_channels), which contradicted Addendum
           A.2 and every consumer -- RollingFeatureExtractor.transform,
           TCN.forward, GRUClassifier.forward and make_fake_data.py all expect
           [N, C, W]. Returning the old layout would have been a silent
           transpose bug at four call sites.

        2. window_end_time is float64 SECONDS, not np.datetime64. alarm.py's
           `t`, lead_time(), select_threshold() and the fake cache are all in
           seconds-from-instance-start; datetime64 would need converting back
           at every one of them.

        Windows are dropped when the observed fraction falls below
        min_valid_frac (DL2.4 -- don't feed the model mostly-imputed data and
        call it signal) or, under nan_label_policy="drop", when any block in
        the window has an unlabeled (NaN) span. Every real Event-9 instance has
        such spans -- median 11.2% of samples, up to 50.4%
        (DATA_FINDINGS.md §5) -- and the project statement never says what to
        do with them. "drop" is the conservative reading: an unlabeled span is
        not evidence of normal operation, so counting it as Normal would pad
        the false-alarm denominator with time nobody vouched for.
        """
        filled, mask, raw_block, has_nan, n_channels = self._prepare(
            instance, channel_means, normalize
        )
        W = self.window_size
        empty = (
            np.zeros((0, n_channels, W), dtype="float32"),
            np.zeros(0, dtype="int64"),
            np.zeros(0, dtype=object),
            np.zeros(0, dtype="float64"),
        )
        if filled is None:
            return empty
        keep = self._keep_indices(mask, has_nan, mask.shape[-1])
        if not keep:
            return empty

        X = np.stack([filled[:, s : s + W] for s in keep]).astype("float32")
        y = np.array(
            [label_window(raw_block[s : s + W], instance.event_code, self.label_rule) for s in keep],
            dtype="int64",
        )
        well_ids = np.array([instance.well_id] * len(keep), dtype=object)
        # Last raw sample of the window, in seconds from the instance's start.
        t_end = np.array([(s + W) * self.decimate - 1 for s in keep], dtype="float64")
        return X, y, well_ids, t_end

    def window_masks(
        self,
        instance: InstanceRecord,
        *,
        channel_means: np.ndarray | None = None,
    ) -> np.ndarray:
        """The uint8 [N, C, W] mask matching build_windows()'s X, row for row.
        Split out so build_windows() keeps the 4-tuple return the data contract
        specifies while build_cache can still write `mask`."""
        filled, mask, _, has_nan, n_channels = self._prepare(instance, channel_means, "none")
        W = self.window_size
        if filled is None:
            return np.zeros((0, n_channels, W), dtype="uint8")
        keep = self._keep_indices(mask, has_nan, mask.shape[-1])
        if not keep:
            return np.zeros((0, n_channels, W), dtype="uint8")
        return np.stack([mask[:, s : s + W] for s in keep]).astype("uint8")


def onset_times(instance: InstanceRecord) -> dict[str, float]:
    """
    Seconds from an instance's first sample to its first TRANSIENT sample and
    to its first ESTABLISHED sample. NaN when that phase never occurs.

    WHICH ONE IS `failure_time` -- a decision, not a detail. The project
    statement (DL8.3) measures lead time against ESTABLISHED-blockage onset. On
    this dataset only 3 of 57 real instances ever reach blockage
    (DATA_FINDINGS.md §3), so that metric would be an n=3 number, undefined in
    most CV folds. The team's decision is therefore:

        failure_time := transient_onset   (14 real instances, 7 wells)

    which asks "how far ahead of the annotator's own onset mark do we fire?" --
    still a genuine early-warning question, and computable per fold.
    `blockage_onset` is carried alongside so the statement's original
    definition can still be reported as a secondary row with n=3 stated openly
    rather than quietly dropped.
    """
    raw = instance.df["class"].to_numpy(dtype="float64")
    ec = instance.event_code
    out = {"transient_onset": float("nan"), "blockage_onset": float("nan")}
    hit = np.flatnonzero(raw == TRANSIENT_OFFSET + ec)
    if hit.size:
        out["transient_onset"] = float(hit[0])
    hit = np.flatnonzero(raw == ec)
    if hit.size:
        out["blockage_onset"] = float(hit[0])
    return out


def normal_seconds(instance: InstanceRecord) -> float:
    """
    Seconds of this instance explicitly labeled normal (raw code 0).

    This is the false-alarm denominator (DL8.2), counted in WALL CLOCK rather
    than in windows: overlapping windows would count the same second
    window_size/stride times over, and DL8.2 warns specifically that the
    denominator must not depend on the windowing parameters. Unlabeled (NaN)
    spans are excluded -- nobody vouched for them being normal.
    """
    raw = instance.df["class"].to_numpy(dtype="float64")
    return float((raw == 0).sum())
