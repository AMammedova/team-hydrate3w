"""Does per-instance score normalisation fix the alarm?

The alarm compares one GLOBAL threshold against every recording, assuming
scores are comparable across recordings. They are not: on fold 1 the GRU ranks
windows inside a recording at ROC-AUC 0.84 while its pooled ROC-AUC is 0.28.

Tested fix: standardise each instance against its own first `warmup` windows,
which is causal and deployable, and is the same argument
normalize_instance(method="warmup") already makes for the raw signal.

Arms: raw, z_warmup (mean/std), robust (median/IQR).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.alarm import alarm_times

STEM = re.compile(r"^(?P<model>.+?)_(?P<condition>real_only|real_plus_sim)_fold(?P<fold>\d+)_seed(?P<seed>\d+)$")
EPS = 1e-6


def _load(path: Path) -> dict:
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    d["score"] = d["probs"][:, 1] + d["probs"][:, 2]
    return d


def _instances(d: dict):
    """Ordered per-instance views of real rows only."""
    out = []
    for i in np.unique(d["inst_id"]):
        m = (d["inst_id"] == i) & (d["is_sim"] == 0)
        if not m.any():
            continue
        order = np.argsort(d["t_end"][m])
        out.append(dict(
            inst=int(i),
            t=d["t_end"][m][order],
            s=d["score"][m][order],
            y=(d["y_true"][m][order] != 0).astype(int),
            fail=float(np.unique(d["failure_time"][m])[0]),
            group=int(np.unique(d["group"][m])[0]),
        ))
    return out


def _transform(inst: dict, arm: str, warmup: int) -> np.ndarray | None:
    """Return the decision series, or None if this instance cannot supply a
    causal warm-up baseline that ends before the transient onset."""
    s, t, fail = inst["s"], inst["t"], inst["fail"]
    if arm == "raw":
        return s
    k = min(warmup, len(s))
    if k < 5:
        return None
    # The warm-up must be genuinely pre-event: if the onset falls inside it,
    # the baseline is contaminated by the very thing we are detecting.
    if np.isfinite(fail) and t[k - 1] >= fail:
        return None
    base = s[:k]
    if arm == "z_warmup":
        mu, sd = float(base.mean()), float(base.std())
        return (s - mu) / (sd + EPS)
    if arm == "robust":
        med = float(np.median(base))
        iqr = float(np.percentile(base, 75) - np.percentile(base, 25))
        return (s - med) / (iqr + EPS)
    raise ValueError(arm)


def _far(insts, series, thr, smooth, min_dur, hours) -> float:
    n = 0
    for inst, ser in zip(insts, series):
        if ser is None:
            continue
        n += len(alarm_times(proba=ser, t=inst["t"], smooth_window=smooth,
                             threshold=thr, min_duration=min_dur))
    return n / hours if hours > 0 else np.inf


def _recall(insts, series, thr, smooth, min_dur):
    flagged, leads, n = 0, [], 0
    for inst, ser in zip(insts, series):
        if ser is None:
            continue
        n += 1
        onsets = alarm_times(proba=ser, t=inst["t"], smooth_window=smooth,
                             threshold=thr, min_duration=min_dur)
        before = [o for o in onsets if o < inst["fail"]]
        if before:
            flagged += 1
            leads.append(inst["fail"] - min(before))
    return flagged, n, leads


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", default="results/model_outputs")
    ap.add_argument("--out", default="results/probe_score_normalisation.csv")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--min-duration", type=float, default=0.0)
    ap.add_argument("--target-far", type=float, default=0.01)
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    out_dir = Path(args.outputs_dir)
    rows = []

    for val_path in sorted(out_dir.glob("*_val.npz")):
        stem = val_path.stem[:-4]
        m = STEM.match(stem)
        test_path = out_dir / f"{stem}_test.npz"
        if not m or not test_path.exists():
            continue
        model, cond, fold = m["model"], m["condition"], int(m["fold"])

        va, te = _load(val_path), _load(test_path)
        va_i, te_i = _instances(va), _instances(te)
        va_h = float(np.sum(va["normal_hours_value"]))
        te_h = float(np.sum(te["normal_hours_value"]))

        for arm in ("raw", "z_warmup", "robust"):
            va_ser = [_transform(x, arm, args.warmup) for x in va_i]
            te_ser = [_transform(x, arm, args.warmup) for x in te_i]

            va_norm = [(x, s) for x, s in zip(va_i, va_ser) if s is not None and not x["y"].any()]
            te_norm = [(x, s) for x, s in zip(te_i, te_ser) if s is not None and not x["y"].any()]
            te_ev = [(x, s) for x, s in zip(te_i, te_ser) if s is not None and x["y"].any()]
            if not va_norm or not te_ev:
                continue

            # Threshold grid must span whatever scale this arm produces.
            allv = np.concatenate([s for _, s in va_norm])
            grid = (np.linspace(0, 1, 201) if arm == "raw"
                    else np.unique(np.percentile(allv, np.linspace(50, 100, 201))))

            best_thr, best_far = None, -1.0
            for thr in grid:
                far = _far([x for x, _ in va_norm], [s for _, s in va_norm],
                           thr, args.smooth_window, args.min_duration, va_h)
                if far <= args.target_far and far > best_far:
                    best_far, best_thr = far, float(thr)
            if best_thr is None:
                best_thr = float(grid[-1])

            far_te = _far([x for x, _ in te_norm], [s for _, s in te_norm],
                          best_thr, args.smooth_window, args.min_duration, te_h)
            flag, n_ev, leads = _recall([x for x, _ in te_ev], [s for _, s in te_ev],
                                        best_thr, args.smooth_window, args.min_duration)

            # Pooled ranking quality of the decision series itself.
            pooled_s = np.concatenate([s for _, s in te_ev] + [s for _, s in te_norm])
            pooled_y = np.concatenate([x["y"] for x, _ in te_ev] + [x["y"] for x, _ in te_norm])
            roc = (float(roc_auc_score(pooled_y, pooled_s))
                   if len(np.unique(pooled_y)) > 1 else np.nan)

            rows.append(dict(
                model=model, condition=cond, fold=fold, arm=arm,
                pooled_roc=round(roc, 3), thr=round(best_thr, 4),
                far_val=round(best_far, 4), far_test=round(far_te, 4),
                n_events=n_ev, n_flagged=flag,
                recall=round(flag / n_ev, 3) if n_ev else np.nan,
                median_lead_s=round(float(np.median(leads)), 1) if leads else np.nan,
            ))

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 220)
    print("=" * 104)
    print(f"PER-INSTANCE SCORE NORMALISATION  (warmup={args.warmup} windows, "
          f"target FAR={args.target_far}/h)")
    print("=" * 104)
    print(df.to_string(index=False))

    print()
    print("=" * 104)
    print("AGGREGATE BY ARM")
    print("=" * 104)
    agg = df.groupby("arm").agg(
        pooled_roc=("pooled_roc", "mean"),
        far_test=("far_test", "mean"),
        events=("n_events", "sum"),
        flagged=("n_flagged", "sum"),
    )
    agg["event_recall"] = (agg["flagged"] / agg["events"]).round(3)
    print(agg.round(4).to_string())

    print()
    print("BY MODEL x ARM (event recall over all test folds)")
    t = df.groupby(["model", "condition", "arm"]).agg(
        flagged=("n_flagged", "sum"), events=("n_events", "sum"),
        roc=("pooled_roc", "mean"), far=("far_test", "mean"))
    t["recall"] = (t["flagged"] / t["events"]).round(3)
    print(t.round(3).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
