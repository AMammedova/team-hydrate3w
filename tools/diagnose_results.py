"""Root-cause diagnostic for the final results.

Separates the failures, which have opposite fixes:
  D1 is the score above chance at all?      lift = PR-AUC / base_rate
  D2 is the score degenerate?               weighted prior => huge ECE
  D3 does the operating point transfer?     val-selected threshold on test
  D4 what is the ceiling?                   threshold re-picked ON test
  D5 per-event separability                 peak pre-onset vs Normal p99
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.alarm import alarm_times

STEM = re.compile(r"^(?P<model>.+?)_(?P<condition>real_only|real_plus_sim)_fold(?P<fold>\d+)_seed(?P<seed>\d+)$")


def positive_score(probs: np.ndarray) -> np.ndarray:
    return probs[:, 1] + probs[:, 2]


def load_run(path: Path) -> dict:
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    # The XGBoost runner also writes calibrated arrays; use the same raw
    # positive score for every model so the comparison is like-for-like.
    d["score"] = positive_score(d["probs"])
    return d


def normal_hours(d: dict) -> float:
    """Normal operating hours represented by this split, summed per well."""
    return float(np.sum(d["normal_hours_value"]))


def split_instances(d: dict):
    """(event_instances, normal_instances) as lists of boolean masks."""
    events, normals = [], []
    for inst in np.unique(d["inst_id"]):
        m = d["inst_id"] == inst
        if not np.all(d["is_sim"][m] == 0):
            continue                       # simulated rows never scored
        (events if np.any(d["y_true"][m] != 0) else normals).append(m)
    return events, normals


def count_false_alarms(d: dict, normals, threshold: float, smooth: int, min_dur: float) -> int:
    n = 0
    for m in normals:
        order = np.argsort(d["t_end"][m])
        n += len(alarm_times(
            proba=d["score"][m][order], t=d["t_end"][m][order],
            smooth_window=smooth, threshold=threshold, min_duration=min_dur,
        ))
    return n


def count_events_flagged(d: dict, events, threshold: float, smooth: int, min_dur: float):
    """(n_flagged, n_events, lead_times) using alarms strictly before onset."""
    flagged, leads = 0, []
    for m in events:
        order = np.argsort(d["t_end"][m])
        t = d["t_end"][m][order]
        fail = float(np.unique(d["failure_time"][m])[0])
        onsets = alarm_times(
            proba=d["score"][m][order], t=t,
            smooth_window=smooth, threshold=threshold, min_duration=min_dur,
        )
        before = [o for o in onsets if o < fail]
        if before:
            flagged += 1
            leads.append(fail - min(before))
    return flagged, len(events), leads


def select_threshold(d: dict, normals, hours: float, target_far: float,
                     smooth: int, min_dur: float, grid: np.ndarray) -> float:
    """Largest FAR not exceeding the budget -> the lowest such threshold."""
    best_t, best_far = 1.0, -1.0
    for thr in grid:
        far = count_false_alarms(d, normals, thr, smooth, min_dur) / hours if hours > 0 else np.inf
        if far <= target_far and far > best_far:
            best_far, best_t = far, float(thr)
    return best_t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", default="results/model_outputs")
    ap.add_argument("--out", default="results/diagnosis.csv")
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--min-duration", type=float, default=0.0)
    ap.add_argument("--target-far", type=float, default=0.01)
    args = ap.parse_args()

    from sklearn.metrics import average_precision_score, roc_auc_score

    out_dir = Path(args.outputs_dir)
    rows, event_rows = [], []
    grid = np.linspace(0, 1, 201)

    for val_path in sorted(out_dir.glob("*_val.npz")):
        stem = val_path.stem[:-4]
        m = STEM.match(stem)
        if not m:
            continue
        test_path = out_dir / f"{stem}_test.npz"
        if not test_path.exists():
            continue
        model, cond, fold = m["model"], m["condition"], int(m["fold"])

        va, te = load_run(val_path), load_run(test_path)
        va_ev, va_no = split_instances(va)
        te_ev, te_no = split_instances(te)
        va_h, te_h = normal_hours(va), normal_hours(te)

        y_te = (te["y_true"] != 0).astype(int)
        base = float(y_te.mean())
        pr = float(average_precision_score(y_te, te["score"])) if len(np.unique(y_te)) > 1 else np.nan
        roc = float(roc_auc_score(y_te, te["score"])) if len(np.unique(y_te)) > 1 else np.nan

        # --- D3: frozen operating point -----------------------------------
        thr = select_threshold(va, va_no, va_h, args.target_far,
                               args.smooth_window, args.min_duration, grid)
        fa_val = count_false_alarms(va, va_no, thr, args.smooth_window, args.min_duration)
        fa_te = count_false_alarms(te, te_no, thr, args.smooth_window, args.min_duration)
        far_val = fa_val / va_h if va_h else np.nan
        far_te = fa_te / te_h if te_h else np.nan
        flag, n_ev, leads = count_events_flagged(te, te_ev, thr, args.smooth_window, args.min_duration)

        # --- D4: oracle threshold picked ON TEST --------------------------
        thr_or = select_threshold(te, te_no, te_h, args.target_far,
                                  args.smooth_window, args.min_duration, grid)
        flag_or, _, leads_or = count_events_flagged(
            te, te_ev, thr_or, args.smooth_window, args.min_duration)

        # --- D2: score degeneracy -----------------------------------------
        s = te["score"]
        rows.append(dict(
            model=model, condition=cond, fold=fold,
            test_base_rate=round(base, 5),
            test_pr_auc=round(pr, 4), lift=round(pr / base, 2) if base else np.nan,
            test_roc_auc=round(roc, 4),
            score_mean=round(float(s.mean()), 4), score_sd=round(float(s.std()), 4),
            score_p50=round(float(np.percentile(s, 50)), 4),
            score_p99=round(float(np.percentile(s, 99)), 4),
            thr_frozen=round(thr, 4), far_val=round(far_val, 4), far_test=round(far_te, 4),
            far_overshoot=round(far_te / args.target_far, 1) if np.isfinite(far_te) else np.nan,
            recall_frozen=round(flag / n_ev, 3) if n_ev else np.nan,
            thr_oracle=round(thr_or, 4),
            recall_oracle=round(flag_or / n_ev, 3) if n_ev else np.nan,
            n_test_events=n_ev, val_normal_h=round(va_h, 1), test_normal_h=round(te_h, 1),
        ))

        # --- D5: per-event separability -----------------------------------
        # The score any alarm must beat: the Normal-window quantile the FAR
        # budget permits in this fold.
        normal_scores = np.concatenate([te["score"][mm] for mm in te_no]) if te_no else np.array([])
        for mm in te_ev:
            order = np.argsort(te["t_end"][mm])
            t = te["t_end"][mm][order]
            sc = te["score"][mm][order]
            fail = float(np.unique(te["failure_time"][mm])[0])
            pre = sc[t < fail]
            event_rows.append(dict(
                model=model, condition=cond, fold=fold,
                well=int(np.unique(te["group"][mm])[0]),
                inst=int(np.unique(te["inst_id"][mm])[0]),
                n_windows=int(mm.sum()), n_pre_onset=int(len(pre)),
                peak_pre_onset=round(float(pre.max()), 4) if len(pre) else np.nan,
                normal_p99=round(float(np.percentile(normal_scores, 99)), 4) if len(normal_scores) else np.nan,
                normal_max=round(float(normal_scores.max()), 4) if len(normal_scores) else np.nan,
                separable_vs_p99=(bool(len(pre) and len(normal_scores)
                                       and pre.max() > np.percentile(normal_scores, 99))),
            ))

    df = pd.DataFrame(rows).sort_values(["model", "condition", "fold"])
    ev = pd.DataFrame(event_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    ev.to_csv(str(Path(args.out).with_name("diagnosis_events.csv")), index=False)

    pd.set_option("display.width", 250)
    print("=" * 118)
    print("D1/D2  RANKING QUALITY AND SCORE SHAPE  (test folds)")
    print("=" * 118)
    print(df[["model", "condition", "fold", "test_base_rate", "test_pr_auc", "lift",
              "test_roc_auc", "score_mean", "score_sd", "score_p99"]].to_string(index=False))

    print()
    print("=" * 118)
    print("D3/D4  THRESHOLD TRANSFER vs ORACLE CEILING   (target FAR = %.3f/h)" % args.target_far)
    print("=" * 118)
    print(df[["model", "condition", "fold", "val_normal_h", "test_normal_h", "thr_frozen",
              "far_val", "far_test", "far_overshoot", "recall_frozen",
              "thr_oracle", "recall_oracle", "n_test_events"]].to_string(index=False))

    print()
    print("=" * 118)
    print("AGGREGATE")
    print("=" * 118)
    agg = df.groupby(["model", "condition"]).agg(
        base=("test_base_rate", "mean"), pr_auc=("test_pr_auc", "mean"),
        lift=("lift", "mean"), roc=("test_roc_auc", "mean"),
        far_test=("far_test", "mean"), overshoot=("far_overshoot", "mean"),
        recall_frozen=("recall_frozen", "mean"), recall_oracle=("recall_oracle", "mean"),
    ).round(4)
    print(agg.to_string())

    if len(ev):
        print()
        print("=" * 118)
        print("D5  PER-EVENT SEPARABILITY (peak pre-onset score vs fold's Normal p99)")
        print("=" * 118)
        sep = ev.groupby(["model", "condition"])["separable_vs_p99"].agg(["sum", "size"])
        sep["frac"] = (sep["sum"] / sep["size"]).round(3)
        print(sep.to_string())
    print(f"\nwrote {args.out} and diagnosis_events.csv")


if __name__ == "__main__":
    main()
