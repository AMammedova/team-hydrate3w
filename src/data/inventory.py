"""
Module 1 — Data Loading & Instance Inventory (W1.3)
See DL_Project_Statement_Hydrate3W.docx, section 4, for the full contract
and rationale (DL1.1-DL1.4).

Enumerates every Event-9 (real + simulated + hand-drawn, per your track
scope) and Normal-Operation instance from a 3W-formatted dataset directory,
and exposes them as a uniform in-memory structure regardless of filename
quirks in a given 3W release.

------------------------------------------------------------------------
VERIFY AGAINST YOUR ACTUAL DOWNLOAD BEFORE TRUSTING FOLD ASSIGNMENTS
------------------------------------------------------------------------
The 3W Dataset's own documentation states plainly that "the filename
reveals its source" but does not pin down an exact regex, and the exact
pattern has shifted across 3W Dataset versions. This module ships with a
best-effort, defensive parser (see `_infer_source_and_well` below) plus a
CLI `--verify` mode that prints every detected (filename -> source,
well_id) pair for a class folder so you can eyeball it against what you
actually downloaded in five minutes, *before* running any CV fold on it.
If the patterns below don't match your download, adjust
`_infer_source_and_well` -- do not silently proceed on a guess.

    python -m src.data.inventory --root data/3W/dataset --event 9 --verify
------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Per the 3W Dataset's own labeling convention: an undesirable event's
# steady-state label is its event code; its transient label is
# 100 + event code (see dataset.ini). Centralized in src/contract.py --
# imported here, not redefined, so this and every other module agree.
from src.contract import TRANSIENT_OFFSET, NORMAL as NORMAL_LABEL

# Non-variable columns present in 3W parquet files.
NON_VARIABLE_COLUMNS = {"class", "state"}

# Best-effort filename source detection. Order matters: check the more
# specific tokens before falling back to "real".
_SIMULATED_RE = re.compile(r"simulated", re.IGNORECASE)
_DRAWN_RE = re.compile(r"drawn", re.IGNORECASE)

# Best-effort well-id extraction for REAL instances: 3W real filenames
# have historically looked like "<WELL-TOKEN>_<14-digit-timestamp>.parquet"
# e.g. "WELL-00019_20140124093303.parquet". Adjust if your download
# differs (see module docstring).
_REAL_WELL_RE = re.compile(r"^(?P<well>[A-Za-z0-9\-]+?)_(?P<ts>\d{8,14})$")


@dataclass
class InstanceRecord:
    """One 3W instance (one file), loaded and normalized."""

    instance_id: str          # unique across the whole inventory (filename stem)
    well_id: str               # real: parsed well token; simulated/drawn: unique pseudo-group
    source: str                 # "real" | "simulated" | "drawn"
    event_code: int             # e.g. 9 for Hydrate in Service Line
    filepath: Path
    df: pd.DataFrame            # DatetimeIndex; N variable columns + "class" (+ "state" if present)
    n_timesteps: int = field(init=False)
    duration_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.n_timesteps = len(self.df)
        if self.n_timesteps >= 2 and isinstance(self.df.index, pd.DatetimeIndex):
            self.duration_seconds = (
                self.df.index[-1] - self.df.index[0]
            ).total_seconds()
        else:
            self.duration_seconds = float("nan")

    @property
    def has_transient(self) -> bool:
        """True if any row is labeled with this event's transient code."""
        if "class" not in self.df.columns:
            return False
        transient_label = TRANSIENT_OFFSET + self.event_code
        return bool((self.df["class"] == transient_label).any())

    @property
    def has_established(self) -> bool:
        """True if any row is labeled with this event's steady-state code."""
        if "class" not in self.df.columns:
            return False
        return bool((self.df["class"] == self.event_code).any())

    def variable_columns(self) -> list[str]:
        return [c for c in self.df.columns if c not in NON_VARIABLE_COLUMNS]

    def missing_fraction(self) -> pd.Series:
        """Per-variable fraction of NaN rows."""
        cols = self.variable_columns()
        if not cols:
            return pd.Series(dtype=float)
        return self.df[cols].isna().mean()


def _infer_source_and_well(stem: str) -> tuple[str, str]:
    """
    Given a filename stem (no extension), return (source, well_id).

    - source: "real" | "simulated" | "drawn"
    - well_id: a grouping key safe to use for well-level CV splitting.
      Simulated/drawn instances each get their OWN unique pseudo-group
      (e.g. "SIM-<stem>") rather than a shared group, since they do not
      correspond to a single real physical well and must never be treated
      as if several of them "are" the same well for grouping purposes.

    See the module docstring: verify this against your actual filenames
    with `--verify` before trusting it.
    """
    if _SIMULATED_RE.search(stem):
        return "simulated", f"SIM-{stem}"
    if _DRAWN_RE.search(stem):
        return "drawn", f"DRAWN-{stem}"

    m = _REAL_WELL_RE.match(stem)
    if m:
        return "real", m.group("well")

    # Fallback: couldn't confidently parse a well token out of a
    # presumed-real filename. Don't guess silently -- log loudly and
    # group this instance under its own singleton "well" so it can
    # never accidentally merge with a different real well.
    logger.warning(
        "Could not parse a well_id from filename stem %r using the "
        "expected real-instance pattern. Treating it as its own "
        "singleton well group (UNKNOWN-%s). Run with --verify and "
        "adjust _REAL_WELL_RE if this happens for many files.",
        stem, stem,
    )
    return "real", f"UNKNOWN-{stem}"


class ThreeWDataset:
    """
    Enumerates 3W instances for a given event code from a 3W-formatted
    dataset directory (the directory that directly contains the
    numbered class subfolders, e.g. ".../3W/dataset").
    """

    def __init__(
        self,
        root_dir: str | Path,
        event_code: int = 9,
        include_real: bool = True,
        include_simulated: bool = True,
        include_normal: bool = True,
        include_drawn: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.event_code = event_code
        self.include_real = include_real
        self.include_simulated = include_simulated
        self.include_normal = include_normal
        self.include_drawn = include_drawn

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"root_dir {self.root_dir} does not exist. Run "
                f"data/download_data.sh first, or pass the correct path "
                f"to the 3W 'dataset' folder."
            )

    def _class_dir(self, label: int) -> Path:
        return self.root_dir / str(label)

    def _iter_parquet_files(self, label: int):
        class_dir = self._class_dir(label)
        if not class_dir.exists():
            logger.warning("Class directory %s not found, skipping.", class_dir)
            return
        yield from sorted(class_dir.glob("*.parquet"))

    def _load_one(self, filepath: Path, event_code: int) -> InstanceRecord:
        df = pd.read_parquet(filepath)
        source, well_id = _infer_source_and_well(filepath.stem)
        return InstanceRecord(
            instance_id=filepath.stem,
            well_id=well_id,
            source=source,
            event_code=event_code,
            filepath=filepath,
            df=df,
        )

    def load_instances(self) -> list[InstanceRecord]:
        """
        Load every requested instance for self.event_code (and, if
        include_normal, class 0) into InstanceRecords.
        """
        records: list[InstanceRecord] = []

        for filepath in self._iter_parquet_files(self.event_code):
            rec = self._load_one(filepath, self.event_code)
            if rec.source == "real" and not self.include_real:
                continue
            if rec.source == "simulated" and not self.include_simulated:
                continue
            if rec.source == "drawn" and not self.include_drawn:
                continue
            records.append(rec)

        if self.include_normal:
            for filepath in self._iter_parquet_files(NORMAL_LABEL):
                rec = self._load_one(filepath, self.event_code)
                # Normal-operation instances are real by construction in
                # the 3W dataset; still run through the same detector for
                # consistency and to catch surprises.
                records.append(rec)

        logger.info("Loaded %d instances (event_code=%d).", len(records), self.event_code)
        return records

    def well_ids(self, source: str | None = None) -> np.ndarray:
        records = self.load_instances()
        if source is not None:
            records = [r for r in records if r.source == source]
        return np.array(sorted({r.well_id for r in records}))

    def summary(self) -> pd.DataFrame:
        """
        One row per instance: instance_id, well_id, source, event_code,
        n_timesteps, duration_seconds, has_transient, has_established,
        plus one missing_frac__<var> column per raw variable.

        This is the table that turns the floor-justification claim in
        the project statement (2.2) and the real-well-count question
        (DL1.2) from assumptions into verified numbers -- run this
        before anything else.
        """
        records = self.load_instances()
        rows = []
        for r in records:
            row = {
                "instance_id": r.instance_id,
                "well_id": r.well_id,
                "source": r.source,
                "event_code": r.event_code,
                "n_timesteps": r.n_timesteps,
                "duration_seconds": r.duration_seconds,
                "has_transient": r.has_transient,
                "has_established": r.has_established,
            }
            for var, frac in r.missing_fraction().items():
                row[f"missing_frac__{var}"] = frac
            rows.append(row)
        return pd.DataFrame(rows)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Path to the 3W 'dataset' directory")
    parser.add_argument("--event", type=int, default=9, help="Event code (9 = Hydrate in Service Line)")
    parser.add_argument("--verify", action="store_true",
                         help="Print detected (filename -> source, well_id) pairs instead of a full summary")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ds = ThreeWDataset(args.root, event_code=args.event)

    if args.verify:
        for label in (0, args.event):
            print(f"\n--- class {label} ---")
            for fp in ds._iter_parquet_files(label):
                source, well_id = _infer_source_and_well(fp.stem)
                print(f"{fp.name:60s} -> source={source:10s} well_id={well_id}")
        return

    summary = ds.summary()
    print(summary.to_string(index=False))
    print("\nCounts by source:")
    print(summary["source"].value_counts().to_string())
    print(f"\nDistinct wells (real, event {args.event} + normal): "
          f"{summary.loc[summary['source'] == 'real', 'well_id'].nunique()}")
    print("\nTimestep count distribution for real, event-labeled instances:")
    real_event = summary[(summary["source"] == "real") & (summary["has_transient"] | summary["has_established"])]
    if len(real_event):
        print(real_event["n_timesteps"].describe().to_string())
    else:
        print("(no real event-labeled instances found -- check --root and --event)")


if __name__ == "__main__":
    _cli()
