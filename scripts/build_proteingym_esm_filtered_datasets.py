#!/usr/bin/env python3
"""Build ESM-filtered ProteinGym datasets for the MTB training pipeline.

This adapter reuses the existing MTB ESM scoring and filtering scripts on the
ProteinGym MTB-style CSVs produced by `build_proteingym_mtb_datasets.py`.

Pipeline:
  1. Rebuild the ProteinGym MTB-style CSV roots from the single-mutation source.
  2. Score the augmented ProteinGym CSV with ESM pseudo-perplexity.
  3. Optionally materialize a pilot subset from the scored rows.
  4. Generate a low-score filtered pool at threshold 0.01.
  5. Generate a top-50% filtered pool.
  6. Copy the filtered CSVs to the filename layout expected by `run_all.py`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_proteingym_mtb_datasets.py"
SCORE_SCRIPT = REPO_ROOT / "1_ESM_PIPELINE" / "2_scoring" / "score_augmented_with_esm.py"
LOW_FILTER_SCRIPT = REPO_ROOT / "1_ESM_PIPELINE" / "4_filtering" / "generate_low_score_datasets.py"
TOP_FILTER_SCRIPT = REPO_ROOT / "1_ESM_PIPELINE" / "4_filtering" / "generate_top_percent_datasets.py"

SOURCE_ROOT = REPO_ROOT / "datasets" / "proteingym_clinical_single_mut"
AUG_DIR = REPO_ROOT / "datasets" / "proteingym_mtb_augmented"
LOW_DIR = REPO_ROOT / "datasets" / "proteingym_mtb_augmented_0p01"
TOP_DIR = REPO_ROOT / "datasets" / "proteingym_mtb_augmented_top50"
PILOT_AUG_DIR = REPO_ROOT / "datasets" / "proteingym_mtb_augmented_25kpilot"
PILOT_LOW_DIR = REPO_ROOT / "datasets" / "proteingym_mtb_augmented_25kpilot_0p01"
PILOT_TOP_DIR = REPO_ROOT / "datasets" / "proteingym_mtb_augmented_25kpilot_top50"


def _run(cmd: list[str]) -> None:
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _normalize_parent_linkage(csv_path: Path) -> bool:
    """Ensure synthetic rows point back to their WT parent via Filename."""
    df = pd.read_csv(csv_path)
    required = {"aug_mutations", "Filename", "augmentation_parent_variant_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    synthetic_mask = df["aug_mutations"].fillna(0).astype(int) > 0
    if not synthetic_mask.any():
        return False

    desired = df.loc[synthetic_mask, "augmentation_parent_variant_id"].astype(str)
    current = df.loc[synthetic_mask, "Filename"].astype(str)
    changed = not desired.equals(current)
    if changed:
        df.loc[synthetic_mask, "Filename"] = desired
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Normalized synthetic Filename linkage in {csv_path}")
    return changed


def _stage_filtered_csv(filtered_dir: Path, source_name: str) -> Path:
    """Copy {source_name} to run_all.py's expected {drug}_augmented.csv name."""
    src = filtered_dir / source_name
    if not src.exists():
        raise FileNotFoundError(f"Expected filtered CSV not found: {src}")
    dst = filtered_dir / "proteingym_augmented.csv"
    shutil.copyfile(src, dst)
    return dst


def _materialize_pilot_subset(source_csv: Path, pilot_dir: Path, pilot_scored_rows: int) -> Path:
    """Create a pilot augmented CSV from the first scored rows with finalized score columns."""
    df = pd.read_csv(source_csv)
    if "PPPL_seq_1" not in df.columns:
        raise ValueError(f"Missing PPPL_seq_1 in {source_csv}")

    scored_df = df[df["PPPL_seq_1"].notna()].copy()
    if len(scored_df) < pilot_scored_rows:
        raise ValueError(
            f"Requested pilot_scored_rows={pilot_scored_rows}, but only {len(scored_df)} rows are scored."
        )

    pilot_df = scored_df.iloc[:pilot_scored_rows].copy()
    pilot_df["PPPL_mean"] = pilot_df["PPPL_seq_1"]

    wt_mask = pilot_df["aug_mutations"].fillna(0).astype(int) == 0
    wt_map = dict(zip(pilot_df.loc[wt_mask, "Filename"], pilot_df.loc[wt_mask, "PPPL_mean"]))
    keep_mask = wt_mask | pilot_df["Filename"].isin(wt_map)
    pilot_df = pilot_df.loc[keep_mask].copy()
    pilot_df["PPPL_ref"] = pilot_df["Filename"].map(wt_map)
    pilot_df["score"] = pilot_df["PPPL_mean"] - pilot_df["PPPL_ref"]
    pilot_df["norm_score"] = np.where(
        pilot_df["PPPL_ref"].replace(0.0, np.nan).notna(),
        pilot_df["score"] / pilot_df["PPPL_ref"].replace(0.0, np.nan),
        np.nan,
    )

    pilot_dir.mkdir(parents=True, exist_ok=True)
    pilot_csv = pilot_dir / "proteingym_augmented.csv"
    pilot_df.to_csv(pilot_csv, index=False)
    print(
        f"[INFO] Materialized ProteinGym pilot subset: rows={len(pilot_df)}, "
        f"wt={(pilot_df['aug_mutations'] == 0).sum()}, aug={(pilot_df['aug_mutations'] > 0).sum()}"
    )
    print(f"[INFO] Pilot CSV: {pilot_csv}")
    return pilot_csv


def _filter_and_stage(input_dir: Path, low_dir: Path, top_dir: Path, threshold: float, retention_rate: float) -> None:
    _run(
        [
            sys.executable,
            str(LOW_FILTER_SCRIPT),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(low_dir),
            "--threshold",
            str(threshold),
        ]
    )
    _stage_filtered_csv(low_dir, "proteingym_filtered.csv")

    _run(
        [
            sys.executable,
            str(TOP_FILTER_SCRIPT),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(top_dir),
            "--retention-rate",
            str(retention_rate),
        ]
    )
    _stage_filtered_csv(top_dir, "proteingym_filtered.csv")


def run(
    threshold: float,
    retention_rate: float,
    overwrite: bool,
    mask_batch_size: int,
    pilot_scored_rows: int | None,
) -> None:
    aug_csv = AUG_DIR / "proteingym_augmented.csv"
    if overwrite or not aug_csv.exists():
        build_cmd = [sys.executable, str(BUILD_SCRIPT)]
        if overwrite:
            build_cmd.append("--overwrite")
        _run(build_cmd)
    else:
        print(f"[INFO] Reusing existing ProteinGym augmented CSV: {aug_csv}")

    if not aug_csv.exists():
        raise FileNotFoundError(f"ProteinGym augmented CSV missing: {aug_csv}")
    _normalize_parent_linkage(aug_csv)

    score_cmd = [
        sys.executable,
        str(SCORE_SCRIPT),
        "--drug",
        "proteingym",
        "--data-root",
        str(AUG_DIR),
        "--mask-batch-size",
        str(mask_batch_size),
    ]
    if pilot_scored_rows is not None:
        score_cmd.extend(["--stop-after-scored-rows", str(pilot_scored_rows)])
    _run(score_cmd)

    if pilot_scored_rows is not None:
        _materialize_pilot_subset(aug_csv, PILOT_AUG_DIR, pilot_scored_rows)
        _filter_and_stage(PILOT_AUG_DIR, PILOT_LOW_DIR, PILOT_TOP_DIR, threshold, retention_rate)
        print("[INFO] ProteinGym ESM-filtered pilot dataset build complete.")
        print(f"[INFO] Pilot low-score output: {PILOT_LOW_DIR / 'proteingym_augmented.csv'}")
        print(f"[INFO] Pilot top-percent output: {PILOT_TOP_DIR / 'proteingym_augmented.csv'}")
        return

    _filter_and_stage(AUG_DIR, LOW_DIR, TOP_DIR, threshold, retention_rate)
    print("[INFO] ProteinGym ESM-filtered dataset build complete.")
    print(f"[INFO] Low-score output: {LOW_DIR / 'proteingym_augmented.csv'}")
    print(f"[INFO] Top-percent output: {TOP_DIR / 'proteingym_augmented.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ESM-filtered ProteinGym MTB-style datasets.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Low-score norm_score threshold (default: 0.01).",
    )
    parser.add_argument(
        "--retention-rate",
        type=float,
        default=50.0,
        help="Top-percent retention rate as fraction or percent (default: 50).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild the MTB-style ProteinGym roots before filtering.",
    )
    parser.add_argument(
        "--mask-batch-size",
        type=int,
        default=32,
        help="Masked-position batch size for ESM PPPL scoring (default: 32).",
    )
    parser.add_argument(
        "--pilot-scored-rows",
        type=int,
        default=None,
        help="If set, stop scoring once this many rows are scored and build pilot filtered datasets from that subset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        threshold=float(args.threshold),
        retention_rate=float(args.retention_rate),
        overwrite=bool(args.overwrite),
        mask_batch_size=int(args.mask_batch_size),
        pilot_scored_rows=args.pilot_scored_rows,
    )


if __name__ == "__main__":
    main()
