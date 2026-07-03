#!/usr/bin/env python3
"""Build a small ProteinGym protein panel for fast MTB-pipeline experiments.

The adapted ProteinGym CSVs do not include a species column. ProteinGym folds are
protein-level here, so this script selects a fixed number of high-coverage
proteins from each fold and writes MTB-compatible single-file roots that keep all
five validation folds usable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIG = REPO_ROOT / "datasets" / "proteingym_mtb_original" / "proteingym.csv"
DEFAULT_AUG = REPO_ROOT / "datasets" / "proteingym_mtb_augmented" / "proteingym_augmented.csv"
DEFAULT_OUT_ORIG = REPO_ROOT / "datasets" / "proteingym_mtb_panel_original"
DEFAULT_OUT_AUG = REPO_ROOT / "datasets" / "proteingym_mtb_panel_augmented"


def _select_panel(orig_df: pd.DataFrame, proteins_per_fold: int, min_class_count: int) -> pd.DataFrame:
    stats = (
        orig_df.groupby(["is_val", "protein_id"], dropna=False)
        .agg(
            n=("protein_id", "size"),
            positives=("Phenotype", "sum"),
            folds=("is_val", "nunique"),
        )
        .reset_index()
    )
    stats["negatives"] = stats["n"] - stats["positives"]

    eligible = stats[(stats["positives"] >= min_class_count) & (stats["negatives"] >= min_class_count)].copy()
    if eligible.empty:
        raise ValueError("No proteins satisfy the requested min-class-count filter.")

    selected = []
    for fold in sorted(orig_df["is_val"].dropna().astype(int).unique()):
        fold_stats = eligible[eligible["is_val"].astype(int) == fold].sort_values(
            ["n", "positives", "protein_id"], ascending=[False, False, True]
        )
        if len(fold_stats) < proteins_per_fold:
            raise ValueError(
                f"Fold {fold} has only {len(fold_stats)} eligible proteins, "
                f"but proteins_per_fold={proteins_per_fold}."
            )
        selected.append(fold_stats.head(proteins_per_fold))

    return pd.concat(selected, ignore_index=True)


def run(
    original_csv: Path,
    augmented_csv: Path,
    out_original_dir: Path,
    out_augmented_dir: Path,
    proteins_per_fold: int,
    min_class_count: int,
    overwrite: bool,
) -> None:
    orig_df = pd.read_csv(original_csv)
    aug_df = pd.read_csv(augmented_csv)

    required = {"protein_id", "Phenotype", "is_val"}
    missing = required - set(orig_df.columns)
    if missing:
        raise ValueError(f"Original CSV missing columns: {sorted(missing)}")
    missing = required - set(aug_df.columns)
    if missing:
        raise ValueError(f"Augmented CSV missing columns: {sorted(missing)}")

    panel = _select_panel(orig_df, proteins_per_fold=proteins_per_fold, min_class_count=min_class_count)
    selected_proteins = set(panel["protein_id"].astype(str))

    out_original_dir.mkdir(parents=True, exist_ok=True)
    out_augmented_dir.mkdir(parents=True, exist_ok=True)
    out_orig_csv = out_original_dir / "proteingym.csv"
    out_aug_csv = out_augmented_dir / "proteingym_augmented.csv"
    manifest_csv = out_augmented_dir / "protein_panel_manifest.csv"

    if not overwrite and (out_orig_csv.exists() or out_aug_csv.exists() or manifest_csv.exists()):
        raise FileExistsError("Panel outputs already exist. Use --overwrite to replace them.")

    panel_orig = orig_df[orig_df["protein_id"].astype(str).isin(selected_proteins)].copy()
    panel_aug = aug_df[aug_df["protein_id"].astype(str).isin(selected_proteins)].copy()

    panel_orig.to_csv(out_orig_csv, index=False)
    panel_aug.to_csv(out_aug_csv, index=False)
    panel.sort_values(["is_val", "n"], ascending=[True, False]).to_csv(manifest_csv, index=False)

    print(f"[INFO] Selected proteins: {len(selected_proteins)}")
    print(f"[INFO] Original rows: {len(panel_orig)}")
    print(f"[INFO] Augmented rows including originals: {len(panel_aug)}")
    print("[INFO] Fold summary:")
    print(panel_orig.groupby("is_val").agg(rows=("protein_id", "size"), proteins=("protein_id", "nunique"), positives=("Phenotype", "sum")).to_string())
    print(f"[INFO] Wrote {out_orig_csv}")
    print(f"[INFO] Wrote {out_aug_csv}")
    print(f"[INFO] Wrote {manifest_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small ProteinGym protein panel for MTB-pipeline experiments.")
    parser.add_argument("--original-csv", type=Path, default=DEFAULT_ORIG)
    parser.add_argument("--augmented-csv", type=Path, default=DEFAULT_AUG)
    parser.add_argument("--out-original-dir", type=Path, default=DEFAULT_OUT_ORIG)
    parser.add_argument("--out-augmented-dir", type=Path, default=DEFAULT_OUT_AUG)
    parser.add_argument("--proteins-per-fold", type=int, default=3)
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        original_csv=args.original_csv,
        augmented_csv=args.augmented_csv,
        out_original_dir=args.out_original_dir,
        out_augmented_dir=args.out_augmented_dir,
        proteins_per_fold=args.proteins_per_fold,
        min_class_count=args.min_class_count,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
