#!/usr/bin/env python
"""
Filter low score augmented samples and generate filtered datasets.

Key logic:
  - Validation set = ONLY clean/WT samples (aug_mutations == 0) with their ORIGINAL is_val
  - Augmented rows keep their original is_val (0..4) so fold exclusion can be done in training

For each drug:
  1. Load from: {input_dir}/{drug}_augmented.csv
  2. Keep only augmented samples with norm_score <= threshold
  3. Keep all WT (aug_mutations == 0) samples with ORIGINAL is_val
  4. Keep original is_val for augmented samples
  5. Save to: {output_dir}/{drug}_filtered.csv

Usage:
    python generate_low_score_datasets.py \
        --input-dir ../../datasets/augmented \
        --output-dir ../../datasets/filtered \
        --threshold 0.01
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def find_augmented_files(input_dir: str) -> List[tuple]:
    """Discover all {drug}_augmented.csv files."""
    augmented_files = []

    for filename in os.listdir(input_dir):
        if filename.endswith("_augmented.csv"):
            drug_name = filename.replace("_augmented.csv", "")
            file_path = os.path.join(input_dir, filename)
            augmented_files.append((drug_name, file_path))

    if not augmented_files:
        raise FileNotFoundError(f"No '*_augmented.csv' files found in {input_dir}")

    print(f"[INFO] Found {len(augmented_files)} drugs:")
    for drug, _ in augmented_files:
        print(f"   {drug}")

    return augmented_files


def compute_norm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized scores: norm_score = score / PPPL_ref"""
    df = df.copy()

    wt_df = df[df["aug_mutations"] == 0]
    if wt_df.empty:
        raise ValueError("No WT rows found (aug_mutations==0).")

    wt_map = dict(zip(wt_df["Filename"], wt_df["PPPL_mean"]))
    df["PPPL_ref"] = df["Filename"].map(wt_map)
    df["PPPL_ref"] = df["PPPL_ref"].replace(0.0, np.nan)
    df["norm_score"] = df["score"] / df["PPPL_ref"]

    return df


def filter_low_score_samples(
    df: pd.DataFrame,
    threshold: float,
    min_threshold: float = None,
) -> pd.DataFrame:
    """Filter by threshold range and keep original is_val for all rows."""
    df_wt = df[df["aug_mutations"] == 0].copy()
    
    df_aug = df[df["aug_mutations"] > 0].copy()
    df_aug = df_aug.dropna(subset=["norm_score"])
    if min_threshold is None:
        df_aug_low = df_aug[df_aug["norm_score"] <= threshold].copy()
    else:
        df_aug_low = df_aug[
            (df_aug["norm_score"] >= min_threshold) & (df_aug["norm_score"] <= threshold)
        ].copy()
    
    return pd.concat([df_wt, df_aug_low], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory with augmented CSVs")
    ap.add_argument("--output-dir", required=True, help="Directory for filtered CSVs")
    ap.add_argument("--threshold", type=float, default=0.01, help="norm_score threshold")
    ap.add_argument(
        "--min-threshold",
        type=float,
        default=None,
        help="Optional lower bound for norm_score (inclusive).",
    )
    ap.add_argument("--phenotype-col", default="Phenotype")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    augmented_files = find_augmented_files(args.input_dir)

    print(f"\n{'='*80}")
    if args.min_threshold is None:
        print(f"Filtering with threshold <= {args.threshold}")
    else:
        print(f"Filtering with threshold in [{args.min_threshold}, {args.threshold}]")
    print(f"{'='*80}\n")

    summary = []

    for drug, path in sorted(augmented_files):
        print(f"\n[{drug.upper()}]")
        print("-" * 40)

        df = pd.read_csv(path)
        
        if "norm_score" not in df.columns:
            if "score" not in df.columns or "PPPL_mean" not in df.columns:
                print(f"  ✗ Missing columns. Skipping.")
                continue
            df = compute_norm_scores(df)

        n_wt = len(df[df["aug_mutations"] == 0])
        n_aug = len(df[df["aug_mutations"] > 0])

        df_filtered = filter_low_score_samples(
            df,
            threshold=args.threshold,
            min_threshold=args.min_threshold,
        )

        n_wt_after = len(df_filtered[df_filtered["aug_mutations"] == 0])
        n_aug_after = len(df_filtered[df_filtered["aug_mutations"] > 0])

        print(f"Before: WT={n_wt}, Aug={n_aug}")
        print(f"After:  WT={n_wt_after}, Aug={n_aug_after} ({n_aug_after/n_aug*100:.1f}%)")

        out_path = os.path.join(args.output_dir, f"{drug}_filtered.csv")
        df_filtered.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        summary.append({
            "drug": drug,
            "wt": n_wt_after,
            "aug_before": n_aug,
            "aug_after": n_aug_after,
            "retention_pct": n_aug_after / n_aug * 100 if n_aug > 0 else 0,
        })

    summary_df = pd.DataFrame(summary)
    if args.min_threshold is None:
        summary_name = f"summary_threshold_{args.threshold}.csv"
    else:
        summary_name = f"summary_threshold_{args.min_threshold}_to_{args.threshold}.csv"
    summary_path = os.path.join(args.output_dir, summary_name)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()
