#!/usr/bin/env python
"""
Filter augmented samples by keeping the top lowest norm_score percentage.

Key logic:
  - Validation set = ONLY clean/WT samples (aug_mutations == 0) with original is_val
  - Kept augmented rows retain original is_val so fold exclusion can be done in training

For each drug:
  1. Load from: {input_dir}/{drug}_augmented.csv
  2. Rank augmented rows by norm_score ascending (lower is better)
  3. Keep top K rows where K = floor(retention_rate * n_aug_valid)
  4. Keep all WT rows (aug_mutations == 0) unchanged
  5. Save to: {output_dir}/{drug}_filtered.csv

Usage:
    python generate_top_percent_datasets.py \
        --input-dir ../../datasets/augmented_mutation \
        --output-dir ../../datasets/augmented_filtered_top50 \
        --retention-rate 0.5
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
    for drug, _ in sorted(augmented_files):
        print(f"   {drug}")

    return augmented_files


def compute_norm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized scores: norm_score = score / PPPL_ref."""
    df = df.copy()

    wt_df = df[df["aug_mutations"] == 0]
    if wt_df.empty:
        raise ValueError("No WT rows found (aug_mutations==0).")

    wt_map = dict(zip(wt_df["Filename"], wt_df["PPPL_mean"]))
    df["PPPL_ref"] = df["Filename"].map(wt_map)
    df["PPPL_ref"] = df["PPPL_ref"].replace(0.0, np.nan)
    df["norm_score"] = df["score"] / df["PPPL_ref"]
    return df


def normalize_retention_rate(value: float) -> float:
    """
    Normalize retention rate to [0,1].
    Accepts either fraction (0.4) or percent (40).
    """
    rate = float(value)
    if rate > 1.0:
        rate = rate / 100.0

    if rate < 0.0 or rate > 1.0:
        raise ValueError("retention-rate must be in [0,1] or [0,100].")
    return rate


def select_top_percent_augmented(df: pd.DataFrame, retention_rate: float) -> pd.DataFrame:
    """Keep top lowest norm_score percentage among augmented rows."""
    df_wt = df[df["aug_mutations"] == 0].copy()

    df_aug = df[df["aug_mutations"] > 0].copy()
    df_aug_valid = df_aug.dropna(subset=["norm_score"]).copy()

    n_aug_valid = len(df_aug_valid)
    if n_aug_valid == 0:
        return pd.concat([df_wt, df_aug_valid], ignore_index=True)

    n_keep = int(np.floor(n_aug_valid * retention_rate))
    if retention_rate > 0 and n_keep == 0:
        n_keep = 1

    df_aug_keep = df_aug_valid.nsmallest(n_keep, "norm_score").copy()

    return pd.concat([df_wt, df_aug_keep], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory with *_augmented.csv")
    ap.add_argument("--output-dir", required=True, help="Directory for filtered CSVs")
    ap.add_argument(
        "--retention-rate",
        type=float,
        required=True,
        help="Keep rate as fraction (e.g., 0.4) or percent (e.g., 40).",
    )
    args = ap.parse_args()

    retention_rate = normalize_retention_rate(args.retention_rate)
    os.makedirs(args.output_dir, exist_ok=True)

    augmented_files = find_augmented_files(args.input_dir)

    print(f"\n{'='*80}")
    print(f"Filtering with top retention rate = {retention_rate:.4f} ({retention_rate*100:.1f}%)")
    print(f"{'='*80}\n")

    summary = []

    for drug, path in sorted(augmented_files):
        print(f"\n[{drug.upper()}]")
        print("-" * 40)

        df = pd.read_csv(path)
        if "norm_score" not in df.columns:
            if "score" not in df.columns or "PPPL_mean" not in df.columns:
                print("  ✗ Missing columns. Skipping.")
                continue
            df = compute_norm_scores(df)

        n_wt = int((df["aug_mutations"] == 0).sum())
        n_aug_total = int((df["aug_mutations"] > 0).sum())
        n_aug_valid = int(((df["aug_mutations"] > 0) & df["norm_score"].notna()).sum())

        df_filtered = select_top_percent_augmented(df, retention_rate)

        n_wt_after = int((df_filtered["aug_mutations"] == 0).sum())
        n_aug_after = int((df_filtered["aug_mutations"] > 0).sum())

        pct_total = (n_aug_after / n_aug_total * 100.0) if n_aug_total > 0 else 0.0
        pct_valid = (n_aug_after / n_aug_valid * 100.0) if n_aug_valid > 0 else 0.0

        print(f"Before: WT={n_wt}, Aug_total={n_aug_total}, Aug_valid={n_aug_valid}")
        print(
            f"After:  WT={n_wt_after}, Aug_kept={n_aug_after} "
            f"(of total={pct_total:.1f}%, of valid={pct_valid:.1f}%)"
        )

        out_path = os.path.join(args.output_dir, f"{drug}_filtered.csv")
        df_filtered.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        summary.append(
            {
                "drug": drug,
                "wt": n_wt_after,
                "aug_before_total": n_aug_total,
                "aug_before_valid": n_aug_valid,
                "aug_after": n_aug_after,
                "retention_pct_total": pct_total,
                "retention_pct_valid": pct_valid,
                "target_retention_pct": retention_rate * 100.0,
            }
        )

    summary_df = pd.DataFrame(summary).sort_values("drug")
    rate_tag = f"{int(round(retention_rate * 100)):02d}pct"
    summary_path = os.path.join(args.output_dir, f"summary_top_{rate_tag}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()
