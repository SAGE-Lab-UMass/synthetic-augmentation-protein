#!/usr/bin/env python
"""
Score augmented sequences with ESM PPPL and compute per-row scores.

This script:
  1. Reads: {input_dir}/{drug}_augmented.csv (or chunks if they exist)
  2. If chunks exist: merges them first
  3. Computes PPPL scores using ESM
  4. Uses rows with aug_name == "none" as WT reference
  5. Computes score = PPPL_mean(row) - PPPL_mean(WT for same Filename)
  6. Writes back to: {input_dir}/{drug}_augmented.csv (IN-PLACE, with PPPL columns added)
  7. Deletes chunk files after merging (if they exist)

Usage:
    python merge_and_score_pppl.py \
        --drug ethambutol \
        --input-dir ../../datasets/augmented
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def merge_chunks_if_exist(drug: str, input_dir: str) -> pd.DataFrame:
    """
    Check for chunk files. If they exist, merge them and return the merged DataFrame.
    If no chunks exist, read the main augmented file.
    
    Returns:
        (merged_df, chunk_files_list)
    """
    pattern = os.path.join(input_dir, f"{drug}_augmented_chunk_*.csv")
    chunk_files = sorted(glob.glob(pattern))

    if chunk_files:
        print(f"[INFO] Found {len(chunk_files)} chunk files, merging...")
        for f in chunk_files:
            print(f"   {f}")

        dfs = []
        for f in chunk_files:
            df_chunk = pd.read_csv(f)
            if "row_idx" not in df_chunk.columns:
                raise ValueError(f"Chunk file {f} missing 'row_idx' column.")
            dfs.append(df_chunk)

        full_df = pd.concat(dfs, axis=0, ignore_index=True)
        full_df = full_df.sort_values(by="row_idx").reset_index(drop=True)
        full_df = full_df.drop(columns=["row_idx"])

        print(f"[INFO] Merged {len(chunk_files)} chunks into {len(full_df)} rows")
        return full_df, chunk_files
    else:
        # No chunks, read the main file
        main_file = os.path.join(input_dir, f"{drug}_augmented.csv")
        if not os.path.exists(main_file):
            raise FileNotFoundError(f"File not found: {main_file}")

        print(f"[INFO] No chunks found, reading: {main_file}")
        return pd.read_csv(main_file), []


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute scores using WT rows as reference.
    
    score = PPPL_mean(row) - PPPL_mean(WT for same Filename)
    """
    if "PPPL_mean" not in df.columns:
        raise ValueError("PPPL_mean column not found. Run ESM scoring first!")

    if "aug_mutations" not in df.columns:
        raise ValueError("aug_mutations column not found.")

    wt_df = df[df["aug_mutations"] == 0]
    if wt_df.empty:
        raise ValueError("No WT rows (aug_mutations==0) found.")

    # Filename -> WT_PPPL_mean
    wt_map = {row["Filename"]: row["PPPL_mean"] for _, row in wt_df.iterrows()}

    scores = []
    for idx, row in df.iterrows():
        filename = row["Filename"]
        wt_mean = wt_map.get(filename, np.nan)
        pppl_mean = row["PPPL_mean"]

        if np.isnan(wt_mean) or np.isnan(pppl_mean):
            score = np.nan
        else:
            score = pppl_mean - wt_mean

        scores.append(score)

    df["score"] = scores
    print(f"[INFO] Computed scores for {len(df)} rows")

    return df


def merge_and_score(
    drug: str,
    input_dir: str,
) -> str:
    """
    Main function: merge chunks (if any), compute scores, save in-place, delete chunks.
    """
    # Merge chunks or read main file
    full_df, chunk_files = merge_chunks_if_exist(drug, input_dir)

    # Compute scores
    full_df = compute_scores(full_df)

    # Save in-place (overwrite the main file)
    out_path = os.path.join(input_dir, f"{drug}_augmented.csv")
    full_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved scored dataset to: {out_path}")

    # Delete chunk files
    if chunk_files:
        print(f"[INFO] Deleting {len(chunk_files)} chunk files...")
        for f in chunk_files:
            os.remove(f)
            print(f"   Deleted: {f}")

    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Merge chunks and score augmented sequences with PPPL."
    )
    ap.add_argument(
        "--drug",
        required=True,
        help="Drug name (e.g., ethambutol).",
    )
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing augmented CSV or chunks (e.g., ../../datasets/augmented).",
    )
    args = ap.parse_args()

    merge_and_score(drug=args.drug, input_dir=args.input_dir)


if __name__ == "__main__":
    main()