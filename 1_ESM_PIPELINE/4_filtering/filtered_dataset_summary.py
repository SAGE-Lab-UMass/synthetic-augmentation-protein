#!/usr/bin/env python
"""
Generate summary table after filtering.

Creates a table with columns:
  - Drug
  - Total Sample (original) 
  - Total Sample (filtered)
  - %Susceptible (original)
  - %Susceptible (filtered)

Usage:
    python filtered_dataset_summary.py \
        --original-dir ../../datasets/original \
        --filtered-dir ../../datasets/filtered \
        --output-dir ../../plots/filter_analysis
"""

import argparse
import os
import pandas as pd


def load_dataset(drug: str, input_dir: str, is_filtered: bool = False) -> pd.DataFrame:
    """Load dataset (original or filtered)."""
    if is_filtered:
        filename = f"{drug}_filtered.csv"
    else:
        filename = f"{drug}.csv"
    
    path = os.path.join(input_dir, filename)
    
    if not os.path.exists(path):
        return None
    
    return pd.read_csv(path)


def compute_stats(df: pd.DataFrame, phenotype_col: str = "Phenotype", is_filtered: bool = False) -> dict:
    """Compute total samples and % susceptible."""
    if df is None or len(df) == 0:
        return {"total": 0, "susceptible": 0, "pct_susceptible": 0.0}
    
    # For original: only count WT samples (aug_mutations == 0)
    # For filtered: count ALL samples (WT + augmented that passed filtering)
    if not is_filtered:
        # Original dataset: only WT samples
        df_wt = df[df["aug_mutations"] == 0] if "aug_mutations" in df.columns else df
    else:
        # Filtered dataset: all samples (WT + passed augmented)
        df_wt = df
    
    total = len(df_wt)
    
    if phenotype_col not in df_wt.columns:
        return {"total": total, "susceptible": 0, "pct_susceptible": 0.0}
    
    # Susceptible = Phenotype < 0.5 (resistant = >= 0.5)
    susceptible = (df_wt[phenotype_col] < 0.5).sum()
    pct = (susceptible / total * 100) if total > 0 else 0.0
    
    return {
        "total": total,
        "susceptible": susceptible,
        "pct_susceptible": pct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", required=True, help="Directory with original CSVs")
    parser.add_argument("--filtered-dir", required=True, help="Directory with filtered CSVs")
    parser.add_argument("--output-dir", required=True, help="Directory for output table")
    parser.add_argument("--phenotype-col", default="Phenotype", help="Phenotype column name")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("FILTERED DATASET SUMMARY")
    print("="*80)
    
    # Find all drugs
    drugs = set()
    
    # From original
    if os.path.exists(args.original_dir):
        for f in os.listdir(args.original_dir):
            if f.endswith(".csv"):
                drugs.add(f.replace(".csv", ""))
    
    # From filtered
    if os.path.exists(args.filtered_dir):
        for f in os.listdir(args.filtered_dir):
            if f.endswith("_filtered.csv"):
                drugs.add(f.replace("_filtered.csv", ""))
    
    drugs = sorted(drugs)
    print(f"\nFound {len(drugs)} drugs")
    
    results = []
    
    for drug in drugs:
        print(f"  Processing: {drug}")
        
        # Load original
        df_orig = load_dataset(drug, args.original_dir, is_filtered=False)
        stats_orig = compute_stats(df_orig, args.phenotype_col, is_filtered=False)
        
        # Load filtered
        df_filt = load_dataset(drug, args.filtered_dir, is_filtered=True)
        stats_filt = compute_stats(df_filt, args.phenotype_col, is_filtered=True)
        
        results.append({
            "Drug": drug.capitalize(),
            "Total Sample (original)": stats_orig["total"],
            "Total Sample (filtered)": stats_filt["total"],
            "%Susceptible (original)": round(stats_orig["pct_susceptible"], 2),
            "%Susceptible (filtered)": round(stats_filt["pct_susceptible"], 2),
        })
    
    # Create DataFrame
    df_summary = pd.DataFrame(results)
    
    # Save
    output_path = os.path.join(args.output_dir, "filtered_dataset_summary.csv")
    df_summary.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(df_summary.to_string(index=False))
    
    print("\n" + "="*80)
    print(f"Saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()