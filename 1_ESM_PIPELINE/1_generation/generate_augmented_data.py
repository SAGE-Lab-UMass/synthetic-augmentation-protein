#!/usr/bin/env python
"""
Generate augmented protein sequences for a given drug dataset.

This script:
  - Reads from: {input_dir}/{drug}.csv
  - Generates augmented samples (15 augmentation types × N samples per type)
  - Writes to: {output_dir}/{drug}_augmented.csv

Usage:
    python generate_augmented_data.py \
        --drug ethambutol \
        --input-dir ../../datasets/original \
        --output-dir ../../datasets/augmented \
        --num-aug-per-config 10
"""

import argparse
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from augmentation import SequenceAugmenter

VALID_AA = list("ACDEFGHIKLMNPQRSTVWY")


def find_sequence_columns(df: pd.DataFrame) -> List[str]:
    """Return all columns whose name starts with 'seq_'."""
    return [c for c in df.columns if c.startswith("seq_")]


def build_augmentation_configs(
    aug_types: Optional[List[str]] = None,
    mutate_frac: float = 0.005,
    mutate_randomly: bool = False,
) -> List[Dict]:
    """
    Create augmentation configs.
    
    Args:
        aug_types: List of augmentation types to use (e.g., ['D', 'IT', 'DITM']).
                  If None, uses all 15 types (all non-empty combinations of D/I/T/M).
        mutate_frac: Fraction of sequence to mutate (default 0.005 = 0.5%).
        mutate_randomly: If True, sample mutation count in [1, max] instead of fixed max.
    
    Returns:
        List of config dicts with 'name' and 'kwargs'
    """
    base_kwargs = dict(
        insert_max=20,
        shift_max=20,
        mutate_frac=mutate_frac,
        mutate_randomly=mutate_randomly,
    )

    flags = [
        ("D", "use_deletion"),
        ("I", "use_insertion"),
        ("T", "use_translocation"),
        ("M", "use_mutation"),
    ]

    # If specific types requested, only create those
    if aug_types is not None:
        configs = []
        for aug_name in aug_types:
            cfg = base_kwargs.copy()
            cfg.update(
                use_deletion=False,
                use_insertion=False,
                use_translocation=False,
                use_mutation=False,
            )
            
            # Enable operations based on letters in aug_name
            if 'D' in aug_name:
                cfg['use_deletion'] = True
            if 'I' in aug_name:
                cfg['use_insertion'] = True
            if 'T' in aug_name:
                cfg['use_translocation'] = True
            if 'M' in aug_name:
                cfg['use_mutation'] = True
            
            configs.append({"name": aug_name, "kwargs": cfg})
        
        return configs
    
    # Otherwise, generate all 15 combinations
    configs = []
    for mask in range(1, 16):
        cfg = base_kwargs.copy()
        cfg.update(
            use_deletion=False,
            use_insertion=False,
            use_translocation=False,
            use_mutation=False,
        )

        name_parts = []
        for bit_idx, (abbr, key) in enumerate(flags):
            if (mask >> bit_idx) & 1:
                cfg[key] = True
                name_parts.append(abbr)

        aug_name = "".join(name_parts)
        configs.append({"name": aug_name, "kwargs": cfg})

    return configs


def load_drug_dataset(drug: str, input_dir: str) -> pd.DataFrame:
    """
    Load the original CSV for a specific drug.
    
    Expected path: {input_dir}/{drug}.csv
    """
    csv_path = os.path.join(input_dir, f"{drug}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    print(f"[INFO] Loading: {csv_path}")
    return pd.read_csv(csv_path)


def build_joined_sequence(seq_dict: Dict[str, str], seq_cols: List[str]) -> str:
    """Create a stable joined sequence key for duplicate detection."""
    return "||".join(seq_dict.get(c, "") for c in seq_cols)


def mutate_one_position_per_column(
    row: pd.Series,
    seq_cols: List[str],
    rng: np.random.RandomState,
) -> Optional[Dict[str, str]]:
    """
    Apply exactly one point mutation in each sequence column.

    Unknown/non-standard symbols are never mutated.
    If any non-empty sequence column has no valid amino-acid position, returns None.
    """
    seq_map = {}
    n_mutated_cols = 0

    for col in seq_cols:
        seq = str(row[col]).strip() if pd.notna(row[col]) else ""
        seq_map[col] = seq
        if len(seq) == 0:
            continue

        valid_positions = [i for i, aa in enumerate(seq) if aa in VALID_AA]
        if not valid_positions:
            return None

        pos = int(rng.choice(valid_positions))
        aa_wt = seq[pos]
        aa_choices = [aa for aa in VALID_AA if aa != aa_wt]
        aa_new = str(rng.choice(aa_choices))

        seq_list = list(seq)
        seq_list[pos] = aa_new
        seq_map[col] = "".join(seq_list)
        n_mutated_cols += 1

    if n_mutated_cols == 0:
        return None
    return seq_map


def is_mutable_per_column(row: pd.Series, seq_cols: List[str]) -> bool:
    """
    Return True if every non-empty seq_* column has at least one standard AA position.
    """
    has_any_non_empty = False
    for col in seq_cols:
        seq = str(row[col]).strip() if pd.notna(row[col]) else ""
        if len(seq) == 0:
            continue
        has_any_non_empty = True
        if not any(ch in VALID_AA for ch in seq):
            return False
    return has_any_non_empty


def generate_augmented_data(
    drug: str,
    input_dir: str,
    output_dir: str,
    num_aug_per_config: int = 10,
    aug_types: Optional[List[str]] = None,
    mutate_frac: float = 0.005,
    mutate_randomly: bool = False,
    single_mutation_only: bool = False,
    max_regen_attempts: int = 200,
    seed: int = 42,
) -> str:
    """
    Generate augmented data for a single drug.

    For each original row:
      - write one "none" row (no augmentation)
      - for each augmentation config (15 in total),
          generate `num_aug_per_config` augmented rows.

    Returns:
      Path to the output augmented CSV.
    """
    df = load_drug_dataset(drug, input_dir)
    seq_cols = find_sequence_columns(df)

    if not seq_cols:
        raise ValueError(f"No seq_* columns found in dataset for drug '{drug}'")

    print(f"[INFO] Drug: {drug}")
    print(f"[INFO] Sequence columns: {seq_cols}")
    print(f"[INFO] num_aug_per_config = {num_aug_per_config}")
    print(f"[INFO] single_mutation_only = {single_mutation_only}")
    print(f"[INFO] max_regen_attempts = {max_regen_attempts}")

    if single_mutation_only:
        aug_configs = [{"name": "M1", "kwargs": {}}]
        print("[INFO] Using single-mutation mode (exactly 1 mutation per sample).")
    else:
        aug_configs = build_augmentation_configs(
            aug_types,
            mutate_frac=mutate_frac,
            mutate_randomly=mutate_randomly,
        )
        print(f"[INFO] Using {len(aug_configs)} augmentation types:")
        print("       " + ", ".join(cfg["name"] for cfg in aug_configs))

    rng = np.random.RandomState(seed)
    out_rows = []
    # Seed duplicate guard with all original joined sequences (global, not just processed-so-far).
    seen_joined = set()
    for _, base_row_src in df.iterrows():
        base_seq_map = {}
        for col in seq_cols:
            base_seq_map[col] = str(base_row_src[col]).strip() if pd.notna(base_row_src[col]) else ""
        seen_joined.add(build_joined_sequence(base_seq_map, seq_cols))
    n_duplicate_regens = 0
    n_duplicate_skips = 0
    n_no_mutable_skips = 0

    for idx, row in df.iterrows():
        filename = row.get("Filename")
        phenotype = row.get("Phenotype")
        is_val = row.get("is_val")

        # Original (no augmentation) — aug_mutations = 0
        base_row = {
            "Filename": filename,
            "Phenotype": phenotype,
            "is_val": is_val,
            "aug_mutations": 0,
        }
        for col in seq_cols:
            base_row[col] = str(row[col]).strip() if pd.notna(row[col]) else ""
        out_rows.append(base_row)
        seen_joined.add(build_joined_sequence(base_row, seq_cols))

        # Augmented versions
        row_mutable_for_single = is_mutable_per_column(row, seq_cols) if single_mutation_only else True
        for cfg in aug_configs:
            kwargs = cfg["kwargs"]

            for _ in range(num_aug_per_config):
                if single_mutation_only and not row_mutable_for_single:
                    n_no_mutable_skips += 1
                    continue

                accepted = False
                for _attempt in range(max_regen_attempts):
                    aug_row = {
                        "Filename": filename,
                        "Phenotype": phenotype,
                        "is_val": is_val,
                    }

                    if single_mutation_only:
                        seq_map = mutate_one_position_per_column(row, seq_cols, rng)
                        if seq_map is None:
                            n_no_mutable_skips += 1
                            continue
                        total_mutations = 0
                        for col in seq_cols:
                            aug_row[col] = seq_map[col]
                            seq_wt = str(row[col]).strip() if pd.notna(row[col]) else ""
                            if len(seq_wt) == 0:
                                continue
                            # Per-column mode: exactly one mutation in each non-empty column.
                            total_mutations += 1
                    else:
                        augmenter = SequenceAugmenter(**kwargs)
                        total_mutations = 0
                        for col in seq_cols:
                            seq_wt = str(row[col]).strip() if pd.notna(row[col]) else ""
                            if len(seq_wt) == 0:
                                aug_row[col] = ""
                                continue

                            aug_seed = int(rng.randint(0, 1e9))
                            seq_aug = augmenter.augment(seq_wt, seed=aug_seed)
                            aug_row[col] = seq_aug

                            # Count positional diffs only (length-changing ops still supported in legacy mode).
                            total_mutations += sum(a != b for a, b in zip(seq_wt, seq_aug))

                    joined = build_joined_sequence(aug_row, seq_cols)
                    if joined in seen_joined:
                        n_duplicate_regens += 1
                        continue

                    seen_joined.add(joined)
                    aug_row["aug_mutations"] = total_mutations
                    out_rows.append(aug_row)
                    accepted = True
                    break

                if not accepted:
                    n_duplicate_skips += 1

        if (idx + 1) % 50 == 0:
            print(f"[INFO] Processed {idx + 1} samples")

    out_df = pd.DataFrame(out_rows)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{drug}_augmented.csv")
    out_df.to_csv(out_path, index=False)

    print(f"[INFO] Saved augmented dataset to: {out_path}")
    print(f"[INFO] Total rows: {len(out_df)}")
    print(f"[INFO] Duplicate regenerations: {n_duplicate_regens}")
    print(f"[INFO] Duplicate skips (max attempts reached): {n_duplicate_skips}")
    if single_mutation_only:
        print(f"[INFO] Non-mutable regeneration retries: {n_no_mutable_skips}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented sequences for a drug dataset."
    )
    parser.add_argument(
        "--drug",
        type=str,
        required=True,
        help="Drug name (e.g., ethambutol).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing original CSV files (e.g., ../../datasets/original).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save augmented CSV files (e.g., ../../datasets/augmented).",
    )
    parser.add_argument(
        "--aug-types",
        type=str,
        nargs="+",
        default=None,
        help="Specific augmentation types to use (e.g., D IT DITM). If not specified, uses all 15 types.",
    )
    parser.add_argument(
        "--num-aug-per-config",
        type=int,
        default=10,
        help="Number of augmented samples to generate per augmentation type per original sample.",
    )
    parser.add_argument(
        "--mutate-frac",
        type=float,
        default=0.005,
        help="Fraction of sequence positions to mutate (default 0.005 = 0.5%%).",
    )
    parser.add_argument(
        "--mutate-randomly",
        action="store_true",
        default=True,
        help=(
            "If set, randomly sample mutation count in [1, max] instead of always "
            "mutating exactly max positions (at-most mode for controlled experiments)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--single-mutation-only",
        action="store_true",
        default=False,
        help=(
            "If set, ignore D/I/T configs and generate augmentations with exactly one "
            "point mutation per sequence column (seq_*). Unknown symbols are never mutated."
        ),
    )
    parser.add_argument(
        "--max-regen-attempts",
        type=int,
        default=200,
        help="Maximum regeneration attempts when duplicate joined sequence is found.",
    )

    args = parser.parse_args()

    generate_augmented_data(
        drug=args.drug,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_aug_per_config=args.num_aug_per_config,
        aug_types=args.aug_types,
        mutate_frac=args.mutate_frac,
        mutate_randomly=args.mutate_randomly,
        single_mutation_only=args.single_mutation_only,
        max_regen_attempts=args.max_regen_attempts,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
