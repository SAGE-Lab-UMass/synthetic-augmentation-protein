#!/usr/bin/env python
"""
Run training for all drugs.

All arguments must be explicitly specified. This script passes them through
to train.py for each drug found in the data directory.

Usage examples:

    # Exp 0 — baseline, no augmentation, all folds
    for fold in 0 1 2 3 4; do
        python run_all.py \
            --data-dir ../../datasets/original \
            --output-dir ../../runs/exp0/resnet \
            --model-type resnet \
            --stem-channels 64 \
            --pretrain-epochs 25 \
            --finetune-epochs 15 \
            --batch-size 64 \
            --learning-rate 1e-4 \
            --finetune-lr 1e-5 \
            --weight-decay 1e-4 \
            --patience 5 \
            --min-improvement 0.001 \
            --val-fold $fold \
            --aug-mode none \
            --aug-multiplier 0.0 \
            --aug-target all \
            --mutate-frac 0.005 \
            --seed 42
    done

    # Exp 2 - only mutate susceptible only
    python run_all.py \
        --data-dir ../../datasets/original \
        --output-dir ../../runs_Feb19/exp2/resnet \
        --model-type resnet \
        --stem-channels 64 \
        --pretrain-epochs 25 \
        --finetune-epochs 15 \
        --batch-size 64 \
        --learning-rate 1e-4 \
        --finetune-lr 1e-5 \
        --weight-decay 1e-4 \
        --patience 5 \
        --min-improvement 0.001 \
        --val-fold 0 \
        --aug-mode online \
        --aug-multiplier 1 \
        --aug-target susceptible \
        --mutate-frac 0.005 \
        --seed 42

    # Exp 1 — online mutation, all classes
    python run_all.py \
        --data-dir ../../datasets/original \
        --output-dir ../../runs_Feb19/exp1/resnet \
        --model-type resnet \
        --stem-channels 64 \
        --pretrain-epochs 25 \
        --finetune-epochs 15 \
        --batch-size 64 \
        --learning-rate 1e-4 \
        --finetune-lr 1e-5 \
        --weight-decay 1e-4 \
        --patience 5 \
        --min-improvement 0.001 \
        --val-fold 0 \
        --aug-mode online \
        --aug-multiplier 1.0 \
        --aug-target all \
        --mutate-frac 0.005 \
        --seed 42

    # Exp 3 — offline ESM-filtered pool
    python run_all.py \
        --data-dir ../../datasets/original \
        --aug-data-dir ../../datasets/augmented \
        --output-dir ../../runs/exp3/resnet \
        --model-type resnet \
        --stem-channels 64 \
        --pretrain-epochs 25 \
        --finetune-epochs 15 \
        --batch-size 64 \
        --learning-rate 1e-4 \
        --finetune-lr 1e-5 \
        --weight-decay 1e-4 \
        --patience 5 \
        --min-improvement 0.001 \
        --val-fold 0 \
        --aug-mode offline \
        --aug-multiplier 1.0 \
        --aug-target all \
        --mutate-frac 0.005 \
        --seed 42
"""

import argparse
import os
import subprocess
import sys


def find_drug_datasets(data_dir: str) -> dict:
    """
    Find all drug datasets in data_dir.
    Accepts files ending in .csv (any name).
    Returns: {drug_name: csv_path}
    """
    datasets = {}
    for filename in os.listdir(data_dir):
        if not filename.endswith(".csv"):
            continue
        drug = filename.replace(".csv", "")
        datasets[drug] = os.path.join(data_dir, filename)
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Run train.py for all drugs in a directory"
    )

    # ---- Data ----
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing original drug CSV files",
    )
    parser.add_argument(
        "--aug-data-dir",
        default=None,
        help=(
            "Directory containing pre-generated augmented CSVs "
            "(required for --aug-mode offline). "
            "Expected filename pattern: {drug}_augmented.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory; results saved to {output_dir}/{drug}/val_{fold}/",
    )
    parser.add_argument(
        "--phenotype-col",
        default="Phenotype",
        help="Name of the phenotype column (default: Phenotype)",
    )
    parser.add_argument(
        "--val-fold",
        type=int,
        default=0,
        help="Validation fold index (default: 0)",
    )
    parser.add_argument("--nested-validation", action="store_true")
    parser.add_argument("--group-col", default="protein_id")
    parser.add_argument("--inner-val-fraction", type=float, default=0.1)

    # ---- Model ----
    parser.add_argument(
        "--model-type",
        default="resnet",
        choices=["resnet", "simple"],
        help="Model architecture (default: resnet)",
    )
    parser.add_argument(
        "--stem-channels",
        type=int,
        default=64,
        help="Number of stem layer channels (default: 64)",
    )

    # ---- Training schedule ----
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=50,
        help="Number of pretrain epochs (default: 50)",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=10,
        help="Number of finetune epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Pretrain learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-5,
        help="Finetune learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--no-pos-weight",
        action="store_true",
        help="Disable positive class weighting",
    )

    # ---- Early stopping ----
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience, 0 to disable (default: 5)",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.001,
        help="Minimum AUC improvement to reset patience (default: 0.001)",
    )

    # ---- Augmentation ----
    parser.add_argument(
        "--aug-mode",
        default="none",
        choices=["none", "online", "offline"],
        help="Augmentation mode: none / online / offline (default: none)",
    )
    parser.add_argument(
        "--aug-multiplier",
        type=float,
        default=1.0,
        help="Augmented/original ratio per epoch (default: 1.0)",
    )
    parser.add_argument(
        "--aug-target",
        default="all",
        choices=["all", "susceptible"],
        help="Which sequences to augment: all or susceptible (default: all)",
    )
    parser.add_argument(
        "--mutate-frac",
        type=float,
        default=0.005,
        help="Mutation fraction for online mode (default: 0.005)",
    )
    parser.add_argument("--offline-top-retention-rate", type=float, default=None)

    # ---- Misc ----
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # ---- Print all arguments ----
    print("\n" + "=" * 60)
    print("RUN CONFIGURATION")
    print("=" * 60)
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg:<25} = {value}")
    print("=" * 60 + "\n")

    # Validate offline mode requirement
    if args.aug_mode == "offline" and args.aug_data_dir is None:
        parser.error("--aug-data-dir is required when --aug-mode is offline")

    # Find all drug datasets
    datasets = find_drug_datasets(args.data_dir)
    if not datasets:
        print(f"No .csv files found in {args.data_dir}")
        return

    print("=" * 80)
    print(f"Found {len(datasets)} drug(s):")
    for drug, path in sorted(datasets.items()):
        print(f"  {drug}: {os.path.basename(path)}")
    print(f"\nModel:       {args.model_type}")
    print(f"Aug mode:    {args.aug_mode}  multiplier={args.aug_multiplier}  target={args.aug_target}")
    print(f"Val fold:    {args.val_fold}")
    print(f"Output root: {args.output_dir}")
    print("=" * 80)

    results = []
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")

    for drug, data_path in sorted(datasets.items()):
        print(f"\n{'=' * 80}")
        print(f"TRAINING: {drug.upper()}  (fold {args.val_fold})")
        print(f"{'=' * 80}")

        output_dir = os.path.join(args.output_dir, drug, f"val_{args.val_fold}")

        cmd = [
            sys.executable, train_script,
            "--data-path",        data_path,
            "--output-dir",       output_dir,
            "--phenotype-col",    args.phenotype_col,
            "--val-fold",         str(args.val_fold),
            "--model-type",       args.model_type,
            "--stem-channels",    str(args.stem_channels),
            "--pretrain-epochs",  str(args.pretrain_epochs),
            "--finetune-epochs",  str(args.finetune_epochs),
            "--batch-size",       str(args.batch_size),
            "--learning-rate",    str(args.learning_rate),
            "--finetune-lr",      str(args.finetune_lr),
            "--weight-decay",     str(args.weight_decay),
            "--patience",         str(args.patience),
            "--min-improvement",  str(args.min_improvement),
            "--aug-mode",         args.aug_mode,
            "--aug-multiplier",   str(args.aug_multiplier),
            "--aug-target",       args.aug_target,
            "--mutate-frac",      str(args.mutate_frac),
            "--seed",             str(args.seed),
            "--num-workers",      str(args.num_workers),
        ]

        if args.nested_validation:
            cmd += [
                "--nested-validation",
                "--group-col", args.group_col,
                "--inner-val-fraction", str(args.inner_val_fraction),
            ]
        if args.offline_top_retention_rate is not None:
            cmd += ["--offline-top-retention-rate", str(args.offline_top_retention_rate)]

        # Offline mode: resolve augmented CSV path for this drug
        if args.aug_mode == "offline":
            aug_data_path = os.path.join(args.aug_data_dir, f"{drug}_augmented.csv")
            if not os.path.exists(aug_data_path):
                print(f"  WARNING: Augmented file not found: {aug_data_path} — skipping {drug}")
                results.append((drug, "skipped"))
                continue
            cmd += ["--aug-data-path", aug_data_path]

        if args.no_pos_weight:
            cmd.append("--no-pos-weight")

        print(f"Command: {' '.join(cmd)}\n")

        if args.dry_run:
            print("(dry run — not executing)")
            results.append((drug, "skipped"))
            continue

        try:
            subprocess.run(cmd, check=True)
            results.append((drug, "success"))
            print(f"\n✓ {drug} completed successfully")
        except subprocess.CalledProcessError:
            print(f"\n✗ ERROR: Training failed for {drug}")
            results.append((drug, "failed"))

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    success_count = sum(1 for _, s in results if s == "success")
    print(f"Total: {len(results)},  Success: {success_count},  Failed/Skipped: {len(results) - success_count}\n")
    for drug, status in results:
        symbol = "✓" if status == "success" else "✗"
        print(f"  {symbol} {drug}: {status}")
    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
