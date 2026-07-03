#!/usr/bin/env python
"""
Training Script for Drug Resistance Prediction

Two-stage training:
  1. Pretrain: Train on original + augmented samples (aug_mode controls augmentation)
  2. Finetune: Train on original samples only (no augmentation)

Augmentation modes (--aug-mode):
  none:    No augmentation — train on original sequences only (Exp 0)
  online:  On-the-fly mutation per epoch (Exp 1, 2)
  offline: Randomly sample from pre-generated augmented pool per epoch (Exp 3)

All arguments must be explicitly specified when running.

Usage examples:
    # Exp 0 — baseline, no augmentation
    python train.py \
        --data-path ../../datasets/original/ethambutol.csv \
        --output-dir ../../runs/exp0/ethambutol/val_0 \
        --model-type resnet \
        --pretrain-epochs 25 \
        --finetune-epochs 15 \
        --batch-size 64 \
        --learning-rate 1e-4 \
        --finetune-lr 1e-5 \
        --weight-decay 1e-4 \
        --patience 5 \
        --min-improvement 0.001 \
        --val-fold 0 \
        --aug-mode none \
        --aug-multiplier 0.0 \
        --aug-target all \
        --mutate-frac 0.005 \
        --seed 42

    # Exp 1 — online mutation, all classes
    python train.py \
        --data-path ../../datasets/original/ethambutol.csv \
        --output-dir ../../runs/exp1/ethambutol/val_0 \
        --model-type resnet \
        --pretrain-epochs 50 \
        --finetune-epochs 10 \
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

    # Exp 2 — online mutation, susceptible only
    python train.py \
        --data-path ../../datasets/original/ethambutol.csv \
        --output-dir ../../runs/exp2/ethambutol/val_0 \
        ... \
        --aug-mode online \
        --aug-multiplier 1.0 \
        --aug-target susceptible \
        --mutate-frac 0.005

    # Exp 3 — offline ESM-filtered pool
    python train.py \
        --data-path ../../datasets/original/ethambutol.csv \
        --aug-data-path ../../datasets/augmented/ethambutol_augmented.csv \
        --output-dir ../../runs/exp3/ethambutol/val_0 \
        ... \
        --aug-mode offline \
        --aug-multiplier 1.0 \
        --aug-target all \
        --mutate-frac 0.005
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_module import (
    SequenceDataset,
    build_eval_loader,
    build_train_loader,
    build_val_loader,
    compute_max_lengths,
)
from loss import WeightedBCELoss, compute_pos_weight
from models import create_model


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    stage: str = "",
) -> float:
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0

    desc = f"{stage} Epoch {epoch + 1}/{total_epochs}" if stage else f"Epoch {epoch + 1}/{total_epochs}"
    pbar = tqdm(dataloader, desc=desc)

    for sequences, targets in pbar:
        sequences = sequences.to(device)
        targets = targets.to(device)

        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        logits = model(sequences)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model, return dict with 'auc'."""
    model.eval()
    all_targets, all_probs = [], []

    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            logits = model(sequences)
            probs = torch.sigmoid(logits)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    targets_np = np.concatenate(all_targets).flatten()
    probs_np = np.concatenate(all_probs).flatten()
    binary_targets = (targets_np >= 0.5).astype(float)

    try:
        auc = roc_auc_score(binary_targets, probs_np)
    except ValueError:
        auc = 0.5

    return {"auc": float(auc)}


def find_sequence_columns(df: pd.DataFrame) -> list:
    """Find all columns whose name starts with 'seq_'."""
    seq_cols = [col for col in df.columns if col.lower().startswith("seq_")]
    if not seq_cols:
        seq_cols = [col for col in df.columns if "sequence" in col.lower()]
    return seq_cols


def build_nested_protein_split(
    orig_df: pd.DataFrame,
    outer_fold: int,
    group_col: str,
    inner_val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve proteins from every outer-training fold for early stopping."""
    if group_col not in orig_df.columns:
        raise ValueError(f"Nested validation requires group column '{group_col}'")
    if not 0.0 < inner_val_fraction < 1.0:
        raise ValueError("--inner-val-fraction must be strictly between 0 and 1")

    group_fold_counts = orig_df.groupby(group_col)["is_val"].nunique()
    inconsistent = group_fold_counts[group_fold_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            f"{len(inconsistent)} groups occur in multiple outer folds; "
            "nested protein-disjoint evaluation would be invalid"
        )

    outer_test_df = orig_df[orig_df["is_val"] == outer_fold].copy()
    outer_train_df = orig_df[orig_df["is_val"] != outer_fold].copy()
    inner_group_frames = []

    for source_fold in sorted(outer_train_df["is_val"].dropna().astype(int).unique()):
        groups = sorted(
            outer_train_df.loc[
                outer_train_df["is_val"].astype(int) == source_fold, group_col
            ].dropna().astype(str).unique()
        )
        if len(groups) < 2:
            raise ValueError(
                f"Outer-training fold {source_fold} has fewer than two '{group_col}' groups"
            )
        rng = np.random.RandomState(seed + outer_fold * 10_000 + source_fold)
        shuffled = np.asarray(groups, dtype=object)
        rng.shuffle(shuffled)
        n_inner = max(1, int(np.ceil(len(groups) * inner_val_fraction)))
        n_inner = min(n_inner, len(groups) - 1)
        inner_group_frames.append(
            pd.DataFrame({group_col: shuffled[:n_inner], "source_outer_fold": source_fold})
        )

    inner_groups_df = pd.concat(inner_group_frames, ignore_index=True)
    inner_groups = set(inner_groups_df[group_col].astype(str))
    group_values = outer_train_df[group_col].astype(str)
    inner_val_df = outer_train_df[group_values.isin(inner_groups)].copy()
    train_df = outer_train_df[~group_values.isin(inner_groups)].copy()

    split_manifest = pd.concat(
        [
            train_df[[group_col, "is_val"]].drop_duplicates().assign(split="train"),
            inner_val_df[[group_col, "is_val"]].drop_duplicates().assign(split="inner_val"),
            outer_test_df[[group_col, "is_val"]].drop_duplicates().assign(split="outer_test"),
        ],
        ignore_index=True,
    ).rename(columns={"is_val": "source_outer_fold"})
    split_manifest = split_manifest.sort_values(
        ["split", "source_outer_fold", group_col]
    ).reset_index(drop=True)

    return (
        train_df.reset_index(drop=True),
        inner_val_df.reset_index(drop=True),
        outer_test_df.reset_index(drop=True),
        split_manifest,
    )


def select_fold_local_top_percent(
    aug_pool_df: pd.DataFrame, retention_rate: float
) -> tuple[pd.DataFrame, float]:
    """Rank only the current outer fold's training augmentation pool."""
    if not 0.0 < retention_rate <= 1.0:
        raise ValueError("--offline-top-retention-rate must be in (0, 1]")
    if "norm_score" not in aug_pool_df.columns:
        raise ValueError(
            "Fold-local top-percent filtering requires a 'norm_score' column "
            "in the unfiltered scored augmentation CSV"
        )
    valid = aug_pool_df.dropna(subset=["norm_score"]).copy()
    if valid.empty:
        raise ValueError("No augmented rows with valid norm_score remain in this training fold")
    n_keep = max(1, int(np.floor(len(valid) * retention_rate)))
    kept = valid.nsmallest(n_keep, "norm_score").copy().reset_index(drop=True)
    return kept, float(kept["norm_score"].max())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train drug resistance prediction model with controlled augmentation"
    )

    # ---- Data ----
    parser.add_argument(
        "--data-path", "-d",
        required=True,
        help="Path to original dataset CSV (no aug_mutations column needed)",
    )
    parser.add_argument(
        "--aug-data-path",
        default=None,
        help="Path to pre-generated augmented CSV (required for --aug-mode offline)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for model checkpoints and logs",
    )
    parser.add_argument(
        "--phenotype-col",
        default="Phenotype",
        help="Name of phenotype column (default: Phenotype)",
    )
    parser.add_argument(
        "--val-fold",
        type=int,
        default=0,
        help="Outer held-out fold index (default: 0)",
    )
    parser.add_argument(
        "--nested-validation",
        action="store_true",
        help="Use inner protein groups for early stopping and outer fold only for final test",
    )
    parser.add_argument("--group-col", default="protein_id")
    parser.add_argument("--inner-val-fraction", type=float, default=0.1)

    # ---- Model ----
    parser.add_argument(
        "--model-type", "-m",
        choices=["resnet", "simple"],
        default="resnet",
        help="Model architecture: resnet (deep) or simple (shallow) (default: resnet)",
    )
    parser.add_argument(
        "--stem-channels",
        type=int,
        default=64,
        help="Number of channels in stem layer (default: 64)",
    )

    # ---- Training schedule ----
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=50,
        help="Pretrain epochs (default: 50)",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=10,
        help="Finetune epochs, original data only (default: 10)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=1e-4,
        help="Learning rate for pretrain stage (default: 1e-4)",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-5,
        help="Learning rate for finetune stage (default: 1e-5)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--no-pos-weight",
        action="store_true",
        help="Disable positive class weighting in loss",
    )

    # ---- Early stopping ----
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs, 0 to disable (default: 5)",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.001,
        help="Minimum AUC improvement to reset patience counter (default: 0.001)",
    )

    # ---- Augmentation ----
    parser.add_argument(
        "--aug-mode",
        default="none",
        choices=["none", "online", "offline"],
        help=(
            "Augmentation mode: "
            "none=no augmentation (Exp 0), "
            "online=on-the-fly per epoch (Exp 1/2), "
            "offline=sample from pre-generated pool (Exp 3)"
        ),
    )
    parser.add_argument(
        "--aug-multiplier",
        type=float,
        default=1.0,
        help=(
            "Ratio of augmented to original samples per epoch. "
            "1.0 = add up to n_orig augmented samples (total <=2x). "
            "0.0 = no augmentation (use with --aug-mode none)."
        ),
    )
    parser.add_argument(
        "--aug-target",
        default="all",
        choices=["all", "susceptible"],
        help=(
            "Which sequences to augment: "
            "all=both R and S (Exp 1/3), "
            "susceptible=S only / Phenotype==0 (Exp 2)"
        ),
    )
    parser.add_argument(
        "--mutate-frac",
        type=float,
        default=0.005,
        help="Mutation fraction for online mode (default: 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--offline-top-retention-rate",
        type=float,
        default=None,
        help="Rank the current training-only offline pool by norm_score and keep this fraction",
    )

    # ---- Misc ----
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cuda/cpu); auto-detected if not specified",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes",
    )

    args = parser.parse_args()

    # ---- Print all arguments ----
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg:<25} = {value}")
    print("=" * 60 + "\n")

    # ---- Validation of argument combinations ----
    if args.aug_mode == "offline" and args.aug_data_path is None:
        parser.error("--aug-data-path is required when --aug-mode is offline")
    if args.aug_mode == "none" and args.aug_multiplier != 0.0:
        print(f"[WARNING] --aug-mode is none but --aug-multiplier={args.aug_multiplier}. "
              f"Multiplier will be ignored.")

    # ---- Setup ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Load original dataset ----
    print(f"\nLoading original data from: {args.data_path}")
    orig_df = pd.read_csv(args.data_path)
    print(f"  Total samples: {len(orig_df)}")

    if "is_val" not in orig_df.columns:
        raise ValueError("Original dataset must have an 'is_val' column")

    seq_cols = find_sequence_columns(orig_df)
    if not seq_cols:
        raise ValueError(f"No seq_* columns found in {args.data_path}")
    print(f"  Sequence columns: {seq_cols}")

    n_r = (orig_df[args.phenotype_col] > 0.5).sum()
    n_s = (orig_df[args.phenotype_col] <= 0.5).sum()
    print(f"  Class distribution: R={n_r}, S={n_s}")

    # ---- Split original data before any data-dependent preprocessing ----
    if args.nested_validation:
        orig_train_df, inner_val_df, outer_test_df, split_manifest = build_nested_protein_split(
            orig_df, args.val_fold, args.group_col, args.inner_val_fraction, args.seed
        )
        split_manifest.to_csv(output_path / "nested_split_manifest.csv", index=False)
        print(
            f"\nNested split: train={len(orig_train_df)}, inner_val={len(inner_val_df)}, "
            f"outer_test={len(outer_test_df)}"
        )
        print(
            f"  Protein groups: train={orig_train_df[args.group_col].nunique()}, "
            f"inner_val={inner_val_df[args.group_col].nunique()}, "
            f"outer_test={outer_test_df[args.group_col].nunique()}"
        )
    else:
        orig_train_df = orig_df[orig_df["is_val"] != args.val_fold].copy().reset_index(drop=True)
        inner_val_df = orig_df[orig_df["is_val"] == args.val_fold].copy().reset_index(drop=True)
        outer_test_df = inner_val_df
        print(f"\nOriginal train set (is_val != {args.val_fold}): {len(orig_train_df)} samples")

    # ---- Load augmented dataset and exclude both outer-test and inner-val proteins ----
    aug_pool_df = None
    fold_local_top_cutoff = None
    n_aug_before_fold_local_top = None
    if args.aug_mode == "offline":
        print(f"\nLoading augmented pool from: {args.aug_data_path}")
        aug_full_df = pd.read_csv(args.aug_data_path)
        print(f"  Total rows in augmented file: {len(aug_full_df)}")
        if "aug_mutations" not in aug_full_df.columns or "is_val" not in aug_full_df.columns:
            raise ValueError("Augmented dataset must have aug_mutations and is_val columns")

        aug_pool_df = aug_full_df[aug_full_df["aug_mutations"] > 0].copy()
        if args.nested_validation:
            if args.group_col not in aug_pool_df.columns:
                raise ValueError(f"Augmented dataset must contain '{args.group_col}'")
            train_groups = set(orig_train_df[args.group_col].astype(str))
            aug_pool_df = aug_pool_df[
                aug_pool_df[args.group_col].astype(str).isin(train_groups)
            ].copy()
        else:
            aug_pool_df = aug_pool_df[aug_pool_df["is_val"] != args.val_fold].copy()
        aug_pool_df = aug_pool_df.reset_index(drop=True)
        print(f"  Augmented pool after split exclusion: {len(aug_pool_df)}")

        if args.offline_top_retention_rate is not None:
            n_aug_before_fold_local_top = len(aug_pool_df)
            aug_pool_df, fold_local_top_cutoff = select_fold_local_top_percent(
                aug_pool_df, args.offline_top_retention_rate
            )
            print(
                f"  Fold-local top filter: kept {len(aug_pool_df)}/"
                f"{n_aug_before_fold_local_top}; cutoff={fold_local_top_cutoff:.8f}"
            )

    # ---- Compute max lengths using actual training proteins only ----
    max_lengths = compute_max_lengths(orig_train_df, seq_cols)
    total_seq_len = sum(max_lengths.values())
    print(f"\nSequence max lengths (from actual training split):")
    for col, ml in max_lengths.items():
        print(f"  {col}: {ml}")
    print(f"  Total concatenated length: {total_seq_len}")

    # ---- Create model ----
    print(f"\nCreating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        seq_len=total_seq_len if args.model_type == "simple" else None,
        stem_channels=args.stem_channels,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Inner-validation and untouched outer-test loaders ----
    if args.nested_validation:
        val_loader = build_eval_loader(
            inner_val_df, seq_cols, args.phenotype_col, max_lengths,
            args.batch_size, args.num_workers, "Inner validation"
        )
        # Build the outer-test loader only at final evaluation time.
        test_loader = None
    else:
        val_loader = build_val_loader(
            orig_df=orig_df, sequence_columns=seq_cols, phenotype_column=args.phenotype_col,
            val_fold=args.val_fold, max_lengths=max_lengths, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        test_loader = val_loader

    config = {
        "data_path": args.data_path,
        "aug_data_path": args.aug_data_path,
        "sequence_columns": seq_cols,
        "phenotype_column": args.phenotype_col,
        "val_fold": args.val_fold,
        "nested_validation": args.nested_validation,
        "group_col": args.group_col,
        "inner_val_fraction": args.inner_val_fraction,
        "model_type": args.model_type,
        "stem_channels": args.stem_channels,
        "total_seq_len": total_seq_len,
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "finetune_lr": args.finetune_lr,
        "weight_decay": args.weight_decay,
        "use_pos_weight": not args.no_pos_weight,
        "patience": args.patience,
        "min_improvement": args.min_improvement,
        "aug_mode": args.aug_mode,
        "aug_multiplier": args.aug_multiplier,
        "aug_target": args.aug_target,
        "mutate_frac": args.mutate_frac,
        "offline_top_retention_rate": args.offline_top_retention_rate,
        "fold_local_top_cutoff": fold_local_top_cutoff,
        "n_aug_before_fold_local_top": n_aug_before_fold_local_top,
        "n_aug_after_fold_local_top": len(aug_pool_df) if aug_pool_df is not None else None,
        "seed": args.seed,
        "n_orig_train": len(orig_train_df),
        "n_inner_val": len(inner_val_df),
        "n_outer_test": len(outer_test_df),
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to: {output_path / 'config.json'}")

    # ==========================================================================
    # STAGE 1: PRETRAIN (original + augmented)
    # ==========================================================================

    if args.pretrain_epochs > 0:
        print(f"\n{'=' * 60}")
        print(f"STAGE 1: PRETRAIN  (aug_mode={args.aug_mode}, "
              f"multiplier={args.aug_multiplier}, target={args.aug_target})")
        print(f"{'=' * 60}")

        # Compute pos_weight from a sample loader at epoch 0
        sample_loader = build_train_loader(
            orig_train_df=orig_train_df,
            sequence_columns=seq_cols,
            phenotype_column=args.phenotype_col,
            aug_mode=args.aug_mode,
            aug_multiplier=args.aug_multiplier,
            aug_target=args.aug_target,
            max_lengths=max_lengths,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epoch=0,
            mutate_frac=args.mutate_frac,
            aug_pool_df=aug_pool_df,
            seed=args.seed,
        )
        pos_weight = None
        if not args.no_pos_weight:
            pos_weight = compute_pos_weight(sample_loader, device)

        criterion = WeightedBCELoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        best_val_auc = float("-inf")
        patience_counter = 0
        history_pretrain = []

        for epoch in range(args.pretrain_epochs):
            # Rebuild loader each epoch so augmented samples vary
            train_loader = build_train_loader(
                orig_train_df=orig_train_df,
                sequence_columns=seq_cols,
                phenotype_column=args.phenotype_col,
                aug_mode=args.aug_mode,
                aug_multiplier=args.aug_multiplier,
                aug_target=args.aug_target,
                max_lengths=max_lengths,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                epoch=epoch,
                mutate_frac=args.mutate_frac,
                aug_pool_df=aug_pool_df,
                seed=args.seed,
            )

            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch, args.pretrain_epochs, stage="Pretrain",
            )
            train_metrics = evaluate_model(model, train_loader, device)
            val_metrics = evaluate_model(model, val_loader, device)

            print(f"\n  Train Loss: {train_loss:.4f}")
            print(f"  Train AUC:  {train_metrics['auc']:.4f}")
            print(f"  Val AUC:    {val_metrics['auc']:.4f}")

            history_pretrain.append({
                "epoch": epoch,
                "stage": "pretrain",
                "train_loss": train_loss,
                "train_auc": train_metrics["auc"],
                "val_auc": val_metrics["auc"],
            })

            if val_metrics["auc"] - best_val_auc > args.min_improvement:
                best_val_auc = val_metrics["auc"]
                patience_counter = 0
                torch.save(model.state_dict(), output_path / "best_pretrain_model.pt")
                print(f"  ✓ New best pretrain model (val AUC: {best_val_auc:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement (patience: {patience_counter}/{args.patience})")

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\n  Early stopping at pretrain epoch {epoch + 1}")
                break

        pd.DataFrame(history_pretrain).to_csv(output_path / "pretrain_history.csv", index=False)

        # Load best pretrain weights before finetune
        best_pt = output_path / "best_pretrain_model.pt"
        if best_pt.exists():
            model.load_state_dict(torch.load(best_pt, map_location=device))
            print(f"\nLoaded best pretrain model (val AUC: {best_val_auc:.4f})")

    # ==========================================================================
    # STAGE 2: FINETUNE (original data only, no augmentation)
    # ==========================================================================

    if args.finetune_epochs > 0:
        print(f"\n{'=' * 60}")
        print(f"STAGE 2: FINETUNE  (original sequences only, no augmentation)")
        print(f"{'=' * 60}")

        # Fixed finetune loader — no augmentation regardless of aug_mode
        print("\nFinetune training set (original training proteins only):")
        finetune_dataset = SequenceDataset(
            orig_train_df,
            sequence_columns=seq_cols,
            phenotype_column=args.phenotype_col,
            max_lengths=max_lengths,
        )
        finetune_loader = DataLoader(
            finetune_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        pos_weight_ft = None
        if not args.no_pos_weight:
            pos_weight_ft = compute_pos_weight(finetune_loader, device)

        criterion_ft = WeightedBCELoss(pos_weight=pos_weight_ft)
        optimizer_ft = optim.AdamW(
            model.parameters(),
            lr=args.finetune_lr,
            weight_decay=args.weight_decay,
        )
        print(f"Finetune learning rate: {args.finetune_lr}")

        best_val_auc = float("-inf")
        patience_counter = 0
        history_finetune = []

        for epoch in range(args.finetune_epochs):
            train_loss = train_epoch(
                model, finetune_loader, criterion_ft, optimizer_ft, device,
                epoch, args.finetune_epochs, stage="Finetune",
            )
            train_metrics = evaluate_model(model, finetune_loader, device)
            val_metrics = evaluate_model(model, val_loader, device)

            print(f"\n  Train Loss: {train_loss:.4f}")
            print(f"  Train AUC:  {train_metrics['auc']:.4f}")
            print(f"  Val AUC:    {val_metrics['auc']:.4f}")

            history_finetune.append({
                "epoch": epoch,
                "stage": "finetune",
                "train_loss": train_loss,
                "train_auc": train_metrics["auc"],
                "val_auc": val_metrics["auc"],
            })

            if val_metrics["auc"] - best_val_auc > args.min_improvement:
                best_val_auc = val_metrics["auc"]
                patience_counter = 0
                torch.save(model.state_dict(), output_path / "best_model.pt")
                print(f"  ✓ New best model (val AUC: {best_val_auc:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement (patience: {patience_counter}/{args.patience})")

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\n  Early stopping at finetune epoch {epoch + 1}")
                break

        pd.DataFrame(history_finetune).to_csv(output_path / "finetune_history.csv", index=False)

    # ==========================================================================
    # FINAL EVALUATION
    # ==========================================================================

    print(f"\n{'=' * 60}")
    print(f"FINAL EVALUATION")
    print(f"{'=' * 60}")

    # Load best available model
    if args.finetune_epochs > 0 and (output_path / "best_model.pt").exists():
        model.load_state_dict(torch.load(output_path / "best_model.pt", map_location=device))
        print("Loaded best finetune model")
    elif args.pretrain_epochs > 0 and (output_path / "best_pretrain_model.pt").exists():
        model.load_state_dict(torch.load(output_path / "best_pretrain_model.pt", map_location=device))
        print("Loaded best pretrain model")

    if args.nested_validation:
        test_loader = build_eval_loader(
            outer_test_df, seq_cols, args.phenotype_col, max_lengths,
            args.batch_size, args.num_workers, "Untouched outer test"
        )
    final_test = evaluate_model(model, test_loader, device)
    if args.nested_validation:
        print(
            f"\nFinal outer-test AUC (untouched original sequences, fold "
            f"{args.val_fold}): {final_test['auc']:.4f}"
        )
    else:
        print(f"\nFinal Val AUC (original sequences, fold {args.val_fold}): {final_test['auc']:.4f}")

    final_results = {
        "val_auc": final_test["auc"],
        "outer_test_auc": final_test["auc"],
        "val_fold": args.val_fold,
        "nested_validation": args.nested_validation,
        "metric_role": "untouched_outer_test" if args.nested_validation else "validation",
    }
    with open(output_path / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nAll outputs saved to: {output_path}")
    print(f"  config.json, final_results.json")
    print(f"  best_pretrain_model.pt, best_model.pt")
    print(f"  pretrain_history.csv, finetune_history.csv")


if __name__ == "__main__":
    main()