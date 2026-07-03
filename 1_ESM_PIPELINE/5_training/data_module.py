"""
Data Module for Drug Resistance Prediction

Supports three augmentation modes:
  - none:    Train on original sequences only (Exp 0 baseline)
  - online:  On-the-fly mutation augmentation each epoch (Exp 1, 2)
  - offline: Pre-generated augmented pool, randomly sampled each epoch (Exp 3)

Key design:
  - Validation: always original sequences only, from is_val == val_fold
  - Training:   original sequences from is_val != val_fold
  - Offline:    augmented pool excludes any row whose is_val == val_fold (leakage prevention)
  - Multiplier: controls how many augmented samples are added per epoch relative to train originals
  - Aug-target: 'all' augments both R and S; 'susceptible' only augments S (Phenotype == 0)
  - Padding:    right-pad to max length for SimpleCNN; no padding for ResNet

Balancing design:
  - Original training samples are always kept unchanged.
  - For aug_target='all', both online and offline augmentation use a class-balanced
    augmented subset per epoch (~50/50 R/S when both classes are present).
  - Sampling is without replacement within each class until that class pool is exhausted;
    once exhausted, that class pool is reshuffled and reused.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Dict, List, Optional, Tuple

from augmentation import SequenceAugmenter


# Standard 20 amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def encode_aa_one_hot(sequence: str, max_len: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode an amino acid sequence.

    Args:
        sequence: Amino acid sequence string
        max_len:  If provided, right-pad or truncate to this length (SimpleCNN).
                  If None, use actual sequence length (ResNet).

    Returns:
        np.ndarray of shape (20, L)
    """
    seq = sequence.upper()
    out_len = max_len if max_len is not None else len(seq)

    encoded = np.zeros((20, out_len), dtype=np.float32)
    for i, aa in enumerate(seq):
        if i >= out_len:
            break
        if aa in AA_TO_IDX:
            encoded[AA_TO_IDX[aa], i] = 1.0
        # Unknown characters (-, X, etc.) remain all-zero

    return encoded


def compute_max_lengths(
    df: pd.DataFrame,
    sequence_columns: List[str],
) -> Dict[str, int]:
    """
    Compute max length per sequence column across the full dataset.
    Should be called on the original dataset so lengths are consistent
    across all splits. Only required for SimpleCNN.
    """
    max_lengths = {}
    for col in sequence_columns:
        seqs = df[col].dropna()
        max_lengths[col] = int(seqs.apply(lambda x: len(str(x))).max()) if len(seqs) > 0 else 0
    return max_lengths


# ---------------------------------------------------------------------------
# Epoch-wise balanced/cycled subset helpers
# ---------------------------------------------------------------------------


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()



def _shuffle_pool(
    df: pd.DataFrame,
    base_seed: int,
    cycle_idx: int,
    stream_offset: int,
) -> pd.DataFrame:
    if len(df) == 0:
        return _empty_like(df)
    random_state = base_seed + stream_offset + cycle_idx
    return df.sample(frac=1.0, replace=False, random_state=random_state).reset_index(drop=True)



def _take_from_cycled_pool(
    df: pd.DataFrame,
    n_rows: int,
    offset: int,
    base_seed: int,
    stream_offset: int,
) -> pd.DataFrame:
    """
    Deterministically walk through repeated shuffled copies of df.
    This gives no replacement within a cycle, then reshuffles once exhausted.
    """
    if n_rows <= 0:
        return _empty_like(df)
    if len(df) == 0:
        raise ValueError("Cannot sample from an empty augmentation pool")

    pool_size = len(df)
    start_cycle = offset // pool_size
    start_pos = offset % pool_size
    rows_left = n_rows
    pieces = []

    cycle_idx = start_cycle
    pos_in_cycle = start_pos
    while rows_left > 0:
        shuffled = _shuffle_pool(df, base_seed=base_seed, cycle_idx=cycle_idx, stream_offset=stream_offset)
        take = min(rows_left, pool_size - pos_in_cycle)
        pieces.append(shuffled.iloc[pos_in_cycle:pos_in_cycle + take].copy())
        rows_left -= take
        cycle_idx += 1
        pos_in_cycle = 0

    return pd.concat(pieces, ignore_index=True)



def _balanced_counts(n_to_use: int, epoch: int) -> Tuple[int, int]:
    half = n_to_use // 2
    if n_to_use % 2 == 0:
        return half, half

    # Alternate the extra sample across epochs to avoid systematic bias.
    if epoch % 2 == 0:
        return half + 1, half
    return half, half + 1



def _balanced_offsets(n_to_use: int, epoch: int) -> Tuple[int, int]:
    half = n_to_use // 2
    if n_to_use % 2 == 0:
        return epoch * half, epoch * half

    # Extra sample goes to R on even epochs, S on odd epochs.
    offset_r = epoch * half + ((epoch + 1) // 2)
    offset_s = epoch * half + (epoch // 2)
    return offset_r, offset_s



def build_epoch_aug_subset(
    pool_df: pd.DataFrame,
    phenotype_column: str,
    n_to_use: int,
    base_seed: int,
    epoch: int,
    balance_labels: bool,
    stream_offset_base: int,
) -> pd.DataFrame:
    """
    Build the augmented subset for one epoch.

    If balance_labels=True and both classes are present, returns an approximately
    50/50 R/S subset by cycling independently through class-specific pools.
    Otherwise, cycles through the full pool (or single class pool) without
    replacement until exhausted, then reshuffles and continues.
    """
    valid_df = pool_df[pool_df[phenotype_column].notna()].copy().reset_index(drop=True)
    if n_to_use <= 0 or len(valid_df) == 0:
        return _empty_like(valid_df)

    n_to_use = int(n_to_use)

    if balance_labels:
        pool_r = valid_df[valid_df[phenotype_column] > 0.5].reset_index(drop=True)
        pool_s = valid_df[valid_df[phenotype_column] <= 0.5].reset_index(drop=True)
        if len(pool_r) > 0 and len(pool_s) > 0:
            n_r, n_s = _balanced_counts(n_to_use, epoch)
            offset_r, offset_s = _balanced_offsets(n_to_use, epoch)

            sampled_r = _take_from_cycled_pool(
                pool_r,
                n_rows=n_r,
                offset=offset_r,
                base_seed=base_seed,
                stream_offset=stream_offset_base + 100_000,
            )
            sampled_s = _take_from_cycled_pool(
                pool_s,
                n_rows=n_s,
                offset=offset_s,
                base_seed=base_seed,
                stream_offset=stream_offset_base + 200_000,
            )

            combined = pd.concat([sampled_r, sampled_s], ignore_index=True)
            combined = combined.sample(
                frac=1.0,
                replace=False,
                random_state=base_seed + stream_offset_base + 300_000 + epoch,
            ).reset_index(drop=True)
            return combined

    offset = epoch * n_to_use
    sampled = _take_from_cycled_pool(
        valid_df,
        n_rows=n_to_use,
        offset=offset,
        base_seed=base_seed,
        stream_offset=stream_offset_base + 10_000,
    )
    return sampled.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Base dataset: encodes a fixed DataFrame, no augmentation
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Encodes a fixed DataFrame of protein sequences. No augmentation.
    Used for: validation set, finetune-stage training, offline augmented pool.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sequence_columns: List[str],
        phenotype_column: str,
        max_lengths: Optional[Dict[str, int]] = None,
    ):
        valid_df = df[df[phenotype_column].notna()].copy()
        self.df = valid_df.reset_index(drop=True)
        self.sequence_columns = sequence_columns
        self.phenotype_column = phenotype_column
        self.max_lengths = max_lengths

        n_pos = int((self.df[phenotype_column] > 0.5).sum())
        n_neg = int((self.df[phenotype_column] <= 0.5).sum())
        print(f"  SequenceDataset: {len(self.df)} samples (R: {n_pos}, S: {n_neg})")

    def __len__(self):
        return len(self.df)

    def _encode_row(self, row) -> np.ndarray:
        encoded_sequences = []
        for seq_col in self.sequence_columns:
            seq = row[seq_col]
            max_len = self.max_lengths[seq_col] if self.max_lengths else None
            if pd.isna(seq) or str(seq).strip() == "":
                out_len = max_len if max_len is not None else 1
                encoded = np.zeros((20, out_len), dtype=np.float32)
            else:
                encoded = encode_aa_one_hot(str(seq).strip(), max_len=max_len)
            encoded_sequences.append(encoded)
        return np.concatenate(encoded_sequences, axis=1)  # (20, total_L)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        concat_seq = self._encode_row(row)
        target = float(row[self.phenotype_column])
        return torch.from_numpy(concat_seq).float(), torch.tensor(target, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Online augmentation dataset: mutates on-the-fly each epoch
# ---------------------------------------------------------------------------

class OnlineAugDataset(Dataset):
    """
    Wraps a subset of original sequences and applies random mutation on-the-fly.

    Each __getitem__ call produces one augmented sample. A different augmentation
    is generated each epoch by mixing the epoch number into the random seed.

    Call set_epoch(epoch) before each epoch to vary augmentations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sequence_columns: List[str],
        phenotype_column: str,
        mutate_frac: float,
        max_lengths: Optional[Dict[str, int]] = None,
        seed: int = 42,
        copy_idx: int = 0,
    ):
        valid_df = df[df[phenotype_column].notna()].copy()
        self.df = valid_df.reset_index(drop=True)
        self.sequence_columns = sequence_columns
        self.phenotype_column = phenotype_column
        self.mutate_frac = mutate_frac
        self.max_lengths = max_lengths
        self.base_seed = seed
        self.copy_idx = copy_idx
        self.epoch = 0

        n_r = int((self.df[phenotype_column] > 0.5).sum())
        n_s = int((self.df[phenotype_column] <= 0.5).sum())
        print(f"  OnlineAugDataset (copy {copy_idx}): {len(self.df)} samples (R: {n_r}, S: {n_s})")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Deterministic but epoch-varying seed: same sample differs each epoch.
        aug_seed = hash((self.base_seed, self.epoch, self.copy_idx, int(idx))) % (2 ** 31)

        augmenter = SequenceAugmenter(
            mutate_frac=self.mutate_frac,
            mutate_randomly=True,
            use_deletion=False,
            use_insertion=False,
            use_translocation=False,
            use_mutation=True,
        )

        rng = np.random.RandomState(aug_seed)
        encoded_sequences = []

        for seq_col in self.sequence_columns:
            seq = row[seq_col]
            max_len = self.max_lengths[seq_col] if self.max_lengths else None

            if pd.isna(seq) or str(seq).strip() == "":
                out_len = max_len if max_len is not None else 1
                encoded = np.zeros((20, out_len), dtype=np.float32)
            else:
                seq_str = str(seq).strip()
                local_seed = int(rng.randint(0, 1e9))
                seq_aug = augmenter.augment(seq_str, seed=local_seed)
                encoded = encode_aa_one_hot(seq_aug, max_len=max_len)

            encoded_sequences.append(encoded)

        concat_seq = np.concatenate(encoded_sequences, axis=1)
        target = float(row[self.phenotype_column])
        return torch.from_numpy(concat_seq).float(), torch.tensor(target, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Offline augmentation dataset: resamples from pre-generated pool each epoch
# ---------------------------------------------------------------------------

class OfflineAugDataset(Dataset):
    """
    Holds the full pre-generated augmented pool (aug_mutations > 0, is_val != val_fold).

    For aug_target='all', each epoch draws a class-balanced augmented subset.
    Sampling is without replacement within each class until that class pool is
    exhausted, then that class pool is reshuffled and reused.

    For single-class pools (for example aug_target='susceptible'), the dataset
    cycles through that one pool without replacement until exhausted, then
    reshuffles and continues.

    Call resample(epoch) at the start of each epoch to refresh the active subset.
    """

    def __init__(
        self,
        aug_pool_df: pd.DataFrame,
        sequence_columns: List[str],
        phenotype_column: str,
        n_to_use: int,
        max_lengths: Optional[Dict[str, int]] = None,
        seed: int = 42,
    ):
        valid_df = aug_pool_df[aug_pool_df[phenotype_column].notna()].copy()
        self.pool_df = valid_df.reset_index(drop=True)
        self.sequence_columns = sequence_columns
        self.phenotype_column = phenotype_column
        self.n_to_use = max(int(n_to_use), 0)
        self.max_lengths = max_lengths
        self.base_seed = seed

        self.pool_r = self.pool_df[self.pool_df[phenotype_column] > 0.5].reset_index(drop=True)
        self.pool_s = self.pool_df[self.pool_df[phenotype_column] <= 0.5].reset_index(drop=True)
        self.use_balanced_sampling = len(self.pool_r) > 0 and len(self.pool_s) > 0

        self.active_df = self._build_active_df(epoch=0)

        print(
            "  OfflineAugDataset: "
            f"pool size={len(self.pool_df)} (R: {len(self.pool_r)}, S: {len(self.pool_s)}), "
            f"using {self.n_to_use} per epoch"
        )
        if self.use_balanced_sampling:
            print("    Balanced offline sampling enabled: ~50/50 R/S per epoch")
        else:
            print("    Single-class offline pool detected; using cyclic sampling within available class")

    def _build_active_df(self, epoch: int) -> pd.DataFrame:
        return build_epoch_aug_subset(
            pool_df=self.pool_df,
            phenotype_column=self.phenotype_column,
            n_to_use=self.n_to_use,
            base_seed=self.base_seed,
            epoch=epoch,
            balance_labels=self.use_balanced_sampling,
            stream_offset_base=0,
        )

    def resample(self, epoch: int):
        self.active_df = self._build_active_df(epoch)
        n_r = int((self.active_df[self.phenotype_column] > 0.5).sum())
        n_s = int((self.active_df[self.phenotype_column] <= 0.5).sum())
        print(f"    [Offline epoch {epoch}] augmented subset: {len(self.active_df)} samples (R: {n_r}, S: {n_s})")

    def __len__(self):
        return len(self.active_df)

    def __getitem__(self, idx):
        row = self.active_df.iloc[idx]
        encoded_sequences = []

        for seq_col in self.sequence_columns:
            seq = row[seq_col]
            max_len = self.max_lengths[seq_col] if self.max_lengths else None

            if pd.isna(seq) or str(seq).strip() == "":
                out_len = max_len if max_len is not None else 1
                encoded = np.zeros((20, out_len), dtype=np.float32)
            else:
                encoded = encode_aa_one_hot(str(seq).strip(), max_len=max_len)

            encoded_sequences.append(encoded)

        concat_seq = np.concatenate(encoded_sequences, axis=1)
        target = float(row[self.phenotype_column])
        return torch.from_numpy(concat_seq).float(), torch.tensor(target, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Factory: build train / val loaders
# ---------------------------------------------------------------------------


def build_val_loader(
    orig_df: pd.DataFrame,
    sequence_columns: List[str],
    phenotype_column: str,
    val_fold: int,
    max_lengths: Optional[Dict[str, int]],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """
    Validation DataLoader — always original sequences from is_val == val_fold.
    """
    val_df = orig_df[orig_df["is_val"] == val_fold].copy()
    print(f"\nValidation set (fold {val_fold}, original sequences only):")
    val_dataset = SequenceDataset(
        val_df,
        sequence_columns=sequence_columns,
        phenotype_column=phenotype_column,
        max_lengths=max_lengths,
    )
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )



def build_eval_loader(
    eval_df: pd.DataFrame,
    sequence_columns: List[str],
    phenotype_column: str,
    max_lengths: Optional[Dict[str, int]],
    batch_size: int,
    num_workers: int,
    split_name: str,
) -> DataLoader:
    """Build a non-shuffled loader for an explicit original-sequence split."""
    print(f"\n{split_name} set (original sequences only):")
    eval_dataset = SequenceDataset(
        eval_df,
        sequence_columns=sequence_columns,
        phenotype_column=phenotype_column,
        max_lengths=max_lengths,
    )
    return DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_train_loader(
    orig_train_df: pd.DataFrame,
    sequence_columns: List[str],
    phenotype_column: str,
    aug_mode: str,
    aug_multiplier: float,
    aug_target: str,
    max_lengths: Optional[Dict[str, int]],
    batch_size: int,
    num_workers: int,
    epoch: int,
    mutate_frac: float,
    aug_pool_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> DataLoader:
    """
    Build a training DataLoader for a single epoch.

    Combines:
      - SequenceDataset  (all original train rows, always included)
      - OnlineAugDataset or OfflineAugDataset (augmented rows, capped by multiplier)

    Args:
        orig_train_df:  Original training rows (is_val != val_fold)
        aug_mode:       'none', 'online', or 'offline'
        aug_multiplier: Max augmented/original ratio (e.g. 1.0 means add up to n_orig aug samples)
        aug_target:     'all' or 'susceptible'
        aug_pool_df:    Pre-generated augmented rows (offline mode only);
                        must already have val_fold rows excluded by caller
        epoch:          Current epoch (for seed variation)
        mutate_frac:    Mutation fraction for online mode
    """
    n_orig = len(orig_train_df)
    n_aug_cap = int(n_orig * aug_multiplier)

    # Always include all original training samples
    base_dataset = SequenceDataset(
        orig_train_df,
        sequence_columns=sequence_columns,
        phenotype_column=phenotype_column,
        max_lengths=max_lengths,
    )

    if aug_mode == "none" or n_aug_cap == 0:
        dataset = base_dataset
        aug_count = 0

    elif aug_mode == "online":
        if aug_target == "susceptible":
            eligible_df = orig_train_df[orig_train_df[phenotype_column] <= 0.5].copy()
            balance_online = False
        else:  # 'all'
            eligible_df = orig_train_df.copy()
            balance_online = True

        if len(eligible_df) == 0:
            dataset = base_dataset
            aug_count = 0
        else:
            online_parent_df = build_epoch_aug_subset(
                pool_df=eligible_df,
                phenotype_column=phenotype_column,
                n_to_use=n_aug_cap,
                base_seed=seed,
                epoch=epoch,
                balance_labels=balance_online,
                stream_offset_base=400_000,
            ).reset_index(drop=True)

            aug_dataset = OnlineAugDataset(
                online_parent_df,
                sequence_columns=sequence_columns,
                phenotype_column=phenotype_column,
                mutate_frac=mutate_frac,
                max_lengths=max_lengths,
                seed=seed,
                copy_idx=0,
            )
            aug_dataset.set_epoch(epoch)
            dataset = ConcatDataset([base_dataset, aug_dataset])
            aug_count = len(aug_dataset)

            n_r = int((online_parent_df[phenotype_column] > 0.5).sum())
            n_s = int((online_parent_df[phenotype_column] <= 0.5).sum())
            print(
                f"    [Online epoch {epoch}] augmented parent subset: "
                f"{len(online_parent_df)} samples (R: {n_r}, S: {n_s})"
            )
            if balance_online and n_r > 0 and n_s > 0:
                print("    Balanced online sampling enabled: ~50/50 R/S augmented parents per epoch")

    elif aug_mode == "offline":
        if aug_pool_df is None:
            raise ValueError("aug_pool_df must be provided for offline mode")

        if aug_target == "susceptible":
            pool = aug_pool_df[aug_pool_df[phenotype_column] <= 0.5].copy()
        else:
            pool = aug_pool_df.copy()

        aug_dataset = OfflineAugDataset(
            pool,
            sequence_columns=sequence_columns,
            phenotype_column=phenotype_column,
            n_to_use=n_aug_cap,
            max_lengths=max_lengths,
            seed=seed,
        )
        aug_dataset.resample(epoch)
        dataset = ConcatDataset([base_dataset, aug_dataset])
        aug_count = len(aug_dataset)

    else:
        raise ValueError(f"Unknown aug_mode '{aug_mode}'. Choose from: none, online, offline")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"  [Epoch {epoch}] Train: {n_orig} original + {aug_count} augmented "
        f"= {n_orig + aug_count} total | mode={aug_mode} target={aug_target}"
    )
    return loader
