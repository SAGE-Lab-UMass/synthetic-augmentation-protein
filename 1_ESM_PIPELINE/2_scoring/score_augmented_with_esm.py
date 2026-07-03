#!/usr/bin/env python
"""
Score augmented sequences with ESM-2 using pseudo-perplexity (PPPL).

This script:
  1. Reads: {data_root}/{drug}_augmented.csv
  2. Finds all seq_* columns
  3. Computes PPPL for each sequence using ESM-2
  4. Adds columns: PPPL_<seq_col>, PPPL_mean
  5. Computes score = PPPL_mean(row) - PPPL_mean(WT reference)
  6. Saves IN-PLACE (same file with added columns)

Usage:
    python score_augmented_with_esm.py \
        --drug ethambutol \
        --data-root ../../datasets/augmented \
        --device cuda
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm.auto import tqdm


def find_sequence_columns(df: pd.DataFrame) -> List[str]:
    """Return all columns whose name starts with 'seq_'."""
    return [c for c in df.columns if c.startswith("seq_")]


def _sequence_chunks(seq: str, max_residues: int) -> List[str]:
    """Split a long sequence into contiguous chunks that fit the model window."""
    if len(seq) <= max_residues:
        return [seq]
    return [seq[i : i + max_residues] for i in range(0, len(seq), max_residues)]


def _has_sequence_value(value) -> bool:
    """Return whether a cell contains a real, non-empty sequence string."""
    if pd.isna(value):
        return False
    text = str(value).strip()
    return len(text) > 0 and text.lower() != "nan"


def _row_is_scored(df: pd.DataFrame, idx: int, seq_cols: List[str]) -> bool:
    """Return whether every non-empty sequence column for a row has a PPPL value."""
    for seq_col in seq_cols:
        if not _has_sequence_value(df.at[idx, seq_col]):
            continue
        if pd.isna(df.at[idx, f"PPPL_{seq_col}"]):
            return False
    return True


def _batched(iterable: List[int], batch_size: int):
    """Yield contiguous batches from a list-like sequence."""
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def compute_avg_log_likelihood(seq: str, model, tokenizer, device, mask_batch_size: int = 32):
    """Compute average masked-token log probability for one protein sequence."""
    seq = seq.strip()
    if len(seq) == 0:
        return float("nan")

    model_max_len = getattr(tokenizer, "model_max_length", None)
    if model_max_len is None or model_max_len <= 0 or model_max_len > 10000:
        model_max_len = getattr(getattr(model, "config", None), "max_position_embeddings", 1024)
    max_residues = max(1, int(model_max_len) - 2)

    chunks = _sequence_chunks(seq, max_residues=max_residues)
    chunk_logp_sum = 0.0
    chunk_residue_count = 0

    model.eval()
    with torch.inference_mode():
        for chunk in chunks:
            encoded = tokenizer(chunk, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"].to(device)
            attn_mask = encoded["attention_mask"].to(device)

            L = input_ids.size(1)
            if L <= 2:
                continue

            start = 1
            end = L - 1
            total_logp = 0.0
            count = 0

            positions = list(range(start, end))
            for pos_batch in _batched(positions, batch_size=max(1, int(mask_batch_size))):
                batch_len = len(pos_batch)
                masked = input_ids.repeat(batch_len, 1)
                batch_attn_mask = attn_mask.repeat(batch_len, 1)
                row_indices = torch.arange(batch_len, device=device)
                pos_tensor = torch.tensor(pos_batch, device=device)

                masked[row_indices, pos_tensor] = tokenizer.mask_token_id

                outputs = model(masked, attention_mask=batch_attn_mask)
                logits = outputs.logits[row_indices, pos_tensor]
                log_probs = torch.log_softmax(logits, dim=-1)
                true_ids = input_ids[0, pos_tensor]

                total_logp += float(log_probs.gather(1, true_ids.unsqueeze(1)).sum().item())
                count += batch_len

            if count > 0:
                chunk_logp_sum += total_logp
                chunk_residue_count += count

    return (chunk_logp_sum / chunk_residue_count) if chunk_residue_count else float("nan")


def compute_pppl(avg_logp: float) -> float:
    """Convert average log-likelihood to pseudo-perplexity."""
    if np.isnan(avg_logp):
        return float("nan")
    return float(np.exp(-avg_logp))


def _atomic_to_csv(df: pd.DataFrame, csv_path: str) -> None:
    """Write a CSV atomically so interrupted runs leave a valid checkpoint."""
    target = Path(csv_path)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, target)


def score_augmented_with_esm(
    drug: str,
    data_root: str,
    model_name: str = "facebook/esm2_t12_35M_UR50D",
    device_str: str = None,
    checkpoint_every: int = 1000,
    mask_batch_size: int = 32,
    stop_after_scored_rows: int | None = None,
):
    """
    Score augmented sequences with ESM and compute final scores.

    Saves IN-PLACE to {data_root}/{drug}_augmented.csv
    """
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading ESM: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    csv_path = os.path.join(data_root, f"{drug}_augmented.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    print(f"[INFO] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    seq_cols = find_sequence_columns(df)
    print(f"[INFO] Found sequence columns: {seq_cols}")
    print(f"[INFO] Total rows: {len(df)}")

    pppl_cols = [f"PPPL_{c}" for c in seq_cols]
    for col in pppl_cols:
        if col not in df.columns:
            df[col] = np.nan

    scored_rows = int(sum(_row_is_scored(df, idx, seq_cols) for idx in range(len(df))))
    print(f"[INFO] Rows already scored: {scored_rows}/{len(df)}")
    if stop_after_scored_rows is not None and scored_rows >= stop_after_scored_rows:
        print(f"[INFO] stop_after_scored_rows={stop_after_scored_rows} already satisfied; skipping scoring loop.")
        _atomic_to_csv(df, csv_path)
        return csv_path

    print("[INFO] Computing PPPL scores...")
    for idx in tqdm(range(len(df)), desc="Scoring rows"):
        row = df.iloc[idx]
        row_was_scored = _row_is_scored(df, idx, seq_cols)

        for seq_col in seq_cols:
            seq = str(row[seq_col]) if not pd.isna(row[seq_col]) else ""
            if len(seq) == 0:
                continue

            if not pd.isna(df.at[idx, f"PPPL_{seq_col}"]):
                continue

            avg_logp = compute_avg_log_likelihood(
                seq,
                model,
                tokenizer,
                device,
                mask_batch_size=mask_batch_size,
            )
            df.at[idx, f"PPPL_{seq_col}"] = compute_pppl(avg_logp)

        row_is_scored = _row_is_scored(df, idx, seq_cols)
        if not row_was_scored and row_is_scored:
            scored_rows += 1

        if device.type == "cuda" and (idx + 1) % 100 == 0:
            torch.cuda.empty_cache()

        if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0:
            _atomic_to_csv(df, csv_path)
            print(f"[INFO] Checkpoint saved at row {idx + 1}: {csv_path}")

        if stop_after_scored_rows is not None and scored_rows >= stop_after_scored_rows:
            _atomic_to_csv(df, csv_path)
            print(
                f"[INFO] Reached stop_after_scored_rows={stop_after_scored_rows}; "
                f"saved checkpoint and stopping early."
            )
            return csv_path

    df["PPPL_mean"] = df[pppl_cols].mean(axis=1)

    wt_df = df[df["aug_mutations"] == 0]
    if wt_df.empty:
        raise ValueError("No WT rows found (aug_mutations==0).")

    wt_map = {row["Filename"]: row["PPPL_mean"] for _, row in wt_df.iterrows()}

    print("[INFO] Computing scores...")
    scores = []
    for _, row in df.iterrows():
        filename = row["Filename"]
        wt_mean = wt_map.get(filename, np.nan)
        pppl_mean = row["PPPL_mean"]

        if np.isnan(wt_mean) or np.isnan(pppl_mean):
            score = np.nan
        else:
            score = pppl_mean - wt_mean

        scores.append(score)

    df["score"] = scores

    _atomic_to_csv(df, csv_path)
    print(f"[INFO] Saved updated dataset to: {csv_path}")
    print(f"[INFO] Added columns: {pppl_cols + ['PPPL_mean', 'score']}")

    return csv_path


def main():
    ap = argparse.ArgumentParser(
        description="Score augmented sequences with ESM PPPL."
    )
    ap.add_argument(
        "--drug",
        required=True,
        help="Drug name (e.g., ethambutol)",
    )
    ap.add_argument(
        "--data-root",
        required=True,
        help="Directory containing {drug}_augmented.csv (e.g., ../../datasets/augmented)",
    )
    ap.add_argument(
        "--model-name",
        default="facebook/esm2_t12_35M_UR50D",
        help="ESM model name from HuggingFace",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Device (e.g., 'cuda', 'cuda:0', 'cpu'). Auto-detects if not specified.",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Write an atomic CSV checkpoint every N rows (default: 1000; 0 disables).",
    )
    ap.add_argument(
        "--mask-batch-size",
        type=int,
        default=32,
        help="Number of masked positions to score per forward pass (default: 32).",
    )
    ap.add_argument(
        "--stop-after-scored-rows",
        type=int,
        default=None,
        help="Stop once this many rows have PPPL values checkpointed (default: score all rows).",
    )
    args = ap.parse_args()

    score_augmented_with_esm(
        drug=args.drug,
        data_root=args.data_root,
        model_name=args.model_name,
        device_str=args.device,
        checkpoint_every=args.checkpoint_every,
        mask_batch_size=args.mask_batch_size,
        stop_after_scored_rows=args.stop_after_scored_rows,
    )


if __name__ == "__main__":
    main()
