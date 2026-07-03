#!/usr/bin/env python
"""
Score augmented sequences in chunks (for large datasets or parallel processing).

This script:
  1. Reads: {data_root}/{drug}_augmented.csv
  2. Scores rows [start_idx, end_idx)
  3. Saves chunk to: {data_root}/{drug}_augmented_chunk_{start}_{end}.csv
  
Later use merge_and_score_pppl.py to merge all chunks.

Usage:
    python score_pppl_chunks.py \
        --drug ethambutol \
        --start-idx 0 \
        --end-idx 1000 \
        --data-root ../../datasets/augmented \
        --device cuda:0
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
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


def compute_avg_log_likelihood(seq: str, model, tokenizer, device) -> float:
    """Compute average masked-token log probability."""
    seq = seq.strip()
    if len(seq) == 0:
        return float("nan")

    model_max_len = getattr(tokenizer, "model_max_length", None)
    if model_max_len is None or model_max_len <= 0 or model_max_len > 10000:
        model_max_len = getattr(getattr(model, "config", None), "max_position_embeddings", 1024)
    max_residues = max(1, int(model_max_len) - 2)

    chunks = _sequence_chunks(seq, max_residues=max_residues)
    total_logp = 0.0
    total_count = 0

    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            encoded = tokenizer(chunk, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"].to(device)
            attn_mask = encoded["attention_mask"].to(device)

            L = input_ids.size(1)
            if L <= 2:
                continue

            start = 1
            end = L - 1
            for pos in range(start, end):
                masked = input_ids.clone()
                masked[0, pos] = tokenizer.mask_token_id

                outputs = model(masked, attention_mask=attn_mask)
                logits = outputs.logits
                log_probs = torch.log_softmax(logits[0, pos], dim=-1)
                true_id = int(input_ids[0, pos])

                total_logp += float(log_probs[true_id].item())
                total_count += 1

    return total_logp / total_count if total_count else float("nan")


def compute_pppl(avg_logp: float) -> float:
    """Convert average log-likelihood to pseudo-perplexity."""
    if np.isnan(avg_logp):
        return float("nan")
    return float(np.exp(-avg_logp))


def score_pppl_chunk(
    drug: str,
    start_idx: int,
    end_idx: int,
    data_root: str,
    model_name: str = "facebook/esm2_t12_35M_UR50D",
    device_str: str = None,
) -> str:
    """Score rows [start_idx, end_idx) and save chunk."""
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading ESM: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    # Load full dataset
    csv_path = os.path.join(data_root, f"{drug}_augmented.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    print(f"[INFO] Loading: {csv_path}")
    full_df = pd.read_csv(csv_path)

    n_total = len(full_df)
    if start_idx < 0 or start_idx >= n_total:
        raise ValueError(f"start_idx {start_idx} out of range [0, {n_total})")
    end_idx = min(end_idx, n_total)

    print(f"[INFO] Processing rows {start_idx} to {end_idx} (total: {n_total})")

    # Extract chunk
    df = full_df.iloc[start_idx:end_idx].copy()
    df["row_idx"] = full_df.index[start_idx:end_idx]

    seq_cols = find_sequence_columns(df)
    print(f"[INFO] Sequence columns: {seq_cols}")

    # Add PPPL columns
    pppl_cols = [f"PPPL_{c}" for c in seq_cols]
    for col in pppl_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Score chunk
    print("[INFO] Scoring chunk...")
    for local_i in tqdm(range(len(df)), desc=f"Chunk {start_idx}-{end_idx}"):
        row = df.iloc[local_i]
        global_idx = row["row_idx"]

        for seq_col in seq_cols:
            seq = "" if pd.isna(row[seq_col]) else str(row[seq_col])
            if not seq:
                continue

            if not pd.isna(row.get(f"PPPL_{seq_col}", np.nan)):
                continue

            avg_logp = compute_avg_log_likelihood(seq, model, tokenizer, device)
            df.at[global_idx, f"PPPL_{seq_col}"] = compute_pppl(avg_logp)

        if device.type == "cuda" and (local_i + 1) % 100 == 0:
            torch.cuda.empty_cache()

    # Compute PPPL_mean
    df["PPPL_mean"] = df[pppl_cols].mean(axis=1)

    # Save chunk
    out_path = os.path.join(data_root, f"{drug}_augmented_chunk_{start_idx}_{end_idx}.csv")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved chunk to: {out_path}")

    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", required=True, help="Drug name")
    ap.add_argument("--start-idx", type=int, required=True, help="Start row (inclusive)")
    ap.add_argument("--end-idx", type=int, required=True, help="End row (exclusive)")
    ap.add_argument("--data-root", required=True, help="Directory with augmented CSV")
    ap.add_argument("--model-name", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--device", default=None, help="Device (e.g., 'cuda:0', 'cpu')")
    args = ap.parse_args()

    score_pppl_chunk(
        drug=args.drug,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        data_root=args.data_root,
        model_name=args.model_name,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
