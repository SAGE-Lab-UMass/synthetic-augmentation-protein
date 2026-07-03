#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARM_ORDER = [
    "no_esm_minus_baseline",
    "mut1_0.01_minus_baseline",
    "mut1_top50_minus_baseline",
]

ARM_LABELS = {
    "no_esm_minus_baseline": "Single substitution",
    "mut1_0.01_minus_baseline": "Threshold-filtered\nsingle substitution",
    "mut1_top50_minus_baseline": "Rank-filtered\nsingle substitution",
}

ARM_COLORS = {
    "no_esm_minus_baseline": "#5B84E2",
    "mut1_0.01_minus_baseline": "#63D2A6",
    "mut1_top50_minus_baseline": "#FFC61A",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot BigTB drug-level delta-AUC distributions for the three main augmentation arms."
    )
    p.add_argument("--per-drug-csv", type=Path, required=True)
    p.add_argument("--overall-csv", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    per_drug = pd.read_csv(args.per_drug_csv)
    overall = pd.read_csv(args.overall_csv)
    rng = np.random.default_rng(args.seed)

    fig, ax = plt.subplots(figsize=(11.2, 8.0))
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.2, zorder=0)

    subtitle_rows: list[str] = []

    for idx, arm in enumerate(ARM_ORDER, start=1):
        arm_df = per_drug.loc[per_drug["comparison"] == arm].copy()
        overall_row = overall.loc[overall["comparison"] == arm].iloc[0]
        deltas = arm_df["mean_delta"].to_numpy(dtype=float)

        ax.boxplot(
            [deltas],
            positions=[idx],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 2.0},
            boxprops={"facecolor": ARM_COLORS[arm], "alpha": 0.55, "edgecolor": "black", "linewidth": 1.5},
            whiskerprops={"color": "black", "linewidth": 1.2},
            capprops={"color": "black", "linewidth": 1.2},
        )

        jitter = rng.uniform(-0.10, 0.10, size=len(deltas))
        ax.scatter(
            np.full(len(deltas), idx) + jitter,
            deltas,
            s=55,
            color=ARM_COLORS[arm],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.8,
            zorder=3,
        )

        mean_delta = float(overall_row["mean_drug_delta"])
        ax.hlines(mean_delta, idx - 0.22, idx + 0.22, color="black", linewidth=4, zorder=4)
        ax.text(
            idx,
            max(deltas.max(), mean_delta) + 0.015,
            f"{mean_delta:+.4f}\n{int(overall_row['drugs_improved'])}/8 improved",
            ha="center",
            va="bottom",
            fontsize=12,
        )

        subtitle_rows.append(
            f"{ARM_LABELS[arm].replace(chr(10), ' ')}: q = {overall_row['exact_sign_flip_q_bh_family']:.4f}"
        )

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=12)
    ax.set_ylabel("Per-drug mean $\Delta$AUC vs. baseline", fontsize=16)
    ax.set_title("BigTB 8-drug benchmark", fontsize=24, pad=18)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlim(0.45, 3.55)

    ymin = min(-0.03, per_drug["mean_delta"].min() - 0.02)
    ymax = max(0.12, per_drug["mean_delta"].max() + 0.05)
    ax.set_ylim(ymin, ymax)

    fig.text(
        0.5,
        0.02,
        "Each point is one drug-level mean $\Delta$AUC. Black bars show across-drug means; labels report improved drugs and family-adjusted q-values.",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.text(0.5, 0.94, " | ".join(subtitle_rows), ha="center", va="center", fontsize=11)
    fig.tight_layout(rect=[0.03, 0.06, 0.98, 0.90])

    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
