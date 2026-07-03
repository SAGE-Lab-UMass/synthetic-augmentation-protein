#!/usr/bin/env python3
from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ProteinGym similarity100 within-protein forest plots.")
    p.add_argument("--stats-dir", type=Path, required=True)
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--comparison", action="append", required=True,
                   help="Comparison label exactly as in protein_paired_values.csv, e.g. 'mut1_0p01 - baseline'")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--bootstrap-iterations", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def bh_adjust(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = min(prev, ranked[i] * n / rank)
        adj[i] = value
        prev = value
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out.tolist()


def sign_test_two_sided(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.0
    k = min(positive, negative)
    cdf = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * cdf)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, iters: int) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(iters, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def safe_name(comp: str) -> str:
    return comp.replace(" ", "_").replace("/", "_").replace("-", "_minus_")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protein_fold = pd.read_csv(args.stats_dir / "protein_fold_values.csv")
    protein_paired = pd.read_csv(args.stats_dir / "protein_paired_values.csv")
    protein_tests = pd.read_csv(args.stats_dir / "protein_paired_tests.csv")
    summary_df = pd.read_csv(args.summary_csv)
    summary_df = summary_df.rename(columns={"protein_id": "protein"})

    rng = np.random.default_rng(args.seed)

    for comp in args.comparison:
        comp_df = protein_paired.loc[protein_paired["comparison"] == comp].copy()
        if comp_df.empty:
            raise ValueError(f"Comparison not found: {comp}")

        baseline_arm = comp.split(" - ")[1]
        exp_arm = comp.split(" - ")[0]

        fold_base = protein_fold.loc[protein_fold["arm"] == baseline_arm, ["protein", "fold", "val_auc"]].rename(
            columns={"val_auc": "base_auc"}
        )
        fold_exp = protein_fold.loc[protein_fold["arm"] == exp_arm, ["protein", "fold", "val_auc"]].rename(
            columns={"val_auc": "exp_auc"}
        )
        fold_join = fold_base.merge(fold_exp, on=["protein", "fold"], how="inner")
        fold_join["delta"] = fold_join["exp_auc"] - fold_join["base_auc"]

        rows = []
        for protein, g in fold_join.groupby("protein", sort=False):
            deltas = g["delta"].to_numpy(dtype=float)
            mean_delta = float(np.mean(deltas))
            ci_low, ci_high = bootstrap_mean_ci(deltas, rng, args.bootstrap_iterations)
            improved = int(np.sum(deltas > 0))
            worse = int(np.sum(deltas < 0))
            ties = int(np.sum(np.isclose(deltas, 0.0)))
            p_two = sign_test_two_sided(improved, worse)
            rows.append(
                {
                    "protein": protein,
                    "mean_delta": mean_delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "improved_folds": improved,
                    "worse_folds": worse,
                    "tied_folds": ties,
                    "sign_p_two_sided": p_two,
                }
            )
        per_protein = pd.DataFrame(rows)
        per_protein["sign_q_bh"] = bh_adjust(per_protein["sign_p_two_sided"].tolist())
        per_protein = per_protein.merge(
            summary_df[["protein", "original_rows", "n_positive", "n_negative"]],
            on="protein",
            how="left",
        )
        per_protein["label"] = per_protein.apply(
            lambda r: f"{r['protein']} (n = {int(r['original_rows'])})", axis=1
        )
        per_protein = per_protein.sort_values("mean_delta")

        overall = protein_tests.loc[protein_tests["comparison"] == comp].iloc[0]
        overall_deltas = comp_df["delta"].to_numpy(dtype=float)
        overall_ci_low, overall_ci_high = bootstrap_mean_ci(overall_deltas, rng, args.bootstrap_iterations)

        y = np.arange(len(per_protein))
        x = per_protein["mean_delta"].to_numpy(dtype=float)
        xerr_low = x - per_protein["ci_low"].to_numpy(dtype=float)
        xerr_high = per_protein["ci_high"].to_numpy(dtype=float) - x
        xerr = np.vstack([np.maximum(xerr_low, 0), np.maximum(xerr_high, 0)])

        fig_h = max(10.5, 0.18 * len(per_protein) + 2.8)
        fig, ax = plt.subplots(figsize=(11.8, fig_h))
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.errorbar(x, y, xerr=xerr, fmt="o", color="#1f5a85", ecolor="#8bb6d6", capsize=2.5, ms=4)
        ax.axvline(overall["mean_delta"], color="#b3422f", linewidth=1.4, label="Across-protein mean")
        ax.set_yticks(y)
        ax.set_yticklabels(per_protein["label"].tolist(), fontsize=7)
        ax.set_xlabel("Validation AUC difference relative to baseline")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        title = comp.replace("mut1_0p01", r"ESM-filtered single-substitution ($\sigma \leq 0.01$)")
        title = title.replace("mut1_top50", "ESM-filtered top-50")
        title = title.replace("single_mut", "Unfiltered single-mutation")
        title = title.replace("baseline", "Baseline")
        fig.suptitle(title.replace(" - ", " vs. "), y=0.992, fontsize=14)
        subtitle = (
            f"Mean = {overall['mean_delta']:.4f}, "
            f"95% CI [{overall_ci_low:.4f}, {overall_ci_high:.4f}], "
            f"Wilcoxon p = {overall['wilcoxon_p_two_sided']:.4g}"
        )
        fig.text(0.125, 0.965, subtitle, ha="left", va="top", fontsize=10)
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout(rect=[0.22, 0.03, 0.98, 0.94])

        stem = safe_name(comp)
        png_path = args.output_dir / f"forestplot_proteingym_similarity100_{stem}.png"
        pdf_path = args.output_dir / f"forestplot_proteingym_similarity100_{stem}.pdf"
        csv_path = args.output_dir / f"forestplot_proteingym_similarity100_{stem}.csv"
        per_protein.to_csv(csv_path, index=False)
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png_path}")
        print(f"wrote {pdf_path}")
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
