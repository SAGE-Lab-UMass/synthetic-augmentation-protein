#!/usr/bin/env python
"""
Drug-balanced statistical tests for augmentation strategy comparisons.

This diagnostic compares model metrics across matched drug/fold results. It treats
drug as the main unit of inference:

1. Collect paired fold-level values for each requested comparison.
2. Average paired differences within each drug.
3. Test drug-level deltas with an exact sign-flip permutation test.
4. Report bootstrap confidence intervals by resampling drugs.
5. Apply Benjamini-Hochberg FDR correction across planned comparisons.
6. Plot per-drug delta forest plots.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / f"{os.environ.get('USER', 'user')}_matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


FOLD_RE = re.compile(r"^val_?(\d+)$")
IGNORE_DIRS = {".ipynb_checkpoints", "__pycache__"}


DEFAULT_SERIES = [
    ("baseline", "runs/exp0"),
    ("no_esm", "runs_new/exp1_m1_b"),
    ("mut0.05_0.01", "runs_new/exp3_m1_0p01"),
    ("mut1_0.01", "runs/exp3_1_m1_0p01"),
    ("mut1_top50", "runs/exp3_1_m1_top50"),
]

DEFAULT_COMPARISONS = [
    ("baseline", "no_esm"),
    ("baseline", "mut0.05_0.01"),
    ("baseline", "mut1_0.01"),
    ("baseline", "mut1_top50"),
    ("no_esm", "mut0.05_0.01"),
    ("no_esm", "mut1_0.01"),
    ("no_esm", "mut1_top50"),
    ("mut0.05_0.01", "mut1_0.01"),
    ("mut0.05_0.01", "mut1_top50"),
    ("mut1_0.01", "mut1_top50"),
]


COMPARISON_FAMILIES = {
    "no_esm_minus_baseline": "secondary_no_esm_vs_baseline",
    "mut0.05_0.01_minus_baseline": "primary_esm_vs_baseline",
    "mut1_0.01_minus_baseline": "primary_esm_vs_baseline",
    "mut1_top50_minus_baseline": "primary_esm_vs_baseline",
    "mut0.05_0.01_minus_no_esm": "esm_vs_no_esm",
    "mut1_0.01_minus_no_esm": "esm_vs_no_esm",
    "mut1_top50_minus_no_esm": "esm_vs_no_esm",
    "mut1_0.01_minus_mut0.05_0.01": "within_esm_settings",
    "mut1_top50_minus_mut0.05_0.01": "within_esm_settings",
    "mut1_top50_minus_mut1_0.01": "within_esm_settings",
}


def parse_label_path(value: str) -> Tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL:PATH")
    label, path = value.split(":", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL:PATH")
    return label, Path(path)


def parse_comparison(value: str) -> Tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected BASELINE:EXPERIMENT, meaning EXPERIMENT - BASELINE")
    baseline, experiment = value.split(":", 1)
    baseline = baseline.strip()
    experiment = experiment.strip()
    if not baseline or not experiment:
        raise argparse.ArgumentTypeError("Expected non-empty BASELINE:EXPERIMENT")
    return baseline, experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drug-balanced statistical tests for augmentation settings")
    parser.add_argument(
        "--series",
        action="append",
        type=parse_label_path,
        default=[],
        metavar="LABEL:PATH",
        help="Run root. Repeat for each setting. Default uses the four main augmentation settings.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        type=parse_comparison,
        default=[],
        metavar="BASELINE:EXPERIMENT",
        help="Comparison to test as EXPERIMENT - BASELINE. Repeat as needed.",
    )
    parser.add_argument("--metric", default="val_auc", help="Metric key in final_results.json.")
    parser.add_argument("--model-subdir", default="resnet", help="Model subdirectory to use if present.")
    parser.add_argument(
        "--exclude-drug",
        action="append",
        default=[],
        help="Drug to exclude from paired testing. Repeat to exclude multiple drugs.",
    )
    parser.add_argument(
        "--alternative",
        choices=["greater", "less", "two-sided"],
        default="greater",
        help="Alternative for testing experiment minus baseline deltas.",
    )
    parser.add_argument("--output-dir", default="diagnostic/07_statistical_tests/results/augmentation_setting_tests")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fig-dpi", type=int, default=220)
    return parser.parse_args()


def parse_fold_token(name: str) -> Optional[int]:
    match = FOLD_RE.match(name)
    return int(match.group(1)) if match else None


def visible_subdirs(path: Path) -> List[str]:
    if not path.exists():
        return []
    return sorted(
        p.name
        for p in path.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name not in IGNORE_DIRS
    )


def resolve_series_root(path: Path, model_subdir: str) -> Path:
    path = path.expanduser()
    if (path / model_subdir).is_dir():
        return path / model_subdir
    return path


def resolve_result_dir(path: Path) -> Optional[Path]:
    if (path / "final_results.json").exists():
        return path
    none_dir = path / "none"
    if (none_dir / "final_results.json").exists():
        return none_dir
    candidates = [child for child in path.iterdir() if child.is_dir() and (child / "final_results.json").exists()] if path.exists() else []
    if not candidates:
        return None
    for child in candidates:
        if child.name == "none":
            return child
    return candidates[0] if len(candidates) == 1 else None


def find_results(root: Path, model_subdir: str) -> "OrderedDict[str, OrderedDict[int, Path]]":
    root = resolve_series_root(root, model_subdir)
    results: "OrderedDict[str, OrderedDict[int, Path]]" = OrderedDict()
    if not root.exists():
        return results

    top_dirs = visible_subdirs(root)
    top_fold_dirs = [name for name in top_dirs if parse_fold_token(name) is not None]

    if top_fold_dirs:
        for fold_name in top_fold_dirs:
            fold = parse_fold_token(fold_name)
            assert fold is not None
            fold_path = root / fold_name
            for drug in visible_subdirs(fold_path):
                drug_path = fold_path / drug
                chosen = resolve_result_dir(drug_path)
                if chosen is None:
                    inner = []
                    for child_name in visible_subdirs(drug_path):
                        inner_fold = parse_fold_token(child_name)
                        if inner_fold is None:
                            continue
                        result_dir = resolve_result_dir(drug_path / child_name)
                        if result_dir is not None:
                            inner.append((inner_fold, result_dir))
                    exact = [p for inner_fold, p in inner if inner_fold == fold]
                    if exact:
                        chosen = exact[0]
                    elif len(inner) == 1:
                        chosen = inner[0][1]
                    elif inner:
                        fold0 = [p for inner_fold, p in inner if inner_fold == 0]
                        chosen = fold0[0] if fold0 else sorted(inner, key=lambda x: x[0])[0][1]
                if chosen is not None:
                    results.setdefault(drug, OrderedDict())[fold] = chosen
    else:
        for drug in top_dirs:
            drug_path = root / drug
            fold_map: "OrderedDict[int, Path]" = OrderedDict()
            for fold_name in visible_subdirs(drug_path):
                fold = parse_fold_token(fold_name)
                if fold is None:
                    continue
                result_dir = resolve_result_dir(drug_path / fold_name)
                if result_dir is not None:
                    fold_map[fold] = result_dir
            if fold_map:
                results[drug] = OrderedDict(sorted(fold_map.items()))

    return OrderedDict((drug, OrderedDict(sorted(folds.items()))) for drug, folds in sorted(results.items()))


def load_metric(result_dir: Path, metric: str) -> Optional[float]:
    path = result_dir / "final_results.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if metric in data:
        return float(data[metric])
    if metric == "val_auc" and isinstance(data.get("val"), dict) and "auc" in data["val"]:
        return float(data["val"]["auc"])
    return None


def collect_metric_values(
    series: Sequence[Tuple[str, Path]],
    metric: str,
    model_subdir: str,
) -> pd.DataFrame:
    rows = []
    for label, root in series:
        discovered = find_results(root, model_subdir=model_subdir)
        for drug, fold_map in discovered.items():
            for fold, result_dir in fold_map.items():
                value = load_metric(result_dir, metric)
                if value is None:
                    continue
                rows.append(
                    {
                        "setting": label,
                        "root": str(root),
                        "resolved_result_dir": str(result_dir),
                        "drug": drug,
                        "fold": int(fold),
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows).sort_values(["setting", "drug", "fold"]).reset_index(drop=True)


def collect_paired_fold_values(metric_df: pd.DataFrame, comparisons: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for baseline, experiment in comparisons:
        b = metric_df.loc[metric_df["setting"] == baseline, ["drug", "fold", "value"]].rename(
            columns={"value": "baseline_value"}
        )
        e = metric_df.loc[metric_df["setting"] == experiment, ["drug", "fold", "value"]].rename(
            columns={"value": "experiment_value"}
        )
        paired = b.merge(e, on=["drug", "fold"], how="inner")
        comparison = f"{experiment}_minus_{baseline}"
        for row in paired.itertuples(index=False):
            rows.append(
                {
                    "comparison": comparison,
                    "hypothesis_family": COMPARISON_FAMILIES.get(comparison, "custom"),
                    "baseline_setting": baseline,
                    "experiment_setting": experiment,
                    "drug": row.drug,
                    "fold": int(row.fold),
                    "baseline_value": float(row.baseline_value),
                    "experiment_value": float(row.experiment_value),
                    "delta": float(row.experiment_value - row.baseline_value),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["comparison", "drug", "fold"]).reset_index(drop=True)


def filter_excluded_drugs(df: pd.DataFrame, excluded_drugs: Sequence[str]) -> pd.DataFrame:
    if df.empty or not excluded_drugs:
        return df
    excluded = {drug.lower() for drug in excluded_drugs}
    return df.loc[~df["drug"].astype(str).str.lower().isin(excluded)].reset_index(drop=True)


def bootstrap_ci(values: np.ndarray, iterations: int, rng: np.random.Generator) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(iterations, len(values)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_per_drug(
    paired_df: pd.DataFrame,
    bootstrap_iterations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for (comparison, drug), group in paired_df.groupby(["comparison", "drug"], sort=True):
        deltas = group["delta"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(deltas, bootstrap_iterations, rng)
        rows.append(
            {
                "comparison": comparison,
                "hypothesis_family": group["hypothesis_family"].iloc[0],
                "baseline_setting": group["baseline_setting"].iloc[0],
                "experiment_setting": group["experiment_setting"].iloc[0],
                "drug": drug,
                "n_folds": int(len(deltas)),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else np.nan,
                "ci95_low_fold_bootstrap": ci_low,
                "ci95_high_fold_bootstrap": ci_high,
                "folds_improved": int(np.sum(deltas > 0)),
                "folds_worse": int(np.sum(deltas < 0)),
                "folds_equal": int(np.sum(np.isclose(deltas, 0.0))),
                "percent_folds_improved": float(100.0 * np.mean(deltas > 0)) if len(deltas) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(values: np.ndarray, alternative: str) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    observed = float(np.mean(values))
    means = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(values)):
        means.append(float(np.mean(values * np.asarray(signs))))
    means_arr = np.asarray(means, dtype=float)
    eps = 1e-12
    if alternative == "greater":
        return float(np.mean(means_arr >= observed - eps))
    if alternative == "less":
        return float(np.mean(means_arr <= observed + eps))
    return float(np.mean(np.abs(means_arr) >= abs(observed) - eps))


def wilcoxon_pvalue(values: np.ndarray, alternative: str) -> Tuple[float, float, str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, "no drug-level deltas"
    if np.allclose(values, 0.0):
        return 0.0, 1.0, "all drug-level deltas are zero"
    try:
        stat, p_value = wilcoxon(values, alternative=alternative, zero_method="pratt", method="auto")
        return float(stat), float(p_value), ""
    except ValueError as exc:
        return np.nan, np.nan, str(exc)


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return q.tolist()
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q[order] = adjusted
    return q.tolist()


def summarize_overall(
    per_drug_df: pd.DataFrame,
    alternative: str,
    bootstrap_iterations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for comparison, group in per_drug_df.groupby("comparison", sort=True):
        deltas = group["mean_delta"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(deltas, bootstrap_iterations, rng)
        wil_stat, wil_p, wil_note = wilcoxon_pvalue(deltas, alternative=alternative)
        sign_p = exact_sign_flip_pvalue(deltas, alternative=alternative)
        rows.append(
            {
                "comparison": comparison,
                "hypothesis_family": group["hypothesis_family"].iloc[0],
                "baseline_setting": group["baseline_setting"].iloc[0],
                "experiment_setting": group["experiment_setting"].iloc[0],
                "alternative": alternative,
                "n_drugs": int(len(deltas)),
                "mean_drug_delta": float(np.mean(deltas)),
                "median_drug_delta": float(np.median(deltas)),
                "ci95_low_drug_bootstrap": ci_low,
                "ci95_high_drug_bootstrap": ci_high,
                "exact_sign_flip_p": sign_p,
                "wilcoxon_statistic": wil_stat,
                "wilcoxon_p": wil_p,
                "wilcoxon_note": wil_note,
                "drugs_improved": int(np.sum(deltas > 0)),
                "drugs_worse": int(np.sum(deltas < 0)),
                "drugs_equal": int(np.sum(np.isclose(deltas, 0.0))),
                "percent_drugs_improved": float(100.0 * np.mean(deltas > 0)) if len(deltas) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["exact_sign_flip_q_bh_global"] = benjamini_hochberg(out["exact_sign_flip_p"].tolist())
        out["wilcoxon_q_bh_global"] = benjamini_hochberg(out["wilcoxon_p"].tolist())
        out["exact_sign_flip_q_bh_family"] = np.nan
        out["wilcoxon_q_bh_family"] = np.nan
        for _, idx in out.groupby("hypothesis_family").groups.items():
            idx = list(idx)
            out.loc[idx, "exact_sign_flip_q_bh_family"] = benjamini_hochberg(
                out.loc[idx, "exact_sign_flip_p"].tolist()
            )
            out.loc[idx, "wilcoxon_q_bh_family"] = benjamini_hochberg(
                out.loc[idx, "wilcoxon_p"].tolist()
            )
    return out


def plot_forest(per_drug_df: pd.DataFrame, overall_df: pd.DataFrame, output_dir: Path, metric: str, fig_dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for comparison, group in per_drug_df.groupby("comparison", sort=True):
        group = group.sort_values("mean_delta")
        y = np.arange(len(group))
        x = group["mean_delta"].to_numpy(dtype=float)
        xerr_low = x - group["ci95_low_fold_bootstrap"].to_numpy(dtype=float)
        xerr_high = group["ci95_high_fold_bootstrap"].to_numpy(dtype=float) - x
        xerr = np.vstack([np.maximum(xerr_low, 0), np.maximum(xerr_high, 0)])

        overall = overall_df.loc[overall_df["comparison"] == comparison].iloc[0]
        fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.45 * len(group) + 2.2)))
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.errorbar(x, y, xerr=xerr, fmt="o", color="#1f5a85", ecolor="#8bb6d6", capsize=3)
        ax.axvline(overall["mean_drug_delta"], color="#b3422f", linewidth=1.5, label="Mean across drugs")
        ax.set_yticks(y)
        ax.set_yticklabels(group["drug"].tolist())
        ax.set_xlabel(f"Delta {metric} (experiment - comparison setting)")
        subtitle = (
            f"mean={overall['mean_drug_delta']:.4f}, "
            f"95% CI [{overall['ci95_low_drug_bootstrap']:.4f}, {overall['ci95_high_drug_bootstrap']:.4f}], "
            f"sign-flip p={overall['exact_sign_flip_p']:.4g}, family q={overall['exact_sign_flip_q_bh_family']:.4g}"
        )
        fig.suptitle(comparison.replace("_minus_", " - "), y=0.985, fontsize=12)
        fig.text(0.125, 0.925, subtitle, ha="left", va="top", fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        safe_name = comparison.replace("/", "_").replace(" ", "_")
        fig.savefig(output_dir / f"forestplot_{safe_name}.png", dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)


def write_summary(
    output_dir: Path,
    series: Sequence[Tuple[str, Path]],
    comparisons: Sequence[Tuple[str, str]],
    args: argparse.Namespace,
    metric_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    overall_df: pd.DataFrame,
) -> None:
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Diagnostic 07: drug-balanced augmentation statistical tests\n")
        handle.write("=" * 78 + "\n")
        handle.write(f"metric={args.metric}\n")
        handle.write(f"model_subdir={args.model_subdir}\n")
        handle.write(f"alternative={args.alternative}\n")
        handle.write(f"excluded_drugs={','.join(args.exclude_drug) if args.exclude_drug else 'none'}\n")
        handle.write(f"bootstrap_iterations={args.bootstrap_iterations}\n")
        handle.write(f"seed={args.seed}\n\n")
        handle.write("Series:\n")
        for label, root in series:
            n = int((metric_df["setting"] == label).sum()) if len(metric_df) else 0
            handle.write(f"  {label}: {root} ({n} metric rows)\n")
        handle.write("\nComparisons are EXPERIMENT - BASELINE:\n")
        for baseline, experiment in comparisons:
            n = int(
                (
                    (paired_df["baseline_setting"] == baseline)
                    & (paired_df["experiment_setting"] == experiment)
                ).sum()
            ) if len(paired_df) else 0
            handle.write(f"  {experiment} - {baseline}: {n} paired fold rows\n")
        handle.write("\nOverall tests:\n")
        if len(overall_df):
            handle.write(overall_df.to_string(index=False))
            handle.write("\n")


def validate_inputs(
    series: Sequence[Tuple[str, Path]],
    comparisons: Sequence[Tuple[str, str]],
) -> None:
    labels = [label for label, _ in series]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Duplicate series labels: {duplicates}")
    label_set = set(labels)
    missing = sorted({x for comp in comparisons for x in comp if x not in label_set})
    if missing:
        raise ValueError(f"Comparison labels not found in --series: {missing}")


def main() -> None:
    args = parse_args()
    series = args.series if args.series else [(label, Path(path)) for label, path in DEFAULT_SERIES]
    comparisons = args.comparison if args.comparison else DEFAULT_COMPARISONS
    validate_inputs(series, comparisons)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    metric_df = collect_metric_values(series, metric=args.metric, model_subdir=args.model_subdir)
    metric_df = filter_excluded_drugs(metric_df, args.exclude_drug)
    if metric_df.empty:
        raise RuntimeError("No metric values found. Check --series paths and --model-subdir.")

    paired_df = collect_paired_fold_values(metric_df, comparisons=comparisons)
    if paired_df.empty:
        raise RuntimeError("No paired fold values found. Check that settings share drug/fold IDs.")

    per_drug_df = summarize_per_drug(
        paired_df,
        bootstrap_iterations=args.bootstrap_iterations,
        rng=rng,
    )
    overall_df = summarize_overall(
        per_drug_df,
        alternative=args.alternative,
        bootstrap_iterations=args.bootstrap_iterations,
        rng=rng,
    )

    metric_df.to_csv(output_dir / "metric_values.csv", index=False)
    paired_df.to_csv(output_dir / "paired_fold_values.csv", index=False)
    per_drug_df.to_csv(output_dir / "per_drug_effects.csv", index=False)
    overall_df.to_csv(output_dir / "overall_drug_balanced_tests.csv", index=False)
    plot_forest(per_drug_df, overall_df, output_dir / "plots", metric=args.metric, fig_dpi=args.fig_dpi)
    write_summary(output_dir, series, comparisons, args, metric_df, paired_df, overall_df)

    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
