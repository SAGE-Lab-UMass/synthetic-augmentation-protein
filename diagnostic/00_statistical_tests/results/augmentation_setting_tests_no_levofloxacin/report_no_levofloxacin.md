# Diagnostic 00 Statistical Test Report: Excluding Levofloxacin

## Purpose

This analysis repeats the drug-balanced augmentation statistical test while excluding **levofloxacin**. Levofloxacin was excluded because its dataset is small and its fold-level AUC deltas showed high variance, which can strongly influence the across-drug mean and uncertainty estimates.

The goal is to evaluate whether the conclusions about augmentation remain similar when this high-variance drug is removed.

## Settings Compared

The analysis used validation AUC (`val_auc`) from matched drug/fold results for these settings:

| Label | Run root | Interpretation |
|---|---|---|
| `baseline` | `runs/exp0` | No augmentation baseline |
| `no_esm` | `runs_new/exp1_m1_b` | Augmentation without ESM filtering |
| `mut0.05_0.01` | `runs_new/exp3_m1_0p01` | ESM-filtered augmentation, mutation 0.05, threshold 0.01 |
| `mut1_0.01` | `runs/exp3_1_m1_0p01` | ESM-filtered augmentation, mutation 1, threshold 0.01 |
| `mut1_top50` | `runs/exp3_1_m1_top50` | ESM-filtered augmentation, mutation 1, top 50% |

The analysis used the `resnet` model results.

## Excluded Drug

The following drug was excluded before paired testing:

```text
levofloxacin
```

The remaining 7 drugs were:

```text
ethambutol, ethionamide, isoniazid, moxifloxacin, pyrazinamide, rifampicin, streptomycin
```

## Statistical Method

For each comparison, fold-level paired deltas were computed as:

```text
delta = AUC(experiment, drug, fold) - AUC(comparison setting, drug, fold)
```

Fold-level deltas were averaged within each drug. The 7 drug-level deltas were used as the main inference units.

The primary p-value is the exact one-sided sign-flip p-value over drug-level deltas. The one-sided direction tests whether the experiment setting is better than the comparison setting.

## Multiple-Testing Correction

The output reports two Benjamini-Hochberg FDR corrections:

| Column | Meaning |
|---|---|
| `exact_sign_flip_q_bh_global` | FDR correction across all pairwise comparisons in this run |
| `exact_sign_flip_q_bh_family` | FDR correction only within the comparison's hypothesis family |

Family-level FDR is the preferred value for planned scientific questions.

## Overall Results

| Comparison | Family | Mean AUC Delta | 95% CI | p | Global q | Family q | Drugs Improved |
|---|---|---:|---:|---:|---:|---:|---:|
| `no_esm - baseline` | `secondary_no_esm_vs_baseline` | `+0.0141` | `[+0.0036, +0.0257]` | `0.0625` | `0.1563` | `0.0625` | `4/7` |
| `mut0.05_0.01 - baseline` | `primary_esm_vs_baseline` | `+0.0145` | `[+0.0040, +0.0254]` | `0.0391` | `0.1563` | `0.0586` | `6/7` |
| `mut1_0.01 - baseline` | `primary_esm_vs_baseline` | `+0.0163` | `[+0.0047, +0.0285]` | `0.0234` | `0.1563` | `0.0586` | `6/7` |
| `mut1_top50 - baseline` | `primary_esm_vs_baseline` | `+0.0165` | `[+0.0037, +0.0306]` | `0.0625` | `0.1563` | `0.0625` | `4/7` |
| `mut0.05_0.01 - no_esm` | `esm_vs_no_esm` | `+0.0004` | `[-0.0045, +0.0054]` | `0.4219` | `0.4688` | `0.4219` | `4/7` |
| `mut1_0.01 - no_esm` | `esm_vs_no_esm` | `+0.0023` | `[-0.0006, +0.0053]` | `0.1406` | `0.2813` | `0.2695` | `5/7` |
| `mut1_top50 - no_esm` | `esm_vs_no_esm` | `+0.0025` | `[-0.0008, +0.0066]` | `0.1797` | `0.2995` | `0.2695` | `3/7` |
| `mut1_0.01 - mut0.05_0.01` | `within_esm_settings` | `+0.0018` | `[-0.0028, +0.0069]` | `0.2734` | `0.3418` | `0.4102` | `4/7` |
| `mut1_top50 - mut0.05_0.01` | `within_esm_settings` | `+0.0020` | `[-0.0043, +0.0074]` | `0.2734` | `0.3418` | `0.4102` | `4/7` |
| `mut1_top50 - mut1_0.01` | `within_esm_settings` | `+0.0002` | `[-0.0044, +0.0057]` | `0.4688` | `0.4688` | `0.4688` | `3/7` |

## Interpretation

### 1. Primary Question: Are ESM-Filtered Settings Better Than Baseline?

After excluding levofloxacin, all ESM-filtered settings still show positive average AUC deltas relative to baseline:

- `mut0.05_0.01 - baseline`: `+0.0145`
- `mut1_0.01 - baseline`: `+0.0163`
- `mut1_top50 - baseline`: `+0.0165`

However, family-level FDR q-values are just above 0.05:

- `mut0.05_0.01 - baseline`: family q = `0.0586`
- `mut1_0.01 - baseline`: family q = `0.0586`
- `mut1_top50 - baseline`: family q = `0.0625`

Therefore, excluding levofloxacin weakens the primary result from significant to borderline/suggestive.

### 2. Secondary Question: Does No-ESM Augmentation Beat Baseline?

No-ESM augmentation has a positive mean delta:

```text
no_esm - baseline: +0.0141
```

But it does not reach the conventional 0.05 level:

```text
p = 0.0625, family q = 0.0625
```

This is also borderline/suggestive, not statistically significant.

### 3. Does ESM Filtering Improve Over No-ESM Augmentation?

After excluding levofloxacin, the ESM-filtered settings show only tiny improvements over no-ESM augmentation:

- `mut0.05_0.01 - no_esm`: `+0.0004`
- `mut1_0.01 - no_esm`: `+0.0023`
- `mut1_top50 - no_esm`: `+0.0025`

All confidence intervals cross zero and no family-level q-value is close to 0.05. Therefore, there is no evidence that ESM filtering outperforms no-ESM augmentation in this sensitivity analysis.

### 4. Are Any ESM-Filtered Settings Better Than The Others?

No. Differences among ESM-filtered settings remain very small:

- `+0.0002` to `+0.0020` AUC

All confidence intervals cross zero and family q-values are large.

## Comparison With Full-Drug Analysis

In the full-drug analysis including levofloxacin, the primary ESM-vs-baseline family is significant after family-level FDR correction. After excluding levofloxacin, the same primary comparisons remain positive but become borderline:

```text
full analysis: family q ≈ 0.029-0.031
no-levofloxacin analysis: family q ≈ 0.059-0.063
```

This indicates that levofloxacin contributes meaningfully to the full-drug statistical signal. Since levofloxacin has a small dataset and high variance, the no-levofloxacin analysis should be treated as an important sensitivity check.

## Final Conclusion

After excluding levofloxacin, augmentation still shows positive average validation AUC improvements relative to no augmentation, but the primary ESM-vs-baseline comparisons no longer pass family-level FDR correction at `q < 0.05`. The strongest setting remains `mut1_0.01`, with a mean AUC improvement of `+0.0163`, improvement in `6/7` drugs, and family q = `0.0586`.

The no-levofloxacin sensitivity analysis supports a positive augmentation trend but shows that the full-drug statistical significance depends partly on levofloxacin.

A concise summary statement is:

```text
Excluding levofloxacin, ESM-filtered augmentation continues to show positive AUC improvements over baseline, but the primary family-level FDR results become borderline rather than significant. This suggests that augmentation benefit is present but sensitive to inclusion of the small, high-variance levofloxacin dataset.
```
