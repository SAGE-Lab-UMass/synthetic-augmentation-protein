# Diagnostic 00 Statistical Test Report: Including Levofloxacin

## Purpose

This analysis evaluates whether augmentation settings improve drug phenotype classification performance using all available drugs, including **levofloxacin**.

This is the primary full-drug analysis. A separate sensitivity analysis excluding levofloxacin is saved in:

```text
diagnostic/00_statistical_tests/results/augmentation_setting_tests_no_levofloxacin/
```

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

## Included Drugs

This full analysis includes all 8 drugs:

```text
ethambutol, ethionamide, isoniazid, levofloxacin, moxifloxacin, pyrazinamide, rifampicin, streptomycin
```

## Statistical Method

For each comparison, fold-level paired deltas were computed as:

```text
delta = AUC(experiment, drug, fold) - AUC(comparison setting, drug, fold)
```

Fold-level deltas were averaged within each drug. The 8 drug-level deltas were used as the main inference units, so each drug contributes equally to the overall test.

The primary p-value is the exact one-sided sign-flip p-value over drug-level deltas. The one-sided direction tests whether the experiment setting is better than the comparison setting.

## Multiple-Testing Correction

The output reports two Benjamini-Hochberg FDR corrections:

| Column | Meaning |
|---|---|
| `exact_sign_flip_q_bh_global` | FDR correction across all pairwise comparisons in this run |
| `exact_sign_flip_q_bh_family` | FDR correction only within the comparison's hypothesis family |

Family-level FDR is the preferred value for planned scientific questions because the tested comparisons belong to different hypothesis families.

Hypothesis families:

| Family | Question |
|---|---|
| `primary_esm_vs_baseline` | Are ESM-filtered augmentation settings better than baseline? |
| `secondary_no_esm_vs_baseline` | Is no-ESM augmentation better than baseline? |
| `esm_vs_no_esm` | Does ESM filtering improve over no-ESM augmentation? |
| `within_esm_settings` | Is one ESM-filtered setting better than another? |

## Overall Results

| Comparison | Family | Mean AUC Delta | 95% CI | p | Global q | Family q | Drugs Improved |
|---|---|---:|---:|---:|---:|---:|---:|
| `no_esm - baseline` | `secondary_no_esm_vs_baseline` | `+0.0097` | `[-0.0031, +0.0225]` | `0.1250` | `0.2083` | `0.1250` | `4/8` |
| `mut0.05_0.01 - baseline` | `primary_esm_vs_baseline` | `+0.0238` | `[+0.0070, +0.0450]` | `0.0195` | `0.0977` | `0.0293` | `7/8` |
| `mut1_0.01 - baseline` | `primary_esm_vs_baseline` | `+0.0244` | `[+0.0079, +0.0449]` | `0.0117` | `0.0977` | `0.0293` | `7/8` |
| `mut1_top50 - baseline` | `primary_esm_vs_baseline` | `+0.0256` | `[+0.0074, +0.0479]` | `0.0313` | `0.1042` | `0.0313` | `5/8` |
| `mut0.05_0.01 - no_esm` | `esm_vs_no_esm` | `+0.0141` | `[-0.0029, +0.0426]` | `0.2109` | `0.3013` | `0.2109` | `5/8` |
| `mut1_0.01 - no_esm` | `esm_vs_no_esm` | `+0.0147` | `[+0.0003, +0.0403]` | `0.0703` | `0.1758` | `0.1348` | `6/8` |
| `mut1_top50 - no_esm` | `esm_vs_no_esm` | `+0.0159` | `[-0.0001, +0.0438]` | `0.0898` | `0.1797` | `0.1348` | `4/8` |
| `mut1_0.01 - mut0.05_0.01` | `within_esm_settings` | `+0.0006` | `[-0.0040, +0.0056]` | `0.4063` | `0.4063` | `0.4063` | `4/8` |
| `mut1_top50 - mut0.05_0.01` | `within_esm_settings` | `+0.0018` | `[-0.0038, +0.0065]` | `0.2734` | `0.3418` | `0.4063` | `5/8` |
| `mut1_top50 - mut1_0.01` | `within_esm_settings` | `+0.0012` | `[-0.0034, +0.0061]` | `0.3438` | `0.3819` | `0.4063` | `4/8` |

## Interpretation

### 1. Primary Question: Are ESM-Filtered Settings Better Than Baseline?

Yes, in the full-drug analysis, the primary hypothesis family supports ESM-filtered augmentation over baseline.

The three ESM-filtered settings all show positive mean AUC improvements:

- `mut0.05_0.01 - baseline`: `+0.0238`
- `mut1_0.01 - baseline`: `+0.0244`
- `mut1_top50 - baseline`: `+0.0256`

Their family-level FDR q-values are below 0.05:

- `mut0.05_0.01 - baseline`: family q = `0.0293`
- `mut1_0.01 - baseline`: family q = `0.0293`
- `mut1_top50 - baseline`: family q = `0.0313`

Thus, for the planned primary question, all three ESM-filtered augmentation settings significantly improve validation AUC relative to no augmentation.

If using the more conservative global correction across all exploratory comparisons, these results are not significant (`global q ≈ 0.098-0.104`). Therefore, the interpretation should specify that significance is based on family-level correction for the primary hypothesis family.

### 2. Secondary Question: Does No-ESM Augmentation Beat Baseline?

The no-ESM setting has a smaller positive effect:

```text
no_esm - baseline: mean delta = +0.0097
```

But the confidence interval crosses zero and the p-value is not significant:

```text
p = 0.1250, family q = 0.1250
```

This suggests that augmentation without ESM filtering is not clearly better than baseline in this analysis.

### 3. Does ESM Filtering Improve Over No-ESM Augmentation?

The ESM-filtered settings trend higher than no-ESM augmentation:

- `mut0.05_0.01 - no_esm`: `+0.0141`
- `mut1_0.01 - no_esm`: `+0.0147`
- `mut1_top50 - no_esm`: `+0.0159`

However, none are significant after family-level correction:

- best family q = `0.1348`

Therefore, the data suggest a possible advantage of ESM filtering over no-ESM augmentation, but this evidence is not statistically significant.

### 4. Are Any ESM-Filtered Settings Better Than The Others?

No. Differences among ESM-filtered settings are very small:

- `+0.0006` to `+0.0018` AUC

All confidence intervals cross zero and family q-values are large. Therefore, there is no evidence that one ESM-filtered setting is meaningfully better than another.

## Drug-Specific Interpretation

The effect is heterogeneous across drugs.

Drugs that tend to benefit from augmentation include:

```text
ethionamide, levofloxacin, moxifloxacin, pyrazinamide
```

Drugs with weaker, inconsistent, or sometimes negative effects include:

```text
ethambutol, isoniazid, rifampicin, streptomycin
```

Levofloxacin contributes strongly to the average positive effect in the full-drug analysis, but it also has a small dataset and high fold-level variance. Therefore, this report should be interpreted alongside the no-levofloxacin sensitivity analysis.

## Final Conclusion

Including all 8 drugs, ESM-filtered augmentation significantly improves validation AUC over no augmentation for the planned primary hypothesis family after family-level FDR correction. The strongest setting by raw p-value is `mut1_0.01`, with a mean AUC improvement of `+0.0244`, improvement in `7/8` drugs, and family q = `0.0293`.

No-ESM augmentation does not show significant improvement over baseline. ESM-filtered settings trend higher than no-ESM augmentation, but the difference is not statistically significant. The three ESM-filtered settings perform similarly to each other.

A concise summary statement is:

```text
In the full-drug analysis, ESM-filtered augmentation significantly improved validation AUC over the no-augmentation baseline within the planned primary hypothesis family. This improvement was not observed for no-ESM augmentation, and no ESM-filtered setting was clearly superior to the others. Effects varied by drug, with levofloxacin contributing strongly to the full-drug signal.
```
