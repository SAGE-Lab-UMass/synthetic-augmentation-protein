# Full ProteinGym Statistical Summary

## Purpose

This report compares the completed full ProteinGym 5-fold training arms. Because there are only five paired folds, all p-values are descriptive and have low resolution; they should not be treated as strong inferential evidence.

## Arms

| Arm | Meaning |
|---|---|
| `baseline` | no augmentation |
| `single_mut` | unfiltered synthetic single-mutation augmentation |
| `mut1_0p01_full` | full ProteinGym pool filtered at norm_score <= 0.01 |
| `mut1_top50_full` | full ProteinGym pool retaining the top 50% by ESM score |

## Mean Fold AUC

| arm | count | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| baseline | 5 | 0.639266 | 0.036037 | 0.595576 | 0.671864 |
| mut1_0p01_full | 5 | 0.634090 | 0.041246 | 0.587626 | 0.678402 |
| mut1_top50_full | 5 | 0.626829 | 0.044012 | 0.574145 | 0.668403 |
| single_mut | 5 | 0.641196 | 0.036077 | 0.598751 | 0.669147 |

## Paired Comparisons

| comparison | mean delta | folds improved | sign-flip p (greater) | sign-flip p (two-sided) | Wilcoxon p (greater) | Wilcoxon p (two-sided) |
|---|---:|---:|---:|---:|---:|---:|
| single_mut - baseline | 0.001930 | 3/5 | 0.25000 | 0.50000 | 0.31250 | 0.62500 |
| mut1_0p01_full - baseline | -0.005176 | 2/5 | 0.78125 | 0.50000 | 0.78125 | 0.62500 |
| mut1_top50_full - baseline | -0.012437 | 1/5 | 0.90625 | 0.25000 | 0.90625 | 0.31250 |
| mut1_0p01_full - single_mut | -0.007106 | 1/5 | 0.90625 | 0.25000 | 0.90625 | 0.31250 |
| mut1_top50_full - single_mut | -0.014367 | 1/5 | 0.96875 | 0.12500 | 0.96875 | 0.12500 |
| mut1_top50_full - mut1_0p01_full | -0.007261 | 1/5 | 0.93750 | 0.18750 | 0.93750 | 0.18750 |

## Interpretation

The full-dataset ordering is:

```text
mut1_top50_full < mut1_0p01_full < baseline < single_mut
```

Unfiltered single-mutation augmentation is nearly flat relative to baseline (+0.00193 mean AUC; 3/5 folds improved). The full ESM-filtered arms do not reproduce the selected 15-protein panel pilot improvement: `mut1_0p01_full` is -0.00518 below baseline and `mut1_top50_full` is -0.01244 below baseline. Neither filtered arm improves consistently across folds, and no comparison provides strong paired-fold evidence after accounting for the five-fold sample size.

This result should be reported as a contrast between the selected panel pilot and the full ProteinGym benchmark, not as evidence that ESM filtering improves full ProteinGym performance.
