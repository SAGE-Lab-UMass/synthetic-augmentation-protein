# ProteinGym Similarity100 Within-Protein Results

This note packages the validated per-protein ProteinGym experiment into a paper-ready summary.

## Experiment setup

- Selection rule: top-10-per-gene union from the `eligible269` MMseqs-guided MTB-similarity shortlist
- Final selected proteins: `100`
- Evaluation design: one model per ProteinGym protein
- Fold design: within-protein stratified 5-fold split across labeled variants
- Arms:
  - `baseline`
  - `single_mut`
  - `mut1_0p01`
  - `mut1_top50`
- Total valid result files: `500 / 500` per arm (`100 proteins x 5 folds`)

## Paper-ready summary table

| ProteinGym setting | Evaluation unit | Baseline | Single mut | Mut1 0p01 | Mut1 top50 | Single mut - baseline | 0p01 - baseline | Top50 - baseline | Note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Similarity100 within-protein 5-fold | mean protein AUC across 100 proteins | 0.6433 | 0.6661 | 0.6673 | 0.6709 | 0.0228 | 0.0240 | 0.0276 | All three augmentation arms outperform baseline; `top50` has the highest mean AUC, but differences among augmentation arms are small. |

## Paired protein-level statistics

| Comparison | Proteins | Mean delta | Median delta | Improved | Worse | Equal | Wilcoxon p | Paired t p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_mut - baseline` | 100 | 0.0228 | 0.0103 | 60 | 31 | 9 | 0.000721 | 0.000153 |
| `mut1_0p01 - baseline` | 100 | 0.0240 | 0.0112 | 61 | 33 | 6 | 0.000305 | 0.000042 |
| `mut1_top50 - baseline` | 100 | 0.0276 | 0.0167 | 61 | 34 | 5 | 0.000104 | 0.000037 |
| `mut1_0p01 - single_mut` | 100 | 0.0012 | 0.0000 | 37 | 30 | 33 | 0.330895 | 0.636960 |
| `mut1_top50 - single_mut` | 100 | 0.0047 | 0.0022 | 53 | 39 | 8 | 0.158495 | 0.254453 |
| `mut1_top50 - mut1_0p01` | 100 | 0.0036 | 0.0000 | 49 | 45 | 6 | 0.795148 | 0.412429 |

## Draft results paragraph

To create a ProteinGym evaluation that more closely parallels the MTB one-model-per-drug setting, we selected 100 ProteinGym proteins by taking the top-10-per-gene union from an MMseqs-guided MTB-similarity shortlist and then trained one model per protein using within-protein stratified 5-fold splits. Under this per-protein evaluation, all augmentation settings outperformed baseline. Mean protein AUC increased from `0.6433` for baseline to `0.6661` for `single_mut`, `0.6673` for `mut1_0p01`, and `0.6709` for `mut1_top50`. Relative to baseline, all three augmentation arms showed significant paired protein-level improvements (`single_mut`: Wilcoxon `p=7.21e-4`; `mut1_0p01`: `p=3.05e-4`; `mut1_top50`: `p=1.04e-4`). However, the differences among the three augmentation arms were small and not statistically significant, indicating that augmentation was beneficial in this MTB-guided per-protein ProteinGym setting, while the extra gain from ESM-based filtering over raw single-mutation augmentation was modest.
