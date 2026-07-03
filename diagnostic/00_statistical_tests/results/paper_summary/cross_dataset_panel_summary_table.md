# Cross-Dataset Panel Summary Table

This note summarizes the panel-style MTB and ProteinGym results used for paper drafting, and records how the ProteinGym panels were constructed to parallel the MTB drug-panel workflow.

## ProteinGym Data Preparation

### Full scored source

- Starting point: the full ProteinGym clinical variant dataset adapted into the MTB pipeline schema.
- Original labeled rows: `62,727`
- Proteins: `2,525`
- Augmented rows including originals after single-mutation generation: `125,454`
- All panel datasets below were materialized by subsetting this already scored augmented file; we did not rescore separately for each panel.

### PPPL / ESM scoring completion time

The full ProteinGym augmented file was completed with the sharded resume workflow.

Successful final completion pass:

- Shard jobs: `60465571`, `60465572`, `60465573`, `60465574`
- Merge/filter job: `60465575`
- First shard start: `2026-06-04 16:52:37 UTC`
- Final merge end: `2026-06-05 10:27:24 UTC`
- End-to-end wall clock for the successful final scoring pass: about `17 hours 35 minutes`
- Aggregate shard+merge elapsed time summed across the final successful four shard jobs plus merge: about `53 hours 54 minutes`

Full accumulated scoring effort across the broader resume / timeout history was larger than that final pass alone:

- Earlier long-running scoring jobs included `56836469`, `57799270`, `57981960`, `58140764`, `58768001`, `58804544`, `59738332`, `60087069`, `60087070`, `60087071`, `60087072`
- When those earlier attempts are added to the final successful shard+merge pass, the accumulated elapsed runtime is about `19 days 3 hours`

### Eligibility-defined ProteinGym panels

We expanded the ProteinGym analysis by defining eligibility rules at the protein level, then subsetting the already scored ProteinGym files.

#### Eligible-269 panel

Eligibility rule:

- `n_total >= 30`
- `n_positive >= 10`
- `n_negative >= 10`

Resulting panel:

- Proteins: `269`
- Original rows: `25,385`
- Augmented rows including originals: `50,770`
- `0p01` rows: `48,992`
- Top50-global rows: `39,218`
- Fold protein counts: `53, 55, 52, 55, 54`

#### Eligible-532 panel

Eligibility rule:

- `n_total >= 20`
- `n_positive >= 5`
- `n_negative >= 5`

Resulting panel:

- Proteins: `532`
- Original rows: `36,545`
- Augmented rows including originals: `73,090`
- `0p01` rows: `69,701`
- Top50-global rows: `55,667`
- Fold protein counts: `103, 110, 105, 103, 111`

## ProteinGym Panel Construction Summary

| ProteinGym panel | Eligibility threshold | No. proteins | No. original rows | No. augmented rows | No. `0p01` rows | No. top-50 rows | Foldwise protein counts |
|---|---|---:|---:|---:|---:|---:|---|
| Eligible-269 | `n_total >= 30`, `n_positive >= 10`, `n_negative >= 10` | 269 | 25,385 | 50,770 | 48,992 | 39,218 | 53, 55, 52, 55, 54 |
| Eligible-532 | `n_total >= 20`, `n_positive >= 5`, `n_negative >= 5` | 532 | 36,545 | 73,090 | 69,701 | 55,667 | 103, 110, 105, 103, 111 |

## How This Parallels The MTB Drug Panel

We used ProteinGym in a panel-style way rather than as one fully pooled benchmark.

The parallel to MTB was:

- MTB unit of analysis: `drug`
- ProteinGym panel unit of analysis: one shared `proteingym` panel built from many eligible proteins
- Same four augmentation arms:
  - `baseline`
  - `single_mut` / raw augmentation
  - `mut1_0p01`
  - `mut1_top50`
- Same `is_val` 5-fold training/evaluation structure already present in the pipeline
- Same `run_all.py` / `train.py` training path used for the panel experiments
- Same model family: `resnet`
- Same end metric: fold-level `val_auc`
- Same paired-fold comparison style used for the ProteinGym panel summaries

This is not identical to MTB's across-drug meta-analysis, because ProteinGym does not naturally provide eight separate drug tasks. But it preserves the same panel-style logic: compare augmentation settings on a multi-target panel using the same pipeline and fold structure.

## Evaluation Flow

For each ProteinGym panel, we followed the same steps:

1. Start from the original ProteinGym-adapted CSV and the already scored augmented CSV.
2. Select eligible proteins by the predeclared protein-level thresholds.
3. Materialize four dataset roots:
   - original / baseline root
   - full single-mutation augmented root
   - `0p01` filtered root
   - top50 filtered root
4. Train the same four augmentation arms with the panel-style ProteinGym Slurm wrappers.
5. Collect 5 fold AUCs per arm from `final_results.json`.
6. Summarize mean fold AUC and paired fold deltas.

## Cross-Dataset Summary Table

| Dataset / panel | Evaluation unit | Baseline | Single mut / raw aug | Mut1 0p01 | Mut1 top50 | Single mut - baseline | 0p01 - baseline | Top50 - baseline | Note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| MTB 8-drug panel | mean drug-fold AUC across 8 drugs x 5 folds | 0.7500 | 0.7597 | 0.7744 | 0.7756 | 0.0097 | 0.0244 | 0.0256 | ESM-filtered settings are strongest on MTB; family-level q < 0.05 in the primary MTB analysis. |
| ProteinGym eligible-269 panel | mean fold AUC across 5 folds | 0.5543 | 0.5711 | 0.5358 | 0.5688 | 0.0167 | -0.0185 | 0.0144 | single_mut is highest by mean AUC; top50 is close; 0p01 is unstable; no paired comparison is significant with 5 folds. |
| ProteinGym eligible-532 panel | mean fold AUC across 5 folds | 0.6207 | 0.6231 | 0.6132 | 0.6159 | 0.0024 | -0.0075 | -0.0048 | All four arms are very close; single_mut is only marginally above baseline; both ESM-filtered arms are slightly below baseline on average. |
| ProteinGym similarity100 within-protein | mean protein AUC across 100 proteins | 0.6433 | 0.6661 | 0.6673 | 0.6709 | 0.0228 | 0.0240 | 0.0276 | MTB-guided per-protein selection with within-protein 5-fold evaluation; all augmentation arms significantly outperform baseline, while differences among augmentation arms remain small. |

## Interpretation

- MTB shows the clearest augmentation benefit, with the strongest means for the ESM-filtered settings.
- ProteinGym eligible-269 weakens the story: single_mut is best by mean AUC, top50 is close, and 0p01 is unstable.
- ProteinGym eligible-532 weakens it further: all four arms are nearly tied, with only a tiny mean advantage for single_mut.
- ProteinGym similarity100 within-protein restores a clearer augmentation signal in a truly per-protein setup: all three augmentation arms outperform baseline, and top50 has the highest mean AUC.
- Across ProteinGym panel expansions, the apparent augmentation advantage attenuates as panel breadth increases.
- A more MTB-like per-protein ProteinGym evaluation gives a more favorable result than the broad pooled-panel ProteinGym analyses, suggesting that task framing materially changes the apparent transfer signal.
