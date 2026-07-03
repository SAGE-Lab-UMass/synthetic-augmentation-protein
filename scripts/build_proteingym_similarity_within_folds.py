#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / 'datasets' / 'proteingym_mtb_similarity100_manifest.csv'
DEFAULT_ORIG_IN = REPO_ROOT / 'datasets' / 'proteingym_mtb_similarity100_original'
DEFAULT_AUG_IN = REPO_ROOT / 'datasets' / 'proteingym_mtb_similarity100_augmented'
DEFAULT_AUG_0P01_IN = REPO_ROOT / 'datasets' / 'proteingym_mtb_similarity100_augmented_0p01'
DEFAULT_AUG_TOP50_IN = REPO_ROOT / 'datasets' / 'proteingym_mtb_similarity100_augmented_top50'
DEFAULT_ORIG_OUT = REPO_ROOT / 'datasets' / 'proteingym_mtb_similarity100_within5_original'
DEFAULT_AUG_OUT = REPO_ROOT / 'datasets' / 'proteingym_mtb_similarity100_within5_augmented'
DEFAULT_AUG_0P01_OUT = REPO_ROOT / 'datasets' / 'proteingym_mtb_similarity100_within5_augmented_0p01'
DEFAULT_AUG_TOP50_OUT = REPO_ROOT / 'datasets' / 'proteingym_mtb_similarity100_within5_augmented_top50'
DEFAULT_SUMMARY = REPO_ROOT / 'datasets' / 'proteingym_mtb_similarity100_within5_summary.csv'


def assign_within_protein_folds(orig_df: pd.DataFrame, n_splits: int, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    work = orig_df.copy().reset_index(drop=True)
    labels = work['Phenotype'].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_ids = [-1] * len(work)
    for fold, (_, test_idx) in enumerate(splitter.split(work.index.to_numpy(), labels)):
        for idx in test_idx:
            fold_ids[idx] = fold
    work['is_val'] = fold_ids
    if 'fold' in work.columns:
        work['fold'] = fold_ids
    mapping = dict(zip(work['ProteinGym_variant_id'].astype(str), work['is_val'].astype(int)))
    return work, mapping


def apply_fold_mapping(aug_df: pd.DataFrame, fold_map: dict[str, int]) -> pd.DataFrame:
    work = aug_df.copy().reset_index(drop=True)

    def parent_key(row: pd.Series) -> str:
        if int(row['aug_mutations']) > 0:
            parent = str(row.get('augmentation_parent_variant_id', '')).strip()
            if parent:
                return parent
        return str(row.get('ProteinGym_variant_id') or row.get('variant_id') or row.get('Filename')).strip()

    keys = work.apply(parent_key, axis=1)
    missing = sorted({k for k in keys if k not in fold_map})
    if missing:
        raise ValueError(f'Missing fold assignments for {len(missing)} variants; first few: {missing[:5]}')
    work['is_val'] = [fold_map[k] for k in keys]
    if 'fold' in work.columns:
        work['fold'] = work['is_val']
    return work


def process_one(
    protein_id: str,
    orig_in: Path,
    aug_in: Path,
    aug0_in: Path,
    top_in: Path,
    orig_out: Path,
    aug_out: Path,
    aug0_out: Path,
    top_out: Path,
    n_splits: int,
    seed: int,
) -> dict[str, object]:
    orig_df = pd.read_csv(orig_in / f'{protein_id}.csv')
    aug_df = pd.read_csv(aug_in / f'{protein_id}_augmented.csv')
    aug0_df = pd.read_csv(aug0_in / f'{protein_id}_augmented.csv')
    top_df = pd.read_csv(top_in / f'{protein_id}_augmented.csv')

    orig_folded, fold_map = assign_within_protein_folds(orig_df, n_splits=n_splits, seed=seed)
    aug_folded = apply_fold_mapping(aug_df, fold_map)
    aug0_folded = apply_fold_mapping(aug0_df, fold_map)
    top_folded = apply_fold_mapping(top_df, fold_map)

    orig_out.mkdir(parents=True, exist_ok=True)
    aug_out.mkdir(parents=True, exist_ok=True)
    aug0_out.mkdir(parents=True, exist_ok=True)
    top_out.mkdir(parents=True, exist_ok=True)

    orig_folded.to_csv(orig_out / f'{protein_id}.csv', index=False)
    aug_folded.to_csv(aug_out / f'{protein_id}_augmented.csv', index=False)
    aug0_folded.to_csv(aug0_out / f'{protein_id}_augmented.csv', index=False)
    top_folded.to_csv(top_out / f'{protein_id}_augmented.csv', index=False)

    fold_counts = orig_folded.groupby(['is_val', 'Phenotype']).size().unstack(fill_value=0)
    summary = {
        'protein_id': protein_id,
        'original_rows': len(orig_folded),
        'augmented_rows': len(aug_folded),
        'augmented_0p01_rows': len(aug0_folded),
        'augmented_top50_rows': len(top_folded),
        'n_positive': int(orig_folded['Phenotype'].sum()),
        'n_negative': int(len(orig_folded) - orig_folded['Phenotype'].sum()),
    }
    for fold in range(n_splits):
        summary[f'fold_{fold}_neg'] = int(fold_counts.loc[fold, 0]) if fold in fold_counts.index and 0 in fold_counts.columns else 0
        summary[f'fold_{fold}_pos'] = int(fold_counts.loc[fold, 1]) if fold in fold_counts.index and 1 in fold_counts.columns else 0
    return summary


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    proteins = manifest['protein_id'].astype(str).tolist()
    summaries = []
    for protein_id in proteins:
        summaries.append(
            process_one(
                protein_id=protein_id,
                orig_in=Path(args.orig_in),
                aug_in=Path(args.aug_in),
                aug0_in=Path(args.aug0_in),
                top_in=Path(args.top_in),
                orig_out=Path(args.orig_out),
                aug_out=Path(args.aug_out),
                aug0_out=Path(args.aug0_out),
                top_out=Path(args.top_out),
                n_splits=args.n_splits,
                seed=args.seed,
            )
        )
    summary_df = pd.DataFrame(summaries).sort_values('protein_id')
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_csv, index=False)
    print(f'wrote within-protein datasets for {len(summary_df)} proteins')
    print(f'summary: {args.summary_csv}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build within-protein stratified folds for selected ProteinGym per-protein runs.')
    p.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    p.add_argument('--orig-in', default=str(DEFAULT_ORIG_IN))
    p.add_argument('--aug-in', default=str(DEFAULT_AUG_IN))
    p.add_argument('--aug0-in', default=str(DEFAULT_AUG_0P01_IN))
    p.add_argument('--top-in', default=str(DEFAULT_AUG_TOP50_IN))
    p.add_argument('--orig-out', default=str(DEFAULT_ORIG_OUT))
    p.add_argument('--aug-out', default=str(DEFAULT_AUG_OUT))
    p.add_argument('--aug0-out', default=str(DEFAULT_AUG_0P01_OUT))
    p.add_argument('--top-out', default=str(DEFAULT_AUG_TOP50_OUT))
    p.add_argument('--summary-csv', default=str(DEFAULT_SUMMARY))
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
