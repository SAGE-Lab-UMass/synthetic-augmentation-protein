#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import shutil
import subprocess
import tempfile

CURRENT_PANEL = {
    "ethambutol": ["embA", "embB", "embC"],
    "ethionamide": ["ethA", "ethR", "inhA"],
    "isoniazid": ["inhA", "katG"],
    "levofloxacin": ["gyrA", "gyrB"],
    "moxifloxacin": ["gyrA", "gyrB"],
    "pyrazinamide": ["pncA"],
    "rifampicin": ["rpoB"],
    "streptomycin": ["gid", "rpsL"],
}

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def kmers(seq: str, k: int = 3) -> set[str]:
    seq = str(seq)
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def safe_fasta_id(value: str) -> str:
    return str(value).replace(' ', '_').replace('\t', '_')


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for record_id, sequence in records:
            handle.write(f'>{safe_fasta_id(record_id)}\n')
            handle.write(f'{sequence}\n')


def load_mtb_targets(data_dir: Path) -> list[dict[str, object]]:
    targets: list[dict[str, object]] = []
    for drug, genes in CURRENT_PANEL.items():
        rows = read_csv_rows(data_dir / f'{drug}.csv')
        for idx, gene in enumerate(genes, start=1):
            seq_col = f'seq_{idx}'
            seqs = [str(r.get(seq_col, '')).strip() for r in rows if str(r.get(seq_col, '')).strip()]
            if not seqs:
                continue
            representative, count = Counter(seqs).most_common(1)[0]
            targets.append({
                'drug': drug,
                'gene': gene,
                'seq_col': seq_col,
                'sequence': representative,
                'length': len(representative),
                'representative_count': count,
                'kmers': kmers(representative),
            })
    return targets


def load_proteingym_targets(orig_dir: Path) -> list[dict[str, object]]:
    proteins: list[dict[str, object]] = []
    for path in sorted(orig_dir.glob('*.csv')):
        rows = read_csv_rows(path)
        if not rows:
            continue
        first = rows[0]
        target_sequence = str(
            first.get('target_sequence') or first.get('protein_sequence') or first.get('seq_1') or ''
        ).strip()
        if not target_sequence:
            continue
        labels = [int(r['Phenotype']) for r in rows if str(r.get('Phenotype', '')).strip() != '']
        folds = sorted({int(r['is_val']) for r in rows if str(r.get('is_val', '')).strip() != ''})
        proteins.append({
            'protein_id': str(first.get('protein_id') or first.get('ProteinGym_protein_id') or path.stem),
            'target_sequence': target_sequence,
            'length': len(target_sequence),
            'n_total': len(labels),
            'n_positive': sum(labels),
            'n_negative': len(labels) - sum(labels),
            'folds': ','.join(map(str, folds)),
            'fold_count': len(folds),
            'kmers': kmers(target_sequence),
        })
    return proteins


def best_match_approximate(protein: dict[str, object], mtb_targets: list[dict[str, object]]) -> dict[str, object]:
    pseq = str(protein['target_sequence'])
    pkmers = protein['kmers']
    best = None
    for target in mtb_targets:
        tkmers = target['kmers']
        inter = len(pkmers & tkmers)
        union = len(pkmers | tkmers) or 1
        containment = inter / (len(pkmers) or 1)
        jaccard = inter / union
        length_ratio = min(len(pseq), int(target['length'])) / max(len(pseq), int(target['length']))
        preliminary = (0.6 * containment) + (0.3 * jaccard) + (0.1 * length_ratio)
        row = {
            'best_mtb_drug': target['drug'],
            'best_mtb_gene': target['gene'],
            'best_mtb_seq_col': target['seq_col'],
            'best_mtb_length': target['length'],
            'kmer_intersection': inter,
            'kmer_containment': containment,
            'kmer_jaccard': jaccard,
            'length_ratio': length_ratio,
            '_preliminary': preliminary,
            '_target_seq': target['sequence'],
        }
        if best is None or row['_preliminary'] > best['_preliminary']:
            best = row
    assert best is not None
    align_ratio = seq_ratio(pseq, str(best['_target_seq']))
    best['seqmatcher_ratio'] = align_ratio
    best['combined_score'] = (
        (0.5 * best['kmer_containment'])
        + (0.25 * best['kmer_jaccard'])
        + (0.15 * align_ratio)
        + (0.1 * best['length_ratio'])
    )
    del best['_preliminary']
    del best['_target_seq']
    return best


def run_mmseqs_search(
    proteins: list[dict[str, object]],
    mtb_targets: list[dict[str, object]],
    mmseqs_bin: str,
    threads: int,
    sensitivity: float,
    tmp_root: Path,
) -> dict[str, dict[str, object]]:
    target_index: dict[str, dict[str, object]] = {}
    for target in mtb_targets:
        target['target_id'] = f"{target['gene']}__{target['drug']}__{target['seq_col']}"
        target_index[str(target['target_id'])] = target

    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mmseqs_proteingym_', dir=tmp_root) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        query_fasta = tmp_dir_path / 'proteingym_queries.fasta'
        target_fasta = tmp_dir_path / 'mtb_targets.fasta'
        output_tsv = tmp_dir_path / 'search.tsv'
        mmseqs_tmp = tmp_dir_path / 'tmp'

        write_fasta(
            [(str(protein['protein_id']), str(protein['target_sequence'])) for protein in proteins],
            query_fasta,
        )
        write_fasta(
            [(str(target['target_id']), str(target['sequence'])) for target in mtb_targets],
            target_fasta,
        )

        cmd = [
            mmseqs_bin,
            'easy-search',
            str(query_fasta),
            str(target_fasta),
            str(output_tsv),
            str(mmseqs_tmp),
            '--format-mode', '4',
            '--format-output', 'query,target,evalue,bits,pident,alnlen,qcov,tcov,qlen,tlen,qstart,qend,tstart,tend,mismatch,gapopen',
            '--threads', str(threads),
            '-s', str(sensitivity),
            '-e', '1000000',
            '--prefilter-mode', '2',
            '--mask', '0',
            '--comp-bias-corr', '0',
            '--max-seqs', '1',
            '--max-accept', '1',
            '--sort-results', '1',
        ]
        subprocess.run(cmd, check=True)

        results: dict[str, dict[str, object]] = {}
        with output_tsv.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            for row in reader:
                protein_id = row['query']
                target = target_index[row['target']]
                results[protein_id] = {
                    'best_mtb_drug': target['drug'],
                    'best_mtb_gene': target['gene'],
                    'best_mtb_seq_col': target['seq_col'],
                    'best_mtb_length': target['length'],
                    'mmseqs_evalue': float(row['evalue']),
                    'mmseqs_bits': float(row['bits']),
                    'mmseqs_pident': float(row['pident']),
                    'mmseqs_alnlen': int(float(row['alnlen'])),
                    'mmseqs_qcov': float(row['qcov']),
                    'mmseqs_tcov': float(row['tcov']),
                    'mmseqs_qlen': int(float(row['qlen'])),
                    'mmseqs_tlen': int(float(row['tlen'])),
                    'mmseqs_qstart': int(float(row['qstart'])),
                    'mmseqs_qend': int(float(row['qend'])),
                    'mmseqs_tstart': int(float(row['tstart'])),
                    'mmseqs_tend': int(float(row['tend'])),
                    'mmseqs_mismatch': int(float(row['mismatch'])),
                    'mmseqs_gapopen': int(float(row['gapopen'])),
                    'combined_score': float(row['bits']),
                }

    if len(results) != len(proteins):
        missing = sorted({str(protein['protein_id']) for protein in proteins} - set(results))
        raise RuntimeError(
            f'MMseqs search returned no hit for {len(missing)} proteins; first few: {missing[:5]}'
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Rank ProteinGym proteins by one-way sequence similarity to MTB targets.')
    parser.add_argument('--mtb-data-dir', default=str(REPO_ROOT / 'datasets' / 'original_filtered'))
    parser.add_argument('--proteingym-orig-dir', default=str(REPO_ROOT / 'datasets' / 'proteingym_original'))
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--top-n', type=int, default=50)
    parser.add_argument('--method', choices=['auto', 'approximate', 'mmseqs'], default='auto')
    parser.add_argument('--mmseqs-bin', default='')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--sensitivity', type=float, default=5.7)
    parser.add_argument('--tmp-root', default='/scratch/login')
    args = parser.parse_args()

    mtb_targets = load_mtb_targets(Path(args.mtb_data_dir))
    proteins = load_proteingym_targets(Path(args.proteingym_orig_dir))

    mmseqs_bin = args.mmseqs_bin or shutil.which('mmseqs') or ''
    use_mmseqs = False
    if args.method == 'mmseqs':
        if not mmseqs_bin:
            raise RuntimeError('Requested --method mmseqs but no mmseqs binary was found')
        use_mmseqs = True
    elif args.method == 'auto' and mmseqs_bin:
        use_mmseqs = True

    mmseqs_matches: dict[str, dict[str, object]] = {}
    if use_mmseqs:
        mmseqs_matches = run_mmseqs_search(
            proteins=proteins,
            mtb_targets=mtb_targets,
            mmseqs_bin=mmseqs_bin,
            threads=args.threads,
            sensitivity=args.sensitivity,
            tmp_root=Path(args.tmp_root),
        )

    rows = []
    for protein in proteins:
        if use_mmseqs:
            match = mmseqs_matches[str(protein['protein_id'])]
        else:
            match = best_match_approximate(protein, mtb_targets)
        rows.append({
            'protein_id': protein['protein_id'],
            'protein_length': protein['length'],
            'n_total': protein['n_total'],
            'n_positive': protein['n_positive'],
            'n_negative': protein['n_negative'],
            'folds': protein['folds'],
            'fold_count': protein['fold_count'],
            'similarity_method': 'mmseqs' if use_mmseqs else 'approximate',
            **match,
        })

    rows.sort(
        key=lambda r: (
            -float(r['combined_score']),
            -float(r.get('mmseqs_pident', 0.0)),
            -int(r['n_total']),
            str(r['protein_id']),
        )
    )
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    top_path = out_path.with_name(out_path.stem + f'_top{args.top_n}.csv')
    with top_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows[:args.top_n])
    print(f'wrote {len(rows)} ranked proteins to {out_path}')
    print(f'wrote top {args.top_n} to {top_path}')


if __name__ == '__main__':
    main()
