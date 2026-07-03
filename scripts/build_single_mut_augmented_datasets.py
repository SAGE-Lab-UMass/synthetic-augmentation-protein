#!/usr/bin/env python3
"""Build the current single-mutation augmented dataset root for MTB.

This wraps the stage-1 generator in 1_ESM_PIPELINE so the repo can materialize
the local augmented CSVs expected by the downstream ESM-filtered workflow.

Input:
    datasets/original_filtered/{drug}.csv

Output:
    datasets/augmented_1/{drug}_augmented.csv

Policy:
    - mutation-only augmentation
    - exactly one mutated position per non-empty sequence column
    - 10 augmented samples per original row
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "1_ESM_PIPELINE" / "1_generation" / "generate_augmented_data.py"
INPUT_DIR = REPO_ROOT / "datasets" / "original_filtered"
OUTPUT_DIR = REPO_ROOT / "datasets" / "augmented_1"

DRUGS = [
    "ethambutol",
    "ethionamide",
    "isoniazid",
    "levofloxacin",
    "moxifloxacin",
    "pyrazinamide",
    "rifampicin",
    "streptomycin",
]


def main() -> None:
    if not GENERATOR.exists():
        raise FileNotFoundError(f"Generator not found: {GENERATOR}")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for drug in DRUGS:
        cmd = [
            sys.executable,
            str(GENERATOR),
            "--drug",
            drug,
            "--input-dir",
            str(INPUT_DIR),
            "--output-dir",
            str(OUTPUT_DIR),
            "--aug-types",
            "M",
            "--num-aug-per-config",
            "10",
            "--seed",
            "42",
            "--single-mutation-only",
        ]
        print(f"[RUN] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    print(f"\nWrote augmented datasets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
