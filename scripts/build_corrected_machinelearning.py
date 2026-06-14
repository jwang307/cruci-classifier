#!/usr/bin/env python3
"""Build a MachineLearning-compatible tree from corrected FASTA uploads.

The corrected data currently contains replacement files for:

- fullCP Green train/test
- Rdomain Green train/test
- Rdomain Red train/test
- Rdomain Yellow train/test

This script copies the original MachineLearning tree and overlays those files
using the filenames expected by ``scripts/prepare_fold_csvs.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


OVERLAYS = {
    "GreenTraining.fasta": "fullCP/GreenTraining.fasta",
    "correctedGreenTest.txt": "fullCP/GreenTest.fasta",
    "Green_R_Training.fasta": "Rdomain/Green_R_Training.fasta",
    "correctedGreen_R_Test.fasta": "Rdomain/Green_R_Test.fasta",
    "correctedRed_R_Training.fasta": "Rdomain/Red_R_Training.fasta",
    "correctedRed_R_Test.fasta": "Rdomain/Red_R_Test.fasta",
    "correctedYellow_R_Training.fasta": "Rdomain/Yellow_R_Training.fasta",
    "correctedYellow_R_Test.fasta": "Rdomain/Yellow_R_Test.fasta",
}


def count_fasta_records(path: Path) -> int:
    """Count FASTA records by header lines."""
    return sum(1 for line in path.open(errors="replace") if line.startswith(">"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_root", type=Path, default=Path("data/MachineLearning"))
    parser.add_argument("--corrected_root", type=Path, default=Path("data/corrected_data"))
    parser.add_argument("--out_root", type=Path, default=Path("data/MachineLearning_corrected"))
    args = parser.parse_args()

    if not args.base_root.exists():
        raise FileNotFoundError(args.base_root)
    if not args.corrected_root.exists():
        raise FileNotFoundError(args.corrected_root)

    shutil.copytree(args.base_root, args.out_root, dirs_exist_ok=True)
    overlays = []
    for source_name, destination_name in OVERLAYS.items():
        source = args.corrected_root / source_name
        destination = args.out_root / destination_name
        if not source.exists():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        overlays.append(
            {
                "source": str(source),
                "destination": str(destination),
                "records": count_fasta_records(destination),
            }
        )

    manifest = {
        "base_root": str(args.base_root),
        "corrected_root": str(args.corrected_root),
        "out_root": str(args.out_root),
        "overlays": overlays,
    }
    manifest_path = args.out_root / "corrected_overlay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
