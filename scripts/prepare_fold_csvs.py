#!/usr/bin/env python3
"""Prepare cleaned ESM-ready CSVs for a MachineLearning fold/domain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_DIR = REPO_ROOT / "classifier"
if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))

from dataprocessing import FIELDNAMES, fasta_to_rows  # noqa: E402


FOLD_NAMES = ("Blue", "Green", "Purple", "Red", "Yellow")
DOMAIN_PATTERNS = {
    "fullCP": ("fullCP/{fold}Training.fasta", "fullCP/{fold}Test.fasta"),
    "Rdomain": ("Rdomain/{fold}_R_Training.fasta", "Rdomain/{fold}_R_Test.fasta"),
    "Sdomain": ("Sdomain/{fold}_S_Training.fasta", "Sdomain/{fold}_S_Test.fasta"),
    "Pdomain": ("Pdomain/{fold}_P_Training.fasta", "Pdomain/{fold}_P_Test.fasta"),
}


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict]) -> dict:
    positives = sum(int(row["label"]) for row in rows)
    stops = sum(int(row["stops_removed"]) + int(row["stops_replaced"]) for row in rows)
    invalid = sum(int(row["invalid_removed"]) + int(row["invalid_replaced"]) for row in rows)
    lengths = [int(row["cleaned_length"]) for row in rows]
    return {
        "n": len(rows),
        "positives": positives,
        "negatives": len(rows) - positives,
        "min_len": min(lengths) if lengths else None,
        "max_len": max(lengths) if lengths else None,
        "stops_cleaned": stops,
        "invalid_cleaned": invalid,
    }


def prepare_one(
    *,
    data_root: Path,
    out_dir: Path,
    domain: str,
    fold: str,
    stop_action: str,
    invalid_action: str,
) -> dict:
    train_pattern, test_pattern = DOMAIN_PATTERNS[domain]
    train_fasta = data_root / train_pattern.format(fold=fold)
    test_fasta = data_root / test_pattern.format(fold=fold)
    if not train_fasta.exists():
        raise FileNotFoundError(train_fasta)
    if not test_fasta.exists():
        raise FileNotFoundError(test_fasta)

    fold_out = out_dir / domain / fold
    train_rows = fasta_to_rows(train_fasta, stop_action=stop_action, invalid_action=invalid_action)
    test_rows = fasta_to_rows(test_fasta, stop_action=stop_action, invalid_action=invalid_action)
    train_csv = fold_out / "train.csv"
    test_csv = fold_out / "test.csv"
    _write_csv(train_rows, train_csv)
    _write_csv(test_rows, test_csv)

    metadata = {
        "domain": domain,
        "fold": fold,
        "train_fasta": str(train_fasta),
        "test_fasta": str(test_fasta),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "train": _summary(train_rows),
        "test": _summary(test_rows),
    }
    with (fold_out / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/MachineLearning"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/prepared_folds"))
    parser.add_argument("--domain", choices=[*DOMAIN_PATTERNS.keys(), "all"], default="all")
    parser.add_argument("--fold", choices=[*FOLD_NAMES, "all"], default="all")
    parser.add_argument("--stop_action", choices=["remove", "replace_x", "error"], default="remove")
    parser.add_argument("--invalid_action", choices=["replace_x", "remove", "error"], default="replace_x")
    args = parser.parse_args()

    domains = DOMAIN_PATTERNS.keys() if args.domain == "all" else [args.domain]
    folds = FOLD_NAMES if args.fold == "all" else [args.fold]
    rows = []
    for domain in domains:
        for fold in folds:
            metadata = prepare_one(
                data_root=args.data_root,
                out_dir=args.out_dir,
                domain=domain,
                fold=fold,
                stop_action=args.stop_action,
                invalid_action=args.invalid_action,
            )
            rows.append(metadata)
            print(
                f"{domain}/{fold}: "
                f"train={metadata['train']['n']} ({metadata['train']['positives']}/"
                f"{metadata['train']['negatives']} pos/neg), "
                f"test={metadata['test']['n']} ({metadata['test']['positives']}/"
                f"{metadata['test']['negatives']} pos/neg)"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "prepared_summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
