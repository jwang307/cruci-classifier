#!/usr/bin/env python3
"""Audit the curated MachineLearning FASTA folds for ESM experiments."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_DIR = REPO_ROOT / "classifier"
if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))

from data_utils import clean_protein_sequence, label_from_identifier  # noqa: E402
from prepare_fold_csvs import DOMAIN_PATTERNS, FOLD_NAMES  # noqa: E402


def iter_fasta(path: Path):
    header = None
    seq_parts = []
    with path.open(errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/MachineLearning"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/dataset_audit"))
    parser.add_argument("--stop_action", choices=["remove", "replace_x", "error"], default="remove")
    parser.add_argument("--invalid_action", choices=["replace_x", "remove", "error"], default="replace_x")
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(args.data_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    record_rows: list[dict] = []
    counts_rows: list[dict] = []
    length_rows: list[dict] = []
    nonstandard_rows: list[dict] = []
    duplicate_rows: list[dict] = []
    split_overlap_rows: list[dict] = []

    for domain, (train_pattern, test_pattern) in DOMAIN_PATTERNS.items():
        for fold in FOLD_NAMES:
            split_records = {}
            for split, pattern in (("train", train_pattern), ("test", test_pattern)):
                path = args.data_root / pattern.format(fold=fold)
                records = []
                for header, raw_sequence in iter_fasta(path):
                    cleaned = clean_protein_sequence(
                        raw_sequence,
                        stop_action=args.stop_action,
                        invalid_action=args.invalid_action,
                    )
                    sequence_id = header.split()[0]
                    row = {
                        "domain": domain,
                        "fold": fold,
                        "split": split,
                        "file": str(path),
                        "sequence_id": sequence_id,
                        "header": header,
                        "label": label_from_identifier(sequence_id),
                        "raw_length": cleaned.original_length,
                        "cleaned_length": cleaned.cleaned_length,
                        "stops_removed": cleaned.stops_removed,
                        "stops_replaced": cleaned.stops_replaced,
                        "invalid_removed": cleaned.invalid_removed,
                        "invalid_replaced": cleaned.invalid_replaced,
                        "cleaned_sequence": cleaned.sequence,
                    }
                    records.append(row)
                    record_rows.append(row)
                    if (
                        cleaned.stops_removed
                        or cleaned.stops_replaced
                        or cleaned.invalid_removed
                        or cleaned.invalid_replaced
                    ):
                        nonstandard_rows.append({
                            key: row[key]
                            for key in [
                                "domain",
                                "fold",
                                "split",
                                "sequence_id",
                                "raw_length",
                                "cleaned_length",
                                "stops_removed",
                                "stops_replaced",
                                "invalid_removed",
                                "invalid_replaced",
                            ]
                        })

                split_records[split] = records
                labels = Counter(row["label"] for row in records)
                counts_rows.append({
                    "domain": domain,
                    "fold": fold,
                    "split": split,
                    "n": len(records),
                    "positives": labels.get(1, 0),
                    "negatives": labels.get(0, 0),
                })
                for label in (0, 1):
                    lengths = [row["cleaned_length"] for row in records if row["label"] == label]
                    if lengths:
                        length_rows.append({
                            "domain": domain,
                            "fold": fold,
                            "split": split,
                            "label": label,
                            "n": len(lengths),
                            "min": min(lengths),
                            "median": statistics.median(lengths),
                            "mean": statistics.mean(lengths),
                            "max": max(lengths),
                        })

                by_id = defaultdict(list)
                by_sequence = defaultdict(list)
                for row in records:
                    by_id[row["sequence_id"]].append(row)
                    by_sequence[row["cleaned_sequence"]].append(row)
                for duplicate_type, groups in (("id", by_id), ("cleaned_sequence", by_sequence)):
                    for key, duplicate_records in groups.items():
                        if len(duplicate_records) > 1:
                            duplicate_rows.append({
                                "domain": domain,
                                "fold": fold,
                                "split": split,
                                "duplicate_type": duplicate_type,
                                "duplicate_key": key if duplicate_type == "id" else hashlib.sha1(key.encode()).hexdigest(),
                                "count": len(duplicate_records),
                                "sequence_ids": ",".join(row["sequence_id"] for row in duplicate_records),
                            })

            train_ids = {row["sequence_id"] for row in split_records["train"]}
            test_ids = {row["sequence_id"] for row in split_records["test"]}
            train_sequences = {row["cleaned_sequence"] for row in split_records["train"]}
            test_sequences = {row["cleaned_sequence"] for row in split_records["test"]}
            train_ids_by_sequence = defaultdict(list)
            test_ids_by_sequence = defaultdict(list)
            for row in split_records["train"]:
                train_ids_by_sequence[row["cleaned_sequence"]].append(row["sequence_id"])
            for row in split_records["test"]:
                test_ids_by_sequence[row["cleaned_sequence"]].append(row["sequence_id"])
            for sequence_id in sorted(train_ids & test_ids):
                split_overlap_rows.append({
                    "domain": domain,
                    "fold": fold,
                    "overlap_type": "id",
                    "overlap_key": sequence_id,
                    "train_sequence_ids": sequence_id,
                    "test_sequence_ids": sequence_id,
                })
            for sequence in sorted(train_sequences & test_sequences):
                split_overlap_rows.append({
                    "domain": domain,
                    "fold": fold,
                    "overlap_type": "cleaned_sequence",
                    "overlap_key": hashlib.sha1(sequence.encode()).hexdigest(),
                    "train_sequence_ids": ",".join(sorted(train_ids_by_sequence[sequence])),
                    "test_sequence_ids": ",".join(sorted(test_ids_by_sequence[sequence])),
                })

    write_csv(
        args.out_dir / "sequence_counts_by_fold_domain_class.csv",
        counts_rows,
        ["domain", "fold", "split", "n", "positives", "negatives"],
    )
    write_csv(
        args.out_dir / "length_summary_by_fold_domain_class.csv",
        length_rows,
        ["domain", "fold", "split", "label", "n", "min", "median", "mean", "max"],
    )
    write_tsv(
        args.out_dir / "duplicate_sequences.tsv",
        duplicate_rows,
        ["domain", "fold", "split", "duplicate_type", "duplicate_key", "count", "sequence_ids"],
    )
    write_tsv(
        args.out_dir / "train_test_overlaps.tsv",
        split_overlap_rows,
        ["domain", "fold", "overlap_type", "overlap_key", "train_sequence_ids", "test_sequence_ids"],
    )
    write_tsv(
        args.out_dir / "nonstandard_residues.tsv",
        nonstandard_rows,
        [
            "domain",
            "fold",
            "split",
            "sequence_id",
            "raw_length",
            "cleaned_length",
            "stops_removed",
            "stops_replaced",
            "invalid_removed",
            "invalid_replaced",
        ],
    )

    total_records = len(record_rows)
    total_cleaned_stops = sum(row["stops_removed"] + row["stops_replaced"] for row in record_rows)
    total_invalid = sum(row["invalid_removed"] + row["invalid_replaced"] for row in record_rows)
    report = f"""## Dataset audit

Input: `{args.data_root}`

## Summary

- FASTA records checked: {total_records}
- Domain/fold/split count rows: {len(counts_rows)}
- Stop characters cleaned: {total_cleaned_stops}
- Unsupported residues cleaned: {total_invalid}
- Train/test ID or exact-sequence overlaps: {len(split_overlap_rows)}
- Within-file duplicate ID/exact-sequence groups: {len(duplicate_rows)}
- Records requiring cleanup: {len(nonstandard_rows)}

## Label convention

- `label = 1`: sequence ID contains `cruci`
- `label = 0`: all other sequence IDs

## Notes

- FASTA headers are not passed to ESM; only cleaned amino-acid sequences are tokenized.
- Full-capsid files contain stop characters that are removed before ESM tokenization.
- Metrics should account for severe fold imbalance, especially Red and Purple test folds.
"""
    (args.out_dir / "data_audit_report.md").write_text(report)

    dataset_summary = [{
        "records_checked": total_records,
        "count_rows": len(counts_rows),
        "stop_chars_cleaned": total_cleaned_stops,
        "unsupported_residues_cleaned": total_invalid,
        "train_test_overlaps": len(split_overlap_rows),
        "duplicate_groups": len(duplicate_rows),
        "records_requiring_cleanup": len(nonstandard_rows),
    }]
    write_csv(
        args.out_dir / "dataset_summary.csv",
        dataset_summary,
        [
            "records_checked",
            "count_rows",
            "stop_chars_cleaned",
            "unsupported_residues_cleaned",
            "train_test_overlaps",
            "duplicate_groups",
            "records_requiring_cleanup",
        ],
    )
    print(report)


if __name__ == "__main__":
    main()
