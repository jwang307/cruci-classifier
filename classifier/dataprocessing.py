#!/usr/bin/env python3
"""Parse FASTA files and produce ESM-ready labelled CSVs.

Output columns:
id,sequence,label,original_length,cleaned_length,stops_removed,stops_replaced,
invalid_removed,invalid_replaced

``label`` is 1 if "cruci" appears in the FASTA identifier, otherwise 0.
Usage:
    python classifier/dataprocessing.py \
        --train data/training_dataset_prot.fasta \
        --test data/test_dataset_prot.fasta \
        --out_dir data
"""
import argparse, csv, pathlib
from Bio import SeqIO   # biopython

try:
    from data_utils import clean_protein_sequence, label_from_identifier
except ImportError:  # Allows importing as classifier.dataprocessing.
    from classifier.data_utils import clean_protein_sequence, label_from_identifier


FIELDNAMES = [
    "id",
    "sequence",
    "label",
    "original_length",
    "cleaned_length",
    "stops_removed",
    "stops_replaced",
    "invalid_removed",
    "invalid_replaced",
]


def fasta_to_rows(
    fasta_path,
    *,
    stop_action: str = "remove",
    invalid_action: str = "replace_x",
    keep_empty: bool = False,
):
    rows = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        cleaned = clean_protein_sequence(
            str(rec.seq),
            stop_action=stop_action,
            invalid_action=invalid_action,
        )
        if not cleaned.sequence and not keep_empty:
            continue
        rows.append({
            "id": rec.id,
            "sequence": cleaned.sequence,
            "label": label_from_identifier(rec.id),
            "original_length": cleaned.original_length,
            "cleaned_length": cleaned.cleaned_length,
            "stops_removed": cleaned.stops_removed,
            "stops_replaced": cleaned.stops_replaced,
            "invalid_removed": cleaned.invalid_removed,
            "invalid_replaced": cleaned.invalid_replaced,
        })
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str)
    ap.add_argument("--test", type=str)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--stop_action", choices=["remove", "replace_x", "error"], default="remove")
    ap.add_argument("--invalid_action", choices=["replace_x", "remove", "error"], default="replace_x")
    ap.add_argument("--keep_empty", action="store_true", help="Keep records that are empty after cleanup.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.train:
        file_name = pathlib.Path(args.train).stem
        write_csv(
            fasta_to_rows(
                args.train,
                stop_action=args.stop_action,
                invalid_action=args.invalid_action,
                keep_empty=args.keep_empty,
            ),
            out_dir / f"{file_name}.csv",
        )
    if args.test:
        file_name = pathlib.Path(args.test).stem
        write_csv(
            fasta_to_rows(
                args.test,
                stop_action=args.stop_action,
                invalid_action=args.invalid_action,
                keep_empty=args.keep_empty,
            ),
            out_dir / f"{file_name}.csv",
        )
    print("✓ CSVs written to", out_dir)
