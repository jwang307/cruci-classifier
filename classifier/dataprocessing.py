#!/usr/bin/env python3
"""
Parse FASTA files and produce CSVs with columns:
id,sequence,label  (label = 1 if 'cruci' in id else 0)
Usage:
    python convert_fasta_to_csv.py \
        --train data/train.fasta --test data/test.fasta --out_dir data
"""
import argparse, csv, pathlib
from Bio import SeqIO   # biopython

def fasta_to_rows(fasta_path):
    rows = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        label = 1 if "cruci" in rec.id.lower() else 0
        rows.append({"id": rec.id, "sequence": str(rec.seq), "label": label})
    return rows

def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sequence", "label"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(fasta_to_rows(args.train), out_dir / "train.csv")
    write_csv(fasta_to_rows(args.test),  out_dir / "test.csv")
    print("✓ CSVs written to", out_dir)
