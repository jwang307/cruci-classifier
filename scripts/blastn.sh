#!/usr/bin/env bash

# → adjust these:
INPUT="/home/jwang/dna-gen/crucis/data/split_fastas"        # your original 900-seq FASTA
OUTDIR="/home/jwang/dna-gen/crucis/results/blastn_results"  # where to put the .tsv outputs
DB="nt"                            # remote database
SLEEP_SECS=2                       # delay between requests
MAX_TARGET_SEQS=100
EVALUE="1e-2"

mkdir -p "$OUTDIR"

echo "Found $(ls "$INPUT"/*.fasta | wc -l) sequences. Starting BLAST…"
for f in "$INPUT"/*.fasta; do
  filename=$(basename "$f")
  base="${filename%.fasta}"
  out="$OUTDIR/$base.tsv"

  echo "  ▶ blasting $filename → $base.tsv"
  blastn \
    -query "$f" \
    -db "$DB" \
    -remote \
    -out "$out" \
    -outfmt "6 qseqid sacc staxids sscinames pident length evalue bitscore" \
    -max_target_seqs "$MAX_TARGET_SEQS" \
    -evalue "$EVALUE"

  echo "    ✓ done."
  sleep "$SLEEP_SECS"
done

echo "All done. Results in $OUTDIR/"
