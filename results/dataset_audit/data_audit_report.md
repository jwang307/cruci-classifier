## Dataset audit

Input: `data/MachineLearning`

## Summary

- FASTA records checked: 49420
- Domain/fold/split count rows: 40
- Stop characters cleaned: 10140
- Unsupported residues cleaned: 0
- Train/test ID or exact-sequence overlaps: 2
- Within-file duplicate ID/exact-sequence groups: 1193
- Records requiring cleanup: 10120

## Label convention

- `label = 1`: sequence ID contains `cruci`
- `label = 0`: all other sequence IDs

## Notes

- FASTA headers are not passed to ESM; only cleaned amino-acid sequences are tokenized.
- Full-capsid files contain stop characters that are removed before ESM tokenization.
- Metrics should account for severe fold imbalance, especially Red and Purple test folds.
