## Corrected eukaryotic cruci audit, 2026-06-14

## What is preserved

- Corrected run directory copied locally: `results/grid_esm35m_corrected_20260614/`.
- Hugging Face artifact prefix: `grid_esm35m_corrected_20260614` in `jwang003/cruci-esm35m-grid`.
- HF final upload commit recorded in `launcher.log`: `27abf9dc7a0d406ffda2a6d3f6ede312d0b9ddad`.
- Compact metric table: `results/grid_esm35m_corrected_20260614/aggregate_metrics_compact.csv`.
- Target prediction table: `results/predictions/corrected_eukaryotic_cruci_predictions.tsv`.
- Label audit tables: `results/dataset_audit/corrected_eukaryotic_cruci_label_audit.tsv` and `results/dataset_audit/corrected_eukaryotic_cruci_sequence_conflicts.tsv`.

## Corrected subset metrics

| model | test n | positives | AUROC | AUPRC | balanced acc | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Rdomain/Green` | 522 | 209 | 0.948438 | 0.942374 | 0.886696 | 0.864734 | 0.775959 |
| `Rdomain/Red` | 442 | 6 | 0.893731 | 0.544708 | 0.882263 | 0.243902 | 0.327627 |
| `Rdomain/Yellow` | 564 | 80 | 0.838765 | 0.667416 | 0.756147 | 0.475806 | 0.390827 |
| `fullCP/Green` | 561 | 231 | 0.889007 | 0.882125 | 0.797186 | 0.76044 | 0.597286 |

## Eukaryotic cruci held-out predictions

| model | ID | label | score | pred @0.5 | correct |
|---|---|---:|---:|---:|---:|
| `Rdomain/Green` | `Cruci_Cer_JAGYYM010003673` | 1 | 0.998 | 1 | True |
| `Rdomain/Red` | `Cruci_Carp_JAHDYR010000041` | 1 | 0.958 | 1 | True |
| `Rdomain/Yellow` | `Cruci_PolBeRBZT01000154.1` | 1 | 0.988 | 1 | True |
| `fullCP/Green` | `Cruci_Cer_JAGYYM010003673` | 1 | 0.992 | 1 | True |

## Dataset correction status

The held-out corrected examples are present as `Cruci_*` IDs and are predicted positive with high confidence.

The corrected tree is not fully clean yet. There are 16 residual target `Euk_*` records in `fullCP`/`Rdomain` FASTA files, and labels are inferred from whether the identifier contains `cruci`. These residual `Euk_*` rows are therefore still treated as negatives by the current pipeline.

Most important conflict:

- `Rdomain/Green_R_Training.fasta` contains `Euk_Cer_JAGYYM010003673` as label 0.
- `Rdomain/Green_R_Test.fasta` contains the exact same 144-aa sequence as `Cruci_Cer_JAGYYM010003673` with label 1.

## Interpretation

The corrected run supports this narrow claim: the held-out corrected eukaryotic cruci records are still classified as cruci-positive.

It does not yet support a fully clean final conclusion, because canonicalization and train/test duplicate filtering still need to be applied before rerunning the affected models.

## Recommended follow-up

- Canonicalize known eukaryotic cruci identifiers everywhere:
  - `Euk_Cer_JAGYYM010003673` to `Cruci_Cer_JAGYYM010003673`
  - `Euk_Carp_JAHDYR010000041` to `Cruci_Carp_JAHDYR010000041`
  - `Euk_PolBeRBZT01000154.1` to `Cruci_PolBeRBZT01000154.1`
- During corrected dataset build, fail or filter if the same canonical ID or exact cleaned sequence appears in both train and test for a fold.
- Rerun the affected subset after canonicalization: `Rdomain/Green`, `Rdomain/Red`, `Rdomain/Yellow`, and `fullCP/Green`.
