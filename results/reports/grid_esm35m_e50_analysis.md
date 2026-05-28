## Summary

The five-cluster ESM2-35M grid completed for all 20 `domain x fold` models.

The strongest result is that both full capsid and R-domain models generalize under phylogenetic holdout:

| Input | Mean AUROC | Mean AUPRC | Mean balanced acc. | Mean F1 | Mean MCC |
|---|---:|---:|---:|---:|---:|
| fullCP | 0.898 | 0.838 | 0.811 | 0.652 | 0.479 |
| Rdomain | 0.900 | 0.784 | 0.818 | 0.655 | 0.517 |
| Sdomain | 0.736 | 0.609 | 0.624 | 0.530 | 0.243 |
| Pdomain | 0.724 | 0.584 | 0.656 | 0.515 | 0.234 |

Interpretation: the full capsid signal survives the stricter tree/cluster holdout, and the R domain carries the strongest domain-localized signal.

## Result provenance

Remote/HF run:

```text
HF repo: https://huggingface.co/datasets/jwang003/cruci-esm35m-grid
HF prefix: grid_esm35m/
local remote results path: /workspace/cruci-classifier/results/grid_esm35m_e50
completed models: 20 / 20
best_head.pt: 20
final_head.pt: 20
training_summary.json: 20
test_metrics.json: 20
all_oof_predictions.csv rows: 9875 predictions + header
aggregate_metrics.csv rows: 20 + header
```

All runs used:

```text
ESM model: esm2_t12_35M_UR50D
trained parameters: linear classifier head only
frozen parameters: all ESM encoder weights
epochs: 50
batch size: 32
learning rate: 1e-3
optimizer: Adam
loss: BCEWithLogitsLoss
positive class weight: n_negative / n_positive
validation: stratified 15% split from training clusters only
test: held-out phylogenetic cluster
max_steps: unset
early stopping: none
checkpoint selection: best_head.pt by validation F1 at threshold 0.5
final checkpoint: final_head.pt after epoch 50
```

## Fold-by-fold metrics

| Input | Fold | Best epoch | Steps | Test positives | Test negatives | AUROC | AUPRC | Balanced acc. | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fullCP | Blue | 48 | 2800 | 238 | 169 | 0.910 | 0.931 | 0.825 | 0.860 | 0.654 |
| fullCP | Green | 39 | 2600 | 230 | 330 | 0.889 | 0.881 | 0.797 | 0.759 | 0.596 |
| fullCP | Purple | 42 | 2650 | 445 | 56 | 0.831 | 0.976 | 0.734 | 0.872 | 0.342 |
| fullCP | Red | 39 | 2750 | 6 | 440 | 0.944 | 0.630 | 0.847 | 0.139 | 0.225 |
| fullCP | Yellow | 47 | 2550 | 83 | 485 | 0.914 | 0.774 | 0.853 | 0.628 | 0.575 |
| Rdomain | Blue | 40 | 2700 | 238 | 170 | 0.936 | 0.949 | 0.879 | 0.903 | 0.762 |
| Rdomain | Green | 49 | 2550 | 208 | 313 | 0.948 | 0.942 | 0.886 | 0.864 | 0.775 |
| Rdomain | Purple | 37 | 2600 | 442 | 56 | 0.902 | 0.986 | 0.804 | 0.908 | 0.470 |
| Rdomain | Red | 40 | 2650 | 5 | 436 | 0.871 | 0.382 | 0.767 | 0.162 | 0.218 |
| Rdomain | Yellow | 47 | 2500 | 79 | 484 | 0.843 | 0.663 | 0.752 | 0.436 | 0.359 |
| Sdomain | Blue | 50 | 2800 | 238 | 170 | 0.753 | 0.814 | 0.667 | 0.768 | 0.366 |
| Sdomain | Green | 44 | 2600 | 231 | 330 | 0.925 | 0.897 | 0.768 | 0.703 | 0.610 |
| Sdomain | Purple | 40 | 2650 | 444 | 56 | 0.633 | 0.932 | 0.544 | 0.848 | 0.068 |
| Sdomain | Red | 46 | 2750 | 5 | 440 | 0.671 | 0.058 | 0.545 | 0.037 | 0.031 |
| Sdomain | Yellow | 49 | 2550 | 82 | 485 | 0.697 | 0.342 | 0.596 | 0.297 | 0.141 |
| Pdomain | Blue | 36 | 2800 | 238 | 170 | 0.619 | 0.650 | 0.585 | 0.586 | 0.168 |
| Pdomain | Green | 40 | 2600 | 231 | 330 | 0.906 | 0.887 | 0.841 | 0.813 | 0.677 |
| Pdomain | Purple | 47 | 2650 | 444 | 56 | 0.635 | 0.940 | 0.525 | 0.793 | 0.035 |
| Pdomain | Red | 34 | 2750 | 5 | 440 | 0.738 | 0.060 | 0.692 | 0.058 | 0.098 |
| Pdomain | Yellow | 49 | 2550 | 82 | 485 | 0.723 | 0.381 | 0.635 | 0.326 | 0.190 |

## Does full capsid ESM distinguish classes under phylogenetic holdout?

Yes.

Full capsid has strong cross-cluster ranking:

```text
mean AUROC = 0.898
mean AUPRC = 0.838
mean balanced accuracy = 0.811
```

The signal is not confined to one fold: AUROC is at least 0.831 in every held-out cluster.

The full-capsid model is weaker than the earlier non-five-cluster result in some metrics, but the split is stricter. This supports the conclusion that there is a generalizable capsid-level class signal rather than only random-split homology leakage.

## Which domain carries the strongest signal?

R domain is the strongest domain-specific result.

R-domain performance is essentially tied with full capsid by AUROC and slightly higher by balanced accuracy/MCC:

```text
Rdomain mean AUROC = 0.900
Rdomain mean balanced accuracy = 0.818
Rdomain mean MCC = 0.517
fullCP mean AUROC = 0.898
fullCP mean balanced accuracy = 0.811
fullCP mean MCC = 0.479
```

This supports the biological hypothesis that a nucleic-acid-associated region carries crucivirus-like versus RNA-virus-like signatures.

This is stronger than the previous domain-only inference experiment because the R-domain model was trained and tested on R-domain inputs rather than applying a whole-capsid model out of distribution.

## Are S and P domains informative?

S and P domains are weaker and fold-specific.

Sdomain:

- strong on Green (`AUROC 0.925`),
- moderate on Blue (`0.753`),
- weak on Purple/Red/Yellow (`0.633/0.671/0.697`).

Pdomain:

- strong on Green (`AUROC 0.906`),
- moderate on Red/Yellow (`0.738/0.723`),
- weak on Blue/Purple (`0.619/0.635`).

Interpretation: S/P likely contain some class-correlated information in specific clades, but not a stable universal signature across the five-cluster split. This could reflect weaker biological signal, domain-boundary noise, stronger lineage specificity, or length/composition artifacts.

## Are results robust across folds?

Full capsid and R domain are robust by AUROC; hard-threshold metrics are heavily affected by fold imbalance.

Important fold imbalance:

| Fold | Example imbalance |
|---|---|
| Red | only 5-6 positive examples, ~440 negatives |
| Purple | ~444-445 positives, 56 negatives |
| Yellow | 79-83 positives, ~484-485 negatives |

This explains why F1 can look low in Red despite strong ranking. For example:

```text
fullCP Red AUROC = 0.944
fullCP Red balanced accuracy = 0.847
fullCP Red F1 = 0.139
```

F1 is unstable with only six positives. AUROC, balanced accuracy, MCC, and the continuous scores should be prioritized.

## Threshold behavior

Validation-selected thresholds are generally close to 0.5 for fullCP/R/P, but S-domain thresholds can drift lower:

```text
Sdomain Blue threshold = 0.395
Sdomain Red threshold = 0.401
```

Interpretation: fullCP and R scores are better calibrated around the default boundary than S-domain scores. For tree interpretation, use continuous scores rather than hard labels.

## Biological conclusions and conjectures

1. Full capsid proteins contain a reproducible signal distinguishing crucivirus and RNA-virus labels under phylogenetic holdout.

2. The R domain is the strongest localized signal. This supports the conjecture that class signal may relate to nucleic-acid interaction, genome packaging, or inner capsid features rather than being purely surface/host-facing.

3. Full capsid and R-domain performance being similar suggests that much of the full-capsid signal may be recoverable from R alone. The key next biological table should therefore prioritize sequences where fullCP and R scores agree on discordance.

4. S and P signals are not absent, but they appear clade-specific. The strong Green-fold S/P performance suggests some clusters may carry domain-specific lineage features, but this is not yet evidence for a universal S/P mechanism.

5. Candidate transfer hypotheses should be sequence/clade-specific, not global. The model supports the existence of signatures but does not by itself imply one global transfer direction.

## Strongest caveats

1. Baselines are not complete. Length-only, amino-acid composition, k-mer, frozen-ESM-logreg, label-shuffle, and nearest-neighbor identity baselines are still needed before claiming the ESM signal is nontrivial.

2. R-domain data had two exact train/test overlaps with opposite labels in the audit. They are confined to Rdomain Blue/Green and should be removed or manually adjudicated before final claims.

3. Red/Purple folds are severely imbalanced. Metrics that depend on hard thresholds are unstable in these folds.

4. Clade-level transfer interpretation requires tree mapping. The current grid gives out-of-fold scores, but it does not by itself identify clade-structured discordance without the phylogeny.

5. The HF dataset is private and the cloud instance was no longer reachable during this writeup, so sequence-level high-confidence discordant tables were not regenerated locally in this pass. They should be generated directly from the uploaded `all_oof_predictions.csv`.

## Immediate next analyses

1. Generate the tree annotation tables:

```text
all_model_long_format_predictions.tsv
tree_annotation_scores_wide.tsv
high_confidence_discordant_sequences.tsv
per_cluster_score_summary.tsv
```

High-confidence discordance:

```text
label 1 and score < 0.2  # crucivirus scored RNA-virus-like
label 0 and score > 0.8  # RNA virus scored crucivirus-like
```

Highest-priority candidates are sequences discordant in both fullCP and Rdomain.

2. Run baselines on the same five folds:

- length-only logistic regression,
- amino-acid composition logistic regression,
- 2-mer/3-mer linear model,
- frozen ESM embedding + logistic regression,
- label-shuffle negative control,
- nearest-neighbor sequence-identity classifier.

3. Re-run Rdomain Blue/Green after removing/adjudicating the exact cross-label duplicate pair.

4. Add bootstrap confidence intervals by fold/domain, especially for Red and Purple.

5. Map fullCP and Rdomain continuous scores onto the tree. Look for clade-structured discordance rather than isolated mistakes.

6. If baselines do not explain the signal, proceed to R-domain interpretability:

- sliding-window masking,
- charged/aromatic/basic residue enrichment,
- motif/k-mer enrichment in high-score vs low-score R domains,
- structure mapping of discriminative residues.

## Practical recommendation

For the collaborator update, frame the result as:

> Under a five-cluster phylogenetic holdout, the frozen-ESM2-35M classifier head detects a robust crucivirus-vs-RNA-virus signal in full capsids. Domain-specific training shows that the R domain carries the strongest localized signal, roughly matching full capsid performance by AUROC and balanced accuracy. This supports the R-domain/nucleic-acid-interaction hypothesis, but baseline controls and tree-mapped discordance analysis are required before making mechanistic or transfer-direction claims.

