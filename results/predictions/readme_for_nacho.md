## Files

Use these tables for tree annotation and candidate follow-up:

```text
tree_annotation_scores_wide.tsv
all_model_long_format_predictions.tsv
high_confidence_discordant_sequences.tsv
moderate_discordant_sequences.tsv
fullCP_Rdomain_agreeing_discordant_sequences.tsv
per_cluster_score_summary.tsv
```

## Score convention

```text
label 0 = RNA virus
label 1 = crucivirus
score near 0 = RNA-virus-like capsid/domain signature
score near 1 = crucivirus-like capsid/domain signature
```

Score bands:

| Score | Interpretation |
|---:|---|
| 0.0-0.2 | strongly RNA-virus-like |
| 0.2-0.4 | moderately RNA-virus-like |
| 0.4-0.6 | ambiguous |
| 0.6-0.8 | moderately crucivirus-like |
| 0.8-1.0 | strongly crucivirus-like |

## Best table for tree coloring

Use:

```text
tree_annotation_scores_wide.tsv
```

Each row is a sequence ID with full capsid, R, S, and P domain scores where available.

Recommended columns for first tree plots:

```text
true_label
virus_type
cluster_id
full_capsid_score
R_domain_score
full_R_mean_score
full_R_both_high_confidence_discordant
full_R_both_moderate_discordant
```

## Candidate discordance criteria

High-confidence discordance:

```text
crucivirus with score < 0.2
RNA virus with score > 0.8
```

Moderate discordance:

```text
crucivirus with score < 0.4
RNA virus with score > 0.6
```

Highest-priority candidate rows are in:

```text
fullCP_Rdomain_agreeing_discordant_sequences.tsv
```

These sequences are discordant in both full capsid and R-domain models.

## Caveats

- Classifier discordance is not proof of transfer direction.
- Clade-structured discordance is more meaningful than isolated discordance.
- Rdomain Blue/Green include a known exact cross-label duplicate-pair caveat from the dataset audit.
- Simple baselines and nearest-neighbor identity controls are still needed.
- Source FASTA IDs are not perfectly unique; the long-format table preserves individual records with `sequence_record_id`, while the wide tree table aggregates duplicate raw IDs for tree-label compatibility.
