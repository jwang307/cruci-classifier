## Sequence-level discordance analysis

Inputs:

```text
results/grid_esm35m_e50/all_oof_predictions.csv
results/grid_esm35m_e50/aggregate_metrics.csv
data/MachineLearning/
```

Outputs:

```text
results/predictions/all_model_long_format_predictions.tsv
results/predictions/tree_annotation_scores_wide.tsv
results/predictions/high_confidence_discordant_sequences.tsv
results/predictions/moderate_discordant_sequences.tsv
results/predictions/fullCP_Rdomain_agreeing_discordant_sequences.tsv
results/predictions/per_cluster_score_summary.tsv
```

## Counts

```text
long prediction rows: 9875
unique sequence ids: 2483
wide tree rows: 2483
high-confidence discordant rows: 213
moderate discordant rows: 1308
fullCP+R moderate-agreeing discordant sequences: 86
```

High-confidence discordance is defined as:

```text
label 1 and score < 0.2
label 0 and score > 0.8
```

Moderate discordance is defined as:

```text
label 1 and score < 0.4
label 0 and score > 0.6
```

## High-confidence discordance counts

| input_type   | discordance_direction            |   n |
|:-------------|:---------------------------------|----:|
| P_domain     | crucivirus_scored_rna_like       |  17 |
| P_domain     | rna_virus_scored_crucivirus_like |  20 |
| R_domain     | crucivirus_scored_rna_like       |  22 |
| R_domain     | rna_virus_scored_crucivirus_like |  37 |
| S_domain     | crucivirus_scored_rna_like       |  14 |
| S_domain     | rna_virus_scored_crucivirus_like |  69 |
| full_capsid  | crucivirus_scored_rna_like       |  21 |
| full_capsid  | rna_virus_scored_crucivirus_like |  13 |

## Moderate discordance counts

| input_type   | discordance_direction            |   n |
|:-------------|:---------------------------------|----:|
| P_domain     | crucivirus_scored_rna_like       | 159 |
| P_domain     | rna_virus_scored_crucivirus_like | 264 |
| R_domain     | crucivirus_scored_rna_like       |  77 |
| R_domain     | rna_virus_scored_crucivirus_like | 131 |
| S_domain     | crucivirus_scored_rna_like       | 124 |
| S_domain     | rna_virus_scored_crucivirus_like | 306 |
| full_capsid  | crucivirus_scored_rna_like       | 114 |
| full_capsid  | rna_virus_scored_crucivirus_like | 133 |

## Top high-confidence discordant sequence-domain rows

| sequence_id            | virus_type   | input_type   | cluster_id   |   classifier_score | discordance_direction            |   discordance_strength |
|:-----------------------|:-------------|:-------------|:-------------|-------------------:|:---------------------------------|-----------------------:|
| Cruci_CruV_500         | crucivirus   | R_domain     | Green        |          0.0238311 | crucivirus_scored_rna_like       |               0.976169 |
| Ribo_JAAOEH010002449.1 | rna_virus    | S_domain     | Yellow       |          0.974006  | rna_virus_scored_crucivirus_like |               0.974006 |
| Ribo_ND_144217         | rna_virus    | S_domain     | Yellow       |          0.968793  | rna_virus_scored_crucivirus_like |               0.968793 |
| Cruci_CruV-692         | crucivirus   | R_domain     | Purple       |          0.0316121 | crucivirus_scored_rna_like       |               0.968388 |
| Cruci_CruV-693         | crucivirus   | R_domain     | Purple       |          0.0316121 | crucivirus_scored_rna_like       |               0.968388 |
| Cruci_CruV-788         | crucivirus   | R_domain     | Purple       |          0.0349708 | crucivirus_scored_rna_like       |               0.965029 |
| Cruci_CruV_489         | crucivirus   | R_domain     | Green        |          0.0384818 | crucivirus_scored_rna_like       |               0.961518 |
| Ribo_ND_132327         | rna_virus    | R_domain     | Blue         |          0.960999  | rna_virus_scored_crucivirus_like |               0.960999 |
| Ribo_ND_336310         | rna_virus    | R_domain     | Red          |          0.953148  | rna_virus_scored_crucivirus_like |               0.953148 |
| Cruci_CruV-930         | crucivirus   | R_domain     | Green        |          0.0511397 | crucivirus_scored_rna_like       |               0.94886  |
| Cruci_CruV-788         | crucivirus   | full_capsid  | Purple       |          0.0532241 | crucivirus_scored_rna_like       |               0.946776 |
| Cruci_CruV-852         | crucivirus   | R_domain     | Green        |          0.0557554 | crucivirus_scored_rna_like       |               0.944245 |
| Cruci_CruV-760         | crucivirus   | R_domain     | Green        |          0.0574622 | crucivirus_scored_rna_like       |               0.942538 |
| Ribo_ND_178256         | rna_virus    | R_domain     | Blue         |          0.940366  | rna_virus_scored_crucivirus_like |               0.940366 |
| Ribo_ND_053581         | rna_virus    | P_domain     | Yellow       |          0.935622  | rna_virus_scored_crucivirus_like |               0.935622 |
| Ribo_ND_092298         | rna_virus    | full_capsid  | Blue         |          0.933542  | rna_virus_scored_crucivirus_like |               0.933542 |
| Ribo_ND_142318         | rna_virus    | R_domain     | Yellow       |          0.930155  | rna_virus_scored_crucivirus_like |               0.930155 |
| Cruci_CruV-856         | crucivirus   | R_domain     | Blue         |          0.0719267 | crucivirus_scored_rna_like       |               0.928073 |
| Ribo_ND_335728         | rna_virus    | R_domain     | Yellow       |          0.926434  | rna_virus_scored_crucivirus_like |               0.926434 |
| Ribo_ND_113307         | rna_virus    | S_domain     | Yellow       |          0.925993  | rna_virus_scored_crucivirus_like |               0.925993 |
| Ribo_ND_366175         | rna_virus    | S_domain     | Yellow       |          0.92289   | rna_virus_scored_crucivirus_like |               0.92289  |
| Ribo_ND_159691         | rna_virus    | S_domain     | Yellow       |          0.92148   | rna_virus_scored_crucivirus_like |               0.92148  |
| Cruci_CruV_532         | crucivirus   | R_domain     | Green        |          0.0787016 | crucivirus_scored_rna_like       |               0.921298 |
| Ribo_ND_030168         | rna_virus    | R_domain     | Red          |          0.917883  | rna_virus_scored_crucivirus_like |               0.917883 |
| Cruci_CruV_265         | crucivirus   | full_capsid  | Purple       |          0.0835023 | crucivirus_scored_rna_like       |               0.916498 |

## Full capsid and R-domain agreeing discordance

These are the highest-priority candidate rows for tree mapping because the full capsid and R-domain models agree that the sequence is discordant.

| sequence_id    | virus_type   | fold_id   |   full_capsid_score |   R_domain_score |   S_domain_score |   P_domain_score |   full_R_mean_score | full_R_both_high_confidence_discordant   |
|:---------------|:-------------|:----------|--------------------:|-----------------:|-----------------:|-----------------:|--------------------:|:-----------------------------------------|
| Cruci_CruV-788 | crucivirus   | Purple    |           0.0532241 |        0.0349708 |         0.709645 |        0.791321  |           0.0440975 | True                                     |
| Cruci_CruV_489 | crucivirus   | Green     |           0.0970193 |        0.0384818 |         0.359658 |        0.832168  |           0.0677506 | True                                     |
| Cruci_CruV_500 | crucivirus   | Green     |           0.14216   |        0.0238311 |         0.305986 |        0.493515  |           0.0829955 | True                                     |
| Ribo_ND_336310 | rna_virus    | Red       |           0.867631  |        0.953148  |         0.171083 |        0.110951  |           0.910389  | True                                     |
| Ribo_ND_093303 | rna_virus    | Blue      |           0.87708   |        0.905694  |         0.448454 |        0.12769   |           0.891387  | True                                     |
| Cruci_CruV-930 | crucivirus   | Green     |           0.171512  |        0.0511397 |         0.249778 |        0.862685  |           0.111326  | True                                     |
| Ribo_ND_178256 | rna_virus    | Blue      |           0.835536  |        0.940366  |         0.477833 |        0.335414  |           0.887951  | True                                     |
| Cruci_CruV_130 | crucivirus   | Purple    |           0.135428  |        0.0931225 |         0.612684 |        0.841494  |           0.114275  | True                                     |
| Cruci_CruV-693 | crucivirus   | Purple    |           0.216027  |        0.0316121 |         0.616125 |        0.938099  |           0.12382   | False                                    |
| Cruci_CruV-692 | crucivirus   | Purple    |           0.216027  |        0.0316121 |         0.616125 |        0.938099  |           0.12382   | False                                    |
| Ribo_ND_183795 | rna_virus    | Blue      |           0.883733  |        0.867684  |         0.736428 |        0.0771574 |           0.875709  | True                                     |
| Ribo_ND_030566 | rna_virus    | Blue      |           0.843776  |        0.89923   |         0.23046  |        0.456535  |           0.871503  | True                                     |
| Ribo_ND_207626 | rna_virus    | Blue      |           0.849361  |        0.890719  |         0.524631 |        0.171614  |           0.87004   | True                                     |
| Cruci_CruV_158 | crucivirus   | Blue      |           0.175059  |        0.0919047 |         0.494876 |        0.851128  |           0.133482  | True                                     |
| Ribo_ND_030168 | rna_virus    | Red       |           0.791532  |        0.917883  |         0.299115 |        0.324361  |           0.854707  | False                                    |
| Cruci_CruV-760 | crucivirus   | Green     |           0.257492  |        0.0574622 |         0.676995 |        0.410065  |           0.157477  | False                                    |
| Cruci_CruV-856 | crucivirus   | Blue      |           0.246714  |        0.0719267 |         0.845744 |        0.278591  |           0.15932   | False                                    |
| Ribo_ND_281099 | rna_virus    | Yellow    |           0.786952  |        0.886461  |         0.582995 |        0.455492  |           0.836706  | False                                    |
| Ribo_ND_142318 | rna_virus    | Yellow    |           0.737793  |        0.930155  |         0.233607 |        0.589459  |           0.833974  | False                                    |
| Cruci_CruV_265 | crucivirus   | Purple    |           0.0835023 |        0.257541  |         0.563314 |        0.870109  |           0.170522  | False                                    |
| Cruci_CruV-968 | crucivirus   | Green     |           0.233991  |        0.114694  |         0.558635 |        0.512424  |           0.174343  | False                                    |
| Ribo_ND_106166 | rna_virus    | Yellow    |           0.824258  |        0.826893  |         0.863007 |        0.774159  |           0.825575  | True                                     |
| Cruci_CruV-847 | crucivirus   | Purple    |           0.203023  |        0.147363  |         0.320425 |        0.365431  |           0.175193  | False                                    |
| Cruci_CruV_375 | crucivirus   | Green     |           0.208942  |        0.149712  |         0.828761 |        0.727075  |           0.179327  | False                                    |
| Cruci_CruV-873 | crucivirus   | Yellow    |           0.189398  |        0.171511  |         0.652444 |        0.372858  |           0.180454  | True                                     |

## Interpretation

The sequence-level tables are now ready for tree annotation. The most biologically useful next step is to map `tree_annotation_scores_wide.tsv` onto the capsid phylogeny and ask whether the fullCP/R-domain agreeing discordant sequences cluster together.

Interpretation should still be cautious:

- isolated discordant sequences may be noise, annotation issues, or low-quality/domain-boundary artifacts;
- clade-structured discordance is more meaningful than single-sequence discordance;
- Rdomain Blue/Green still include the known exact cross-label duplicate-pair caveat from the dataset audit;
- nearest-neighbor identity and simple baselines are still required before claiming the ESM signal is nontrivial.
