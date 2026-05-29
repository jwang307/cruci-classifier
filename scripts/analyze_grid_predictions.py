#!/usr/bin/env python3
"""Create plan-required sequence-level outputs from the ESM grid predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_DIR = REPO_ROOT / "classifier"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))

from classifier.dataprocessing import fasta_to_rows  # noqa: E402
from scripts.prepare_fold_csvs import DOMAIN_PATTERNS, FOLD_NAMES  # noqa: E402


DOMAIN_TO_INPUT = {
    "fullCP": "full_capsid",
    "Rdomain": "R_domain",
    "Sdomain": "S_domain",
    "Pdomain": "P_domain",
}
INPUT_TO_DOMAIN = {value: key for key, value in DOMAIN_TO_INPUT.items()}
DOMAIN_ORDER = ["fullCP", "Rdomain", "Sdomain", "Pdomain"]
FOLD_ORDER = list(FOLD_NAMES)
KNOWN_RDOMAIN_DUPLICATE_IDS = {"Cruci_CruV_493", "Ribo_ND_132327"}


def score_band(score: float) -> str:
    if score < 0.2:
        return "strong_rna_like"
    if score < 0.4:
        return "moderate_rna_like"
    if score < 0.6:
        return "ambiguous"
    if score < 0.8:
        return "moderate_crucivirus_like"
    return "strong_crucivirus_like"


def virus_type(label: int) -> str:
    return "crucivirus" if int(label) == 1 else "rna_virus"


def train_clusters(test_fold: str) -> str:
    return ",".join(fold for fold in FOLD_ORDER if fold != test_fold)


def build_length_table(data_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for domain, (train_pattern, test_pattern) in DOMAIN_PATTERNS.items():
        del train_pattern
        for fold in FOLD_ORDER:
            fasta_path = data_root / test_pattern.format(fold=fold)
            if not fasta_path.exists():
                continue
            for row in fasta_to_rows(fasta_path):
                rows.append(
                    {
                        "domain": domain,
                        "fold": fold,
                        "id": row["id"],
                        "original_fasta_header": row["id"],
                        "sequence_length": int(row["cleaned_length"]),
                        "original_length": int(row["original_length"]),
                        "stops_removed": int(row["stops_removed"]),
                        "invalid_replaced": int(row["invalid_replaced"]),
                    }
                )
    return pd.DataFrame(rows)


def write_long_predictions(predictions: pd.DataFrame, metrics: pd.DataFrame, data_root: Path, out_path: Path) -> pd.DataFrame:
    lengths = build_length_table(data_root)
    predictions = predictions.copy()
    lengths = lengths.copy()
    predictions["record_index_within_domain_id"] = predictions.groupby(["domain", "fold", "id"]).cumcount()
    lengths["record_index_within_domain_id"] = lengths.groupby(["domain", "fold", "id"]).cumcount()
    threshold_cols = metrics[["domain", "fold", "selected_threshold"]].copy()
    df = predictions.merge(threshold_cols, on=["domain", "fold"], how="left")
    df = df.merge(lengths, on=["domain", "fold", "id", "record_index_within_domain_id"], how="left")
    df["input_type"] = df["domain"].map(DOMAIN_TO_INPUT)
    df["sequence_id"] = df["id"]
    df["sequence_record_id"] = (
        df["id"].astype(str)
        + "|"
        + df["domain"].astype(str)
        + "|"
        + df["fold"].astype(str)
        + "|record"
        + df["record_index_within_domain_id"].astype(str)
    )
    df["original_fasta_header"] = df["original_fasta_header"].fillna(df["id"])
    df["true_label"] = df["label"].astype(int)
    df["virus_type"] = df["true_label"].map(virus_type)
    df["cluster_id"] = df["fold"]
    df["fold_id"] = df["fold"]
    df["classifier_score"] = df["score"]
    df["prediction_threshold_0_5"] = df["prediction_0_5"].astype(int)
    df["prediction_validation_threshold"] = df["prediction_selected_threshold"].astype(int)
    df["correct_at_0_5"] = df["correct_0_5"].astype(bool)
    df["correct_at_validation_threshold"] = df["correct_selected_threshold"].astype(bool)
    df["model_type"] = "frozen_esm2_35m_head"
    df["train_clusters"] = df["fold"].map(train_clusters)
    df["test_cluster"] = df["fold"]
    df["score_band"] = df["classifier_score"].map(score_band)
    df["discordant_0_5"] = df["prediction_threshold_0_5"] != df["true_label"]
    df["high_confidence_discordant"] = (
        ((df["true_label"] == 1) & (df["classifier_score"] < 0.2))
        | ((df["true_label"] == 0) & (df["classifier_score"] > 0.8))
    )
    df["moderate_discordant"] = (
        ((df["true_label"] == 1) & (df["classifier_score"] < 0.4))
        | ((df["true_label"] == 0) & (df["classifier_score"] > 0.6))
    )
    df["discordance_direction"] = "concordant_or_ambiguous"
    df.loc[(df["true_label"] == 1) & (df["classifier_score"] < 0.4), "discordance_direction"] = (
        "crucivirus_scored_rna_like"
    )
    df.loc[(df["true_label"] == 0) & (df["classifier_score"] > 0.6), "discordance_direction"] = (
        "rna_virus_scored_crucivirus_like"
    )
    df["discordance_strength"] = (df["classifier_score"] - df["true_label"]).abs()
    df["nearest_train_sequence_id"] = pd.NA
    df["nearest_train_sequence_identity"] = pd.NA
    df["notes"] = ""
    duplicate_mask = (df["domain"] == "Rdomain") & df["id"].isin(KNOWN_RDOMAIN_DUPLICATE_IDS)
    df.loc[duplicate_mask, "notes"] = "known_Rdomain_exact_cross_label_duplicate_pair"

    columns = [
        "sequence_id",
        "sequence_record_id",
        "original_fasta_header",
        "virus_type",
        "true_label",
        "cluster_id",
        "fold_id",
        "input_type",
        "domain",
        "record_index_within_domain_id",
        "sequence_length",
        "classifier_score",
        "score_band",
        "prediction_threshold_0_5",
        "prediction_validation_threshold",
        "selected_threshold",
        "correct_at_0_5",
        "correct_at_validation_threshold",
        "discordant_0_5",
        "moderate_discordant",
        "high_confidence_discordant",
        "discordance_direction",
        "discordance_strength",
        "model_type",
        "train_clusters",
        "test_cluster",
        "nearest_train_sequence_id",
        "nearest_train_sequence_identity",
        "notes",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(out_path, sep="\t", index=False)
    return df


def wide_predictions(long_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    base = long_df[["sequence_id", "virus_type", "true_label", "cluster_id", "fold_id"]].drop_duplicates()
    wide = base.copy()
    for input_type in DOMAIN_TO_INPUT.values():
        domain_raw = long_df[long_df["input_type"] == input_type]
        domain_df = (
            domain_raw.groupby(["sequence_id", "fold_id"], dropna=False)
            .agg(
                classifier_score=("classifier_score", "mean"),
                prediction_threshold_0_5=("prediction_threshold_0_5", "max"),
                prediction_validation_threshold=("prediction_validation_threshold", "max"),
                correct_at_0_5=("correct_at_0_5", "mean"),
                correct_at_validation_threshold=("correct_at_validation_threshold", "mean"),
                sequence_length=("sequence_length", "mean"),
                score_band=("score_band", lambda values: ";".join(sorted(set(map(str, values))))),
                high_confidence_discordant=("high_confidence_discordant", "max"),
                moderate_discordant=("moderate_discordant", "max"),
                records=("sequence_record_id", "count"),
            )
            .reset_index()
        ).rename(
            columns={
                "classifier_score": f"{input_type}_score",
                "prediction_threshold_0_5": f"{input_type}_prediction_0_5",
                "prediction_validation_threshold": f"{input_type}_prediction_validation_threshold",
                "correct_at_0_5": f"{input_type}_correct_0_5",
                "correct_at_validation_threshold": f"{input_type}_correct_validation_threshold",
                "sequence_length": f"{input_type}_length",
                "score_band": f"{input_type}_score_band",
                "high_confidence_discordant": f"{input_type}_high_confidence_discordant",
                "moderate_discordant": f"{input_type}_moderate_discordant",
                "records": f"{input_type}_record_count",
            }
        )
        wide = wide.merge(domain_df, on=["sequence_id", "fold_id"], how="left")

    if {"full_capsid_score", "R_domain_score"}.issubset(wide.columns):
        wide["full_R_score_delta"] = wide["full_capsid_score"] - wide["R_domain_score"]
        wide["full_R_mean_score"] = wide[["full_capsid_score", "R_domain_score"]].mean(axis=1)
        full_high = wide["full_capsid_high_confidence_discordant"].astype("boolean").fillna(False).astype(bool)
        r_high = wide["R_domain_high_confidence_discordant"].astype("boolean").fillna(False).astype(bool)
        full_moderate = wide["full_capsid_moderate_discordant"].astype("boolean").fillna(False).astype(bool)
        r_moderate = wide["R_domain_moderate_discordant"].astype("boolean").fillna(False).astype(bool)
        wide["full_R_both_high_confidence_discordant"] = full_high & r_high
        wide["full_R_both_moderate_discordant"] = full_moderate & r_moderate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, sep="\t", index=False)
    return wide


def write_discordant_tables(long_df: pd.DataFrame, wide_df: pd.DataFrame, out_dir: Path) -> None:
    high = long_df[long_df["high_confidence_discordant"]].copy()
    high = high.sort_values(["discordance_strength", "input_type", "sequence_id"], ascending=[False, True, True])
    high.to_csv(out_dir / "high_confidence_discordant_sequences.tsv", sep="\t", index=False)

    moderate = long_df[long_df["moderate_discordant"]].copy()
    moderate = moderate.sort_values(["discordance_strength", "input_type", "sequence_id"], ascending=[False, True, True])
    moderate.to_csv(out_dir / "moderate_discordant_sequences.tsv", sep="\t", index=False)

    priority = wide_df[
        wide_df.get("full_R_both_moderate_discordant", pd.Series(False, index=wide_df.index)).fillna(False)
    ].copy()
    score_cols = [c for c in wide_df.columns if c.endswith("_score")]
    priority_cols = [
        "sequence_id",
        "virus_type",
        "true_label",
        "cluster_id",
        "fold_id",
        *score_cols,
        "full_R_score_delta",
        "full_R_mean_score",
        "full_R_both_high_confidence_discordant",
        "full_R_both_moderate_discordant",
    ]
    priority = priority[[c for c in priority_cols if c in priority.columns]]
    priority.to_csv(out_dir / "fullCP_Rdomain_agreeing_discordant_sequences.tsv", sep="\t", index=False)


def write_cluster_summary(long_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    summary = (
        long_df.groupby(["input_type", "domain", "cluster_id", "virus_type", "true_label"], dropna=False)
        .agg(
            n=("sequence_id", "count"),
            mean_score=("classifier_score", "mean"),
            median_score=("classifier_score", "median"),
            min_score=("classifier_score", "min"),
            max_score=("classifier_score", "max"),
            accuracy_0_5=("correct_at_0_5", "mean"),
            accuracy_validation_threshold=("correct_at_validation_threshold", "mean"),
            high_confidence_discordant_n=("high_confidence_discordant", "sum"),
            moderate_discordant_n=("moderate_discordant", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(out_path, sep="\t", index=False)
    return summary


def plot_score_histograms(long_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for input_type in DOMAIN_TO_INPUT.values():
        frame = long_df[long_df["input_type"] == input_type]
        plt.figure(figsize=(6, 4))
        for label, name, color in [(0, "RNA virus", "tab:blue"), (1, "Crucivirus", "tab:orange")]:
            scores = frame.loc[frame["true_label"] == label, "classifier_score"]
            if len(scores):
                plt.hist(scores, bins=30, alpha=0.55, density=True, label=name, color=color)
        plt.axvline(0.5, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Out-of-fold score")
        plt.ylabel("Density")
        plt.title(f"{input_type} scores by true label")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{input_type}_score_histogram.png", dpi=200)
        plt.close()


def plot_domain_scatter(wide_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not {"full_capsid_score", "R_domain_score"}.issubset(wide_df.columns):
        return
    frame = wide_df.dropna(subset=["full_capsid_score", "R_domain_score"])
    colors = frame["true_label"].map({0: "tab:blue", 1: "tab:orange"})
    plt.figure(figsize=(5, 5))
    plt.scatter(frame["full_capsid_score"], frame["R_domain_score"], c=colors, s=10, alpha=0.65)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Full capsid score")
    plt.ylabel("R-domain score")
    plt.title("Full capsid vs R-domain out-of-fold scores")
    plt.tight_layout()
    plt.savefig(out_dir / "fullCP_vs_Rdomain_score_scatter.png", dpi=200)
    plt.close()


def plot_discordance_counts(long_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = (
        long_df.groupby(["input_type", "discordance_direction"])
        .size()
        .unstack(fill_value=0)
        .reindex(list(DOMAIN_TO_INPUT.values()))
    )
    columns = [
        "crucivirus_scored_rna_like",
        "rna_virus_scored_crucivirus_like",
        "concordant_or_ambiguous",
    ]
    counts = counts[[c for c in columns if c in counts.columns]]
    ax = counts.drop(columns=["concordant_or_ambiguous"], errors="ignore").plot(kind="bar", figsize=(7, 4), rot=0)
    ax.set_ylabel("Moderate discordant sequences")
    ax.set_title("Discordance counts by input type")
    plt.tight_layout()
    plt.savefig(out_dir / "discordance_counts_by_domain.png", dpi=200)
    plt.close()


def markdown_table(df: pd.DataFrame, columns: Iterable[str], n: int = 20) -> str:
    frame = df[list(columns)].head(n).copy()
    return frame.to_markdown(index=False)


def write_report(
    *,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    del cluster_summary
    high = long_df[long_df["high_confidence_discordant"]].copy()
    moderate = long_df[long_df["moderate_discordant"]].copy()
    high_counts = (
        high.groupby(["input_type", "discordance_direction"]).size().reset_index(name="n")
        if len(high)
        else pd.DataFrame(columns=["input_type", "discordance_direction", "n"])
    )
    moderate_counts = (
        moderate.groupby(["input_type", "discordance_direction"]).size().reset_index(name="n")
        if len(moderate)
        else pd.DataFrame(columns=["input_type", "discordance_direction", "n"])
    )
    wide_priority = wide_df[
        wide_df.get("full_R_both_moderate_discordant", pd.Series(False, index=wide_df.index)).fillna(False)
    ].copy()
    if "full_R_mean_score" in wide_priority:
        wide_priority["full_R_discordance_strength"] = (
            wide_priority["full_R_mean_score"] - wide_priority["true_label"]
        ).abs()
        wide_priority = wide_priority.sort_values("full_R_discordance_strength", ascending=False)

    high_top = high.sort_values("discordance_strength", ascending=False)
    report = f"""## Sequence-level discordance analysis

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
long prediction rows: {len(long_df)}
unique sequence ids: {long_df['sequence_id'].nunique()}
wide tree rows: {len(wide_df)}
high-confidence discordant rows: {len(high)}
moderate discordant rows: {len(moderate)}
fullCP+R moderate-agreeing discordant sequences: {len(wide_priority)}
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

{high_counts.to_markdown(index=False)}

## Moderate discordance counts

{moderate_counts.to_markdown(index=False)}

## Top high-confidence discordant sequence-domain rows

{markdown_table(high_top, ['sequence_id', 'virus_type', 'input_type', 'cluster_id', 'classifier_score', 'discordance_direction', 'discordance_strength'], n=25) if len(high_top) else 'None.'}

## Full capsid and R-domain agreeing discordance

These are the highest-priority candidate rows for tree mapping because the full capsid and R-domain models agree that the sequence is discordant.

{markdown_table(wide_priority, ['sequence_id', 'virus_type', 'fold_id', 'full_capsid_score', 'R_domain_score', 'S_domain_score', 'P_domain_score', 'full_R_mean_score', 'full_R_both_high_confidence_discordant'], n=25) if len(wide_priority) else 'None at the moderate threshold.'}

## Interpretation

The sequence-level tables are now ready for tree annotation. The most biologically useful next step is to map `tree_annotation_scores_wide.tsv` onto the capsid phylogeny and ask whether the fullCP/R-domain agreeing discordant sequences cluster together.

Interpretation should still be cautious:

- isolated discordant sequences may be noise, annotation issues, or low-quality/domain-boundary artifacts;
- clade-structured discordance is more meaningful than single-sequence discordance;
- Rdomain Blue/Green still include the known exact cross-label duplicate-pair caveat from the dataset audit;
- nearest-neighbor identity and simple baselines are still required before claiming the ESM signal is nontrivial.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv", type=Path, default=Path("results/grid_esm35m_e50/all_oof_predictions.csv"))
    parser.add_argument("--metrics_csv", type=Path, default=Path("results/grid_esm35m_e50/aggregate_metrics.csv"))
    parser.add_argument("--data_root", type=Path, default=Path("data/MachineLearning"))
    parser.add_argument("--predictions_out_dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--plots_out_dir", type=Path, default=Path("results/plots/phylogeny_outputs"))
    parser.add_argument("--report_out", type=Path, default=Path("results/reports/sequence_discordance_analysis.md"))
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_csv)
    metrics = pd.read_csv(args.metrics_csv)
    long_df = write_long_predictions(
        predictions,
        metrics,
        args.data_root,
        args.predictions_out_dir / "all_model_long_format_predictions.tsv",
    )
    wide_df = wide_predictions(long_df, args.predictions_out_dir / "tree_annotation_scores_wide.tsv")
    write_discordant_tables(long_df, wide_df, args.predictions_out_dir)
    cluster_summary = write_cluster_summary(long_df, args.predictions_out_dir / "per_cluster_score_summary.tsv")
    plot_score_histograms(long_df, args.plots_out_dir)
    plot_domain_scatter(wide_df, args.plots_out_dir)
    plot_discordance_counts(long_df, args.plots_out_dir)
    write_report(long_df=long_df, wide_df=wide_df, cluster_summary=cluster_summary, out_path=args.report_out)

    print(f"Wrote {args.predictions_out_dir}")
    print(f"Wrote {args.plots_out_dir}")
    print(f"Wrote {args.report_out}")


if __name__ == "__main__":
    main()
