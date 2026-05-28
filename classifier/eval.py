#!/usr/bin/env python3
"""Evaluate a trained ESM-2 classifier on a labelled sequence dataset."""
import argparse
import json
import math
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
)
from torch.utils.data import DataLoader

from train import ESMClassifier, SeqDataset, eval_epoch, load_checkpoint_into_model
from train import _format_metric, resolve_device
try:
    from metrics import binary_metrics
except ImportError:
    from classifier.metrics import binary_metrics


def main(args: argparse.Namespace) -> None:
    """Load a checkpoint, run evaluation, and export detailed outputs."""
    device: Union[str, torch.device] = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ESMClassifier().to(device)
    checkpoint_metadata = load_checkpoint_into_model(model, args.checkpoint, map_location=device)
    test_name = Path(args.csv).stem

    dataset = SeqDataset(
        args.csv,
        model.alphabet,
        clean_sequences=not args.no_clean_sequences,
        stop_action=args.stop_action,
        invalid_action=args.invalid_action,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    metrics, details = eval_epoch(model, loader, device, threshold=args.threshold, return_details=True)

    labels = details["labels"].astype(int)
    scores = details["probs"]
    preds = details["preds"]
    preds_0_5 = (scores >= 0.5).astype(int)
    selected_threshold = args.selected_threshold
    checkpoint_training_metadata = checkpoint_metadata.get("metadata", {})
    if selected_threshold is None and isinstance(checkpoint_training_metadata, dict):
        selected_threshold = checkpoint_training_metadata.get("selected_val_threshold")
    preds_selected = (scores >= selected_threshold).astype(int) if selected_threshold is not None else None
    df = pd.DataFrame({
        "id": dataset.ids,
        "label": labels,
        "score": scores,
        "prediction": preds,
        "correct": preds == labels,
        "prediction_0_5": preds_0_5,
        "correct_0_5": preds_0_5 == labels,
    })
    if args.include_sequences:
        df.insert(1, "sequence", dataset.seqs)
    if preds_selected is not None:
        df["prediction_selected_threshold"] = preds_selected
        df["correct_selected_threshold"] = preds_selected == labels
    df.to_csv(out_dir / f"{test_name}_eval_results.csv", index=False)

    display = ConfusionMatrixDisplay.from_predictions(labels, preds, cmap="Blues", colorbar=False)
    display.ax_.set_title(f"Confusion Matrix for {test_name}")
    display.figure_.savefig(out_dir / f"{test_name}_confusion_matrix.png", bbox_inches="tight")
    plt.close(display.figure_)

    if np.unique(labels).size < 2 or labels.sum() == 0:
        best_threshold = 0.5
        pr_df = pd.DataFrame(columns=["threshold", "precision", "recall", "f1"])
        plt.figure()
        plt.text(0.5, 0.5, "PR curve undefined for single-class labels", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_dir / f"{test_name}_precision_recall_curve.png", bbox_inches="tight")
        plt.close()
    else:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        if thresholds.size > 0:
            best_idx = int(np.nanargmax(f1_scores[1:]) + 1)
            best_threshold = float(thresholds[best_idx - 1])
            pr_df = pd.DataFrame({
                "threshold": thresholds,
                "precision": precision[1:],
                "recall": recall[1:],
                "f1": f1_scores[1:],
            })
        else:
            best_threshold = 0.5
            pr_df = pd.DataFrame(columns=["threshold", "precision", "recall", "f1"])
        plt.figure()
        plt.plot(recall, precision, marker=".", linewidth=1.0)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve for {test_name}")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.savefig(out_dir / f"{test_name}_precision_recall_curve.png", bbox_inches="tight")
        plt.close()
    pr_df.to_csv(out_dir / f"{test_name}_precision_recall_curve.csv", index=False)

    plt.figure()
    if np.any(labels == 1):
        plt.hist(
            scores[labels == 1],
            bins=args.hist_bins,
            alpha=0.6,
            density=True,
            label="Positive",
            color="tab:orange",
        )
    if np.any(labels == 0):
        plt.hist(
            scores[labels == 0],
            bins=args.hist_bins,
            alpha=0.6,
            density=True,
            label="Negative",
            color="tab:blue",
        )
    plt.axvline(best_threshold, color="black", linestyle="--", linewidth=1.2, label="Best F1 threshold")
    plt.xlabel("Classifier score")
    plt.ylabel("Density")
    plt.title(f"Score Distribution by Class for {test_name}")
    plt.legend()
    plt.savefig(out_dir / f"{test_name}_score_histogram.png", bbox_inches="tight")
    plt.close()

    print("EVALUATION METRICS")
    for key, value in metrics.items():
        print(f"{key:20s}: {_format_metric(value)}")
    avg_precision = metrics.get("average_precision", math.nan)
    print(f"avg_precision      : {_format_metric(avg_precision)}")
    print(f"best_threshold (max F1): {best_threshold:.4f}")
    metrics_payload = {
        "csv": str(args.csv),
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "threshold": args.threshold,
        "metrics": metrics,
        "metrics_0_5": binary_metrics(labels, scores, thr=0.5),
        "best_threshold_from_eval_set": best_threshold,
    }
    if selected_threshold is not None:
        metrics_payload["selected_threshold"] = selected_threshold
        metrics_payload["metrics_selected_threshold"] = binary_metrics(labels, scores, thr=selected_threshold)
    with (out_dir / f"{test_name}_metrics.json").open("w") as handle:
        json.dump(metrics_payload, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="/large_storage/hielab/jwang/cruci/best.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hist_bins", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--selected_threshold", type=float, default=None)
    parser.add_argument("--stop_action", choices=["remove", "replace_x", "error"], default="remove")
    parser.add_argument("--invalid_action", choices=["replace_x", "remove", "error"], default="replace_x")
    parser.add_argument("--no_clean_sequences", action="store_true")
    parser.add_argument("--include_sequences", action="store_true")
    main(parser.parse_args())
