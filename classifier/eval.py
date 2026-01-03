#!/usr/bin/env python3
"""Evaluate a trained ESM-2 classifier on a labelled sequence dataset."""
import argparse
from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
)
from torch.utils.data import DataLoader

from train import ESMClassifier, SeqDataset, eval_epoch


def main(args: argparse.Namespace) -> None:
    """Load a checkpoint, run evaluation, and export detailed outputs."""
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMClassifier().to(device)
    state: Dict[str, torch.Tensor] = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    test_name = Path(args.csv).stem

    dataset = SeqDataset(args.csv, model.alphabet)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    metrics, details = eval_epoch(model, loader, device, return_details=True)

    labels = details["labels"].astype(int)
    scores = details["probs"]
    preds = details["preds"]
    df = pd.DataFrame({
        "sequence": dataset.seqs,
        "label": labels,
        "score": scores,
        "prediction": preds,
        "correct": preds == labels,
    })
    df.to_csv(f"{test_name}_eval_results.csv", index=False)

    display = ConfusionMatrixDisplay.from_predictions(labels, preds, cmap="Blues", colorbar=False)
    display.ax_.set_title(f"Confusion Matrix for {test_name}")
    display.figure_.savefig(f"{test_name}_confusion_matrix.png", bbox_inches="tight")
    plt.close(display.figure_)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
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
    pr_df.to_csv(f"{test_name}_precision_recall_curve.csv", index=False)

    plt.figure()
    plt.plot(recall, precision, marker=".", linewidth=1.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve for {test_name}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(f"{test_name}_precision_recall_curve.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(
        scores[labels == 1],
        bins=args.hist_bins,
        alpha=0.6,
        density=True,
        label="Positive",
        color="tab:orange",
    )
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
    plt.savefig(f"{test_name}_score_histogram.png", bbox_inches="tight")
    plt.close()

    print("EVALUATION METRICS")
    for key, value in metrics.items():
        print(f"{key:10s}: {value:.4f}")
    print(f"avg_precision: {average_precision_score(labels, scores):.4f}")
    print(f"best_threshold (max F1): {best_threshold:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="/large_storage/hielab/jwang/cruci/best.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hist_bins", type=int, default=30)
    main(parser.parse_args())
