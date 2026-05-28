#!/usr/bin/env python3
"""Aggregate sequential ESM grid outputs into analysis-ready CSVs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}{key}_", child, out)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix[:-1]] = value


def _domain_fold_from_eval_dir(eval_dir: Path, root: Path) -> tuple[str, str]:
    rel = eval_dir.relative_to(root)
    parts = rel.parts
    if len(parts) >= 3 and parts[-1] == "eval":
        return parts[-3], parts[-2]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "unknown", "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/grid_esm35m"))
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--metrics_name", default="test_metrics.json")
    parser.add_argument("--predictions_name", default="test_eval_results.csv")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (args.out_dir or root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for metrics_path in sorted(root.glob(f"*/*/eval/{args.metrics_name}")):
        eval_dir = metrics_path.parent
        domain, fold = _domain_fold_from_eval_dir(eval_dir, root)
        with metrics_path.open() as handle:
            payload = json.load(handle)
        row: dict[str, Any] = {
            "domain": domain,
            "fold": fold,
            "metrics_path": str(metrics_path),
        }
        _flatten("", payload, row)
        metric_rows.append(row)

        predictions_path = eval_dir / args.predictions_name
        if predictions_path.exists():
            frame = pd.read_csv(predictions_path)
            frame.insert(0, "fold", fold)
            frame.insert(0, "domain", domain)
            frame["source_eval_results"] = str(predictions_path)
            prediction_frames.append(frame)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_out = out_dir / "aggregate_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)

    predictions_out = out_dir / "all_oof_predictions.csv"
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_out, index=False)
        prediction_rows = sum(len(frame) for frame in prediction_frames)
    else:
        pd.DataFrame().to_csv(predictions_out, index=False)
        prediction_rows = 0

    manifest_out = out_dir / "run_manifest.json"
    with manifest_out.open("w") as handle:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "root": str(root),
                "metrics_csv": str(metrics_out),
                "predictions_csv": str(predictions_out),
                "metrics_json_count": len(metric_rows),
                "prediction_rows": prediction_rows,
                "runs": [
                    {
                        "domain": row["domain"],
                        "fold": row["fold"],
                        "metrics_path": row["metrics_path"],
                    }
                    for row in metric_rows
                ],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(f"Wrote {metrics_out} rows={len(metrics_df)}")
    print(f"Wrote {predictions_out} rows={prediction_rows}")
    print(f"Wrote {manifest_out}")


if __name__ == "__main__":
    main()
