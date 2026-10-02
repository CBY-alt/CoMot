#!/usr/bin/env python3
"""Aggregate Edge Precision/Recall for the boundary robustness campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "boundary_robustness"
DATASETS = ["amlworld", "elliptic", "dblp", "web_google"]
SETTINGS = [
    "clean",
    "missing10", "missing30", "missing50",
    "spurious10", "spurious30", "spurious50",
    "endpoint10", "endpoint30", "endpoint50",
]
LABELS = {
    "clean": "Clean",
    "missing10": "Missing 10%",
    "missing30": "Missing 30%",
    "missing50": "Missing 50%",
    "spurious10": "Spurious 10%",
    "spurious30": "Spurious 30%",
    "spurious50": "Spurious 50%",
    "endpoint10": "Endpoint 10%",
    "endpoint30": "Endpoint 30%",
    "endpoint50": "Endpoint 50%",
}


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for setting in SETTINGS:
        for dataset in DATASETS:
            for seed in [0, 1, 2]:
                path = root / dataset / setting / f"seed_{seed}" / "metrics.json"
                metrics = json.loads(path.read_text(encoding="utf-8"))
                edge = metrics["edge_level"]
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "setting": setting,
                        "edge_precision": float(edge["precision"]),
                        "edge_recall": float(edge["recall"]),
                        "edge_f1": float(edge["f1"]),
                    }
                )
    return pd.DataFrame(rows)


def aggregate(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting in SETTINGS:
        for dataset in DATASETS:
            group = by_seed[(by_seed.setting == setting) & (by_seed.dataset == dataset)]
            row = {"setting": setting, "dataset": dataset, "num_seeds": group.seed.nunique()}
            for metric in ["edge_precision", "edge_recall", "edge_f1"]:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def formatted_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for setting in SETTINGS:
        row = {"Setting": LABELS[setting]}
        for dataset in DATASETS:
            cell = summary[(summary.setting == setting) & (summary.dataset == dataset)].iloc[0]
            row[dataset] = f"{100 * cell[f'{metric}_mean']:.2f} +/- {100 * cell[f'{metric}_std']:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    root = args.input_root
    by_seed = collect(root)
    summary = aggregate(by_seed)
    by_seed.to_csv(root / "edge_precision_recall_by_seed.csv", index=False, float_format="%.9f")
    summary.to_csv(root / "edge_precision_recall_summary.csv", index=False, float_format="%.9f")

    precision = formatted_table(summary, "edge_precision")
    recall = formatted_table(summary, "edge_recall")
    precision.to_csv(root / "edge_precision_table.csv", index=False)
    recall.to_csv(root / "edge_recall_table.csv", index=False)
    (root / "edge_precision_table.md").write_text(precision.to_markdown(index=False) + "\n", encoding="utf-8")
    (root / "edge_recall_table.md").write_text(recall.to_markdown(index=False) + "\n", encoding="utf-8")

    combined = summary.copy()
    combined["edge_precision_percent"] = combined.apply(
        lambda row: f"{100 * row.edge_precision_mean:.2f} +/- {100 * row.edge_precision_std:.2f}", axis=1
    )
    combined["edge_recall_percent"] = combined.apply(
        lambda row: f"{100 * row.edge_recall_mean:.2f} +/- {100 * row.edge_recall_std:.2f}", axis=1
    )
    combined["edge_f1_percent"] = combined.apply(
        lambda row: f"{100 * row.edge_f1_mean:.2f} +/- {100 * row.edge_f1_std:.2f}", axis=1
    )
    combined = combined[["setting", "dataset", "edge_precision_percent", "edge_recall_percent", "edge_f1_percent"]]
    combined.to_csv(root / "edge_precision_recall_f1_combined.csv", index=False)
    (root / "edge_precision_recall_f1_combined.md").write_text(combined.to_markdown(index=False) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(by_seed), "summary_rows": len(summary), "output_root": str(root)}, indent=2))


if __name__ == "__main__":
    main()
