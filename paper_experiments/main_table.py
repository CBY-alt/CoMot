#!/usr/bin/env python3
"""Aggregate the final Top-1000 main table, including external adapters."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("amlworld", "elliptic", "dblp", "web_google")
METHODS = (
    "cn", "aa", "vf2", "turboiso", "gMatch", "MixMatch", "ISONET",
    "TPAB", "NeuGN", "final", "regal", "bright", "HLOT", "deepwalk",
    "node2vec", "graphsage", "seal", "subgnn", "CSGM", "gspan",
    "fanmod", "comot",
)
EXTERNAL = {"gMatch": "gMatch-adapted", "MixMatch": "MixMatch-adapted"}


def metric_row(path: Path) -> tuple[float, float, float]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    precision = float(metrics["motif_level"]["partial_match"])
    recall = float(metrics["motif_level"]["motif_recall"])
    motif_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return motif_f1, float(metrics["edge_level"]["f1"]), float(metrics["node_level"]["f1"])


def metrics_path(output_root: Path, external_root: Path, dataset: str, method: str, seed: int) -> Path:
    if method in EXTERNAL:
        return external_root / dataset / EXTERNAL[method] / f"seed_{seed}" / "metrics.json"
    return output_root / dataset / method / f"seed_{seed}" / "metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--external-root", type=Path, default=ROOT / "outputs/paper/external_adapters")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/paper/main_table.csv")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]

    rows = []
    for method in METHODS:
        row: dict[str, object] = {"method": method}
        for dataset in DATASETS:
            values = [metric_row(metrics_path(args.output_root, args.external_root, dataset, method, seed)) for seed in seeds]
            for index, metric in enumerate(("motif_f1", "edge_f1", "node_f1")):
                samples = [100.0 * value[index] for value in values]
                row[f"{dataset}_{metric}_mean"] = statistics.mean(samples)
                row[f"{dataset}_{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
