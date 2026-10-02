#!/usr/bin/env python3
"""Generate final dataset construction statistics from processed datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ["amlworld", "elliptic", "dblp", "web_google", "bank_private"]
OUTPUT_COLUMNS = [
    "dataset",
    "status",
    "num_nodes",
    "num_edges",
    "directed",
    "weighted",
    "num_partitions",
    "num_queries",
    "num_ground_truth",
    "num_motif_types",
    "motif_types",
    "avg_motif_nodes",
    "avg_motif_edges",
    "avg_motif_partitions",
    "avg_anchor_nodes",
    "avg_hidden_nodes",
    "avg_hidden_edges",
    "cross_partition_motif_ratio",
    "has_native_labels",
    "has_native_motif_gt",
    "construction_rule",
    "ready_for_main_table",
]
PUBLIC_MAIN_DATASETS = {"amlworld", "elliptic", "dblp", "web_google"}


CONSTRUCTION_RULES = {
    "amlworld": "native AMLWorld laundering motifs from processed graph; cross-partition hidden-node/edge recovery task",
    "elliptic": "constructed from processed Elliptic graph using illicit k-hop ego motifs and hash/time hybrid partitions",
    "dblp": "constructed from processed DBLP collaboration graph using protocol-fixed structural motifs and seeded partitions",
    "web_google": "constructed from sampled SNAP Web-Google hyperlink graph using protocol-fixed directed structural motifs and seeded partitions",
    "bank_private": "private schema only / private data required",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_csv_rows(path: Path) -> str:
    if not path.exists():
        return "N/A"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return str(max(0, sum(1 for _ in handle) - 1))


def fmt_bool(value: Any) -> str:
    if value in ("N/A", None, ""):
        return "N/A"
    return "true" if bool(value) else "false"


def fmt_number(value: Any, digits: int = 2) -> str:
    if value in ("N/A", None, ""):
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def average(values: list[int]) -> str:
    return fmt_number(mean(values), digits=2) if values else "N/A"


def count_partitions(data: Any) -> str:
    if isinstance(data, list):
        return str(len(data))
    if isinstance(data, dict):
        if isinstance(data.get("partitions"), list):
            return str(len(data["partitions"]))
        if isinstance(data.get("node_to_partition"), dict):
            return str(len(set(data["node_to_partition"].values())))
        return str(len(data))
    return "N/A"


def infer_native_labels(metadata: dict[str, Any]) -> str:
    source = str(metadata.get("source", "")).lower()
    if "amlworld" in source or "elliptic" in source:
        return "true"
    if "snap" in source:
        return "false"
    return "N/A"


def infer_native_motif_gt(metadata: dict[str, Any]) -> str:
    if metadata.get("motif_construction") or metadata.get("ground_truth_construction"):
        return "false"
    if metadata.get("has_ground_truth") is True and metadata.get("num_motif_instances") not in (None, 0, "0"):
        return "true"
    return "N/A"


def summarize_public_dataset(dataset_dir: Path) -> dict[str, str]:
    dataset = dataset_dir.name
    metadata_path = dataset_dir / "metadata.json"
    query_path = dataset_dir / "query_motifs.json"
    method_query_path = dataset_dir / "method_queries.json"
    gt_path = dataset_dir / "ground_truth.json"
    partitions_path = dataset_dir / "partitions.json"

    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    query_motifs = load_json(query_path) if query_path.exists() else []
    method_queries = load_json(method_query_path) if method_query_path.exists() else []
    ground_truth = load_json(gt_path) if gt_path.exists() else []
    partitions = load_json(partitions_path) if partitions_path.exists() else None

    motif_nodes = [len(item.get("nodes", [])) for item in query_motifs if isinstance(item, dict)]
    motif_edges = [len(item.get("edges", [])) for item in query_motifs if isinstance(item, dict)]
    anchor_nodes = [len(item.get("anchor_nodes", [])) for item in query_motifs if isinstance(item, dict)]
    hidden_nodes = [len(item.get("hidden_nodes", [])) for item in query_motifs if isinstance(item, dict)]
    hidden_edges = [len(item.get("hidden_edges", [])) for item in query_motifs if isinstance(item, dict)]
    motif_types = sorted(
        {
            str(item.get("motif_type"))
            for item in query_motifs
            if isinstance(item, dict) and item.get("motif_type") not in (None, "")
        }
    )
    motif_partitions = [
        len(item.get("true_partitions", []))
        for item in ground_truth
        if isinstance(item, dict) and isinstance(item.get("true_partitions", []), list)
    ]
    cross_partition = [value for value in motif_partitions if value > 1]
    cross_ratio = len(cross_partition) / len(motif_partitions) if motif_partitions else "N/A"

    num_nodes = metadata.get("num_nodes", count_csv_rows(dataset_dir / "nodes.csv"))
    num_edges = metadata.get("num_edges", count_csv_rows(dataset_dir / "edges.csv"))

    return {
        "dataset": dataset,
        "status": "completed",
        "num_nodes": fmt_number(num_nodes),
        "num_edges": fmt_number(num_edges),
        "directed": fmt_bool(metadata.get("directed", "N/A")),
        "weighted": fmt_bool(metadata.get("weighted", "N/A")),
        "num_partitions": count_partitions(partitions),
        "num_queries": fmt_number(len(method_queries) if isinstance(method_queries, list) else "N/A"),
        "num_ground_truth": fmt_number(len(ground_truth) if isinstance(ground_truth, list) else "N/A"),
        "num_motif_types": fmt_number(len(motif_types)),
        "motif_types": ";".join(motif_types) if motif_types else "N/A",
        "avg_motif_nodes": average(motif_nodes),
        "avg_motif_edges": average(motif_edges),
        "avg_motif_partitions": average(motif_partitions),
        "avg_anchor_nodes": average(anchor_nodes),
        "avg_hidden_nodes": average(hidden_nodes),
        "avg_hidden_edges": average(hidden_edges),
        "cross_partition_motif_ratio": fmt_number(cross_ratio, digits=4),
        "has_native_labels": infer_native_labels(metadata),
        "has_native_motif_gt": infer_native_motif_gt(metadata),
        "construction_rule": CONSTRUCTION_RULES[dataset],
        "ready_for_main_table": fmt_bool(dataset in PUBLIC_MAIN_DATASETS),
    }


def summarize_bank_private(dataset_dir: Path) -> dict[str, str]:
    status_path = dataset_dir / "status.json"
    status = "private_data_required"
    if status_path.exists():
        status = load_json(status_path).get("status", status)
    return {
        "dataset": "bank_private",
        "status": status,
        "num_nodes": "N/A",
        "num_edges": "N/A",
        "directed": "N/A",
        "weighted": "N/A",
        "num_partitions": "N/A",
        "num_queries": "N/A",
        "num_ground_truth": "N/A",
        "num_motif_types": "N/A",
        "motif_types": "N/A",
        "avg_motif_nodes": "N/A",
        "avg_motif_edges": "N/A",
        "avg_motif_partitions": "N/A",
        "avg_anchor_nodes": "N/A",
        "avg_hidden_nodes": "N/A",
        "avg_hidden_edges": "N/A",
        "cross_partition_motif_ratio": "N/A",
        "has_native_labels": "N/A",
        "has_native_motif_gt": "N/A",
        "construction_rule": CONSTRUCTION_RULES["bank_private"],
        "ready_for_main_table": "false",
    }


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, str]], path: Path) -> None:
    compact_columns = [
        "dataset",
        "status",
        "num_nodes",
        "num_edges",
        "directed",
        "weighted",
        "num_partitions",
        "num_queries",
        "num_ground_truth",
        "num_motif_types",
        "avg_motif_nodes",
        "avg_motif_edges",
        "avg_motif_partitions",
        "cross_partition_motif_ratio",
        "ready_for_main_table",
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{" + "l" * len(compact_columns) + "}",
        r"\hline",
        " & ".join(latex_escape(col) for col in compact_columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(col, "N/A")) for col in compact_columns) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Dataset construction statistics. Bank-Private is schema-only and excluded from the main comparison until compatible private raw data is available.}",
            r"\label{tab:dataset_construction_stats}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_rows(processed_root: Path) -> list[dict[str, str]]:
    rows = []
    for dataset in DATASETS:
        dataset_dir = processed_root / dataset
        if dataset == "bank_private":
            rows.append(summarize_bank_private(dataset_dir))
        else:
            rows.append(summarize_public_dataset(dataset_dir))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_root", default="data/processed")
    parser.add_argument("--summary_dir", default="outputs/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    summary_dir = Path(args.summary_dir)
    rows = build_rows(processed_root)
    csv_path = summary_dir / "dataset_construction_stats.csv"
    latex_path = summary_dir / "dataset_construction_stats_latex.tex"
    write_csv(rows, csv_path)
    write_latex(rows, latex_path)
    print(json.dumps({"csv": str(csv_path), "latex": str(latex_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
