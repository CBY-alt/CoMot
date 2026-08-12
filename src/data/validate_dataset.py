import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.standard_format import (
    EDGES_COLUMNS,
    LABELS_COLUMNS,
    NODES_COLUMNS,
    STANDARD_FILES,
)


JSON_LIST_FILES = {
    "partitions.json": ["partition_id", "visible_nodes", "visible_edges"],
    "query_motifs.json": [
        "query_id",
        "motif_type",
        "nodes",
        "edges",
        "anchor_nodes",
        "hidden_nodes",
        "hidden_edges",
        "description",
    ],
    "ground_truth.json": ["query_id", "true_nodes", "true_edges", "true_partitions", "motif_instance_id"],
}

METADATA_FIELDS = [
    "dataset_name",
    "graph_type",
    "directed",
    "weighted",
    "has_timestamps",
    "has_node_features",
    "has_edge_features",
    "has_ground_truth",
    "source",
    "citation",
]


class ValidationReport:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, int] = {}

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def require_columns(df: pd.DataFrame, columns: List[str], path: Path, report: ValidationReport) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        report.error(f"{path.name} missing columns: {missing}")


def load_json(path: Path, report: ValidationReport):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        report.error(f"{path.name} is not valid JSON: {exc}")
        return None


def validate_json_list_file(path: Path, required_fields: List[str], report: ValidationReport) -> None:
    payload = load_json(path, report)
    if payload is None:
        return
    if not isinstance(payload, list):
        report.error(f"{path.name} must be a JSON list")
        return
    for idx, item in enumerate(payload[:100]):
        if not isinstance(item, dict):
            report.error(f"{path.name}[{idx}] must be an object")
            continue
        missing = [field for field in required_fields if field not in item]
        if missing:
            report.error(f"{path.name}[{idx}] missing fields: {missing}")
    report.stats[path.name.replace(".json", "_count")] = len(payload)


def validate_metadata(path: Path, report: ValidationReport) -> None:
    payload = load_json(path, report)
    if payload is None:
        return
    if not isinstance(payload, dict):
        report.error("metadata.json must be a JSON object")
        return
    missing = [field for field in METADATA_FIELDS if field not in payload]
    if missing:
        report.error(f"metadata.json missing fields: {missing}")


def validate_dataset(dataset_dir: Path, sample_edges: int = 500_000) -> ValidationReport:
    dataset_dir = Path(dataset_dir)
    report = ValidationReport()

    if not dataset_dir.exists():
        report.error(f"Dataset directory does not exist: {dataset_dir}")
        return report

    for filename in STANDARD_FILES:
        if not (dataset_dir / filename).exists():
            report.error(f"Missing required file: {filename}")

    if report.errors:
        return report

    nodes_path = dataset_dir / "nodes.csv"
    edges_path = dataset_dir / "edges.csv"
    labels_path = dataset_dir / "labels.csv"

    nodes_df = pd.read_csv(nodes_path, dtype=str)
    require_columns(nodes_df, NODES_COLUMNS, nodes_path, report)
    if "node_id" in nodes_df.columns:
        duplicate_nodes = int(nodes_df["node_id"].duplicated().sum())
        if duplicate_nodes:
            report.error(f"nodes.csv contains duplicate node_id values: {duplicate_nodes}")
        report.stats["num_nodes"] = int(len(nodes_df))

    edge_head = pd.read_csv(edges_path, dtype=str, nrows=sample_edges)
    require_columns(edge_head, EDGES_COLUMNS, edges_path, report)
    if {"edge_id", "src", "dst"}.issubset(edge_head.columns):
        duplicate_edges = int(edge_head["edge_id"].duplicated().sum())
        if duplicate_edges:
            report.error(f"edges.csv sample contains duplicate edge_id values: {duplicate_edges}")
        node_ids = set(nodes_df["node_id"].astype(str).tolist()) if "node_id" in nodes_df.columns else set()
        missing_src = int((~edge_head["src"].astype(str).isin(node_ids)).sum())
        missing_dst = int((~edge_head["dst"].astype(str).isin(node_ids)).sum())
        if missing_src or missing_dst:
            report.error(
                f"edges.csv sample references unknown nodes: missing_src={missing_src}, missing_dst={missing_dst}"
            )
        report.stats["sampled_edges"] = int(len(edge_head))

    labels_head = pd.read_csv(labels_path, dtype=str, nrows=sample_edges)
    require_columns(labels_head, LABELS_COLUMNS, labels_path, report)
    if "object_type" in labels_head.columns:
        allowed = {"node", "edge", "motif"}
        bad_types = sorted(set(labels_head["object_type"].dropna().astype(str)) - allowed)
        if bad_types:
            report.error(f"labels.csv contains unsupported object_type values in sample: {bad_types}")

    for json_name, fields in JSON_LIST_FILES.items():
        validate_json_list_file(dataset_dir / json_name, fields, report)
    validate_metadata(dataset_dir / "metadata.json", report)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a CoMot standard processed dataset directory.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--sample_edges", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_dataset(Path(args.dataset_dir), sample_edges=args.sample_edges)
    payload = {
        "ok": report.ok,
        "errors": report.errors,
        "warnings": report.warnings,
        "stats": report.stats,
    }
    print(json.dumps(payload, indent=2))
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
