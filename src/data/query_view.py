import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


ALLOWED_TOP_LEVEL_FIELDS = {
    "query_id",
    "motif_type",
    "nodes",
    "edges",
    "anchor_nodes",
    "description",
    "visible_nodes",
    "visible_edges",
    "metadata",
}

FORBIDDEN_EXACT_FIELDS = {
    "hidden_nodes",
    "hidden_edges",
    "node_mapping",
    "edge_mapping",
    "true_nodes",
    "true_edges",
    "true_partitions",
    "motif_instance_id",
    "ground_truth",
    "ground_truth_nodes",
    "ground_truth_edges",
    "ground_truth_partitions",
}

FORBIDDEN_FIELD_FRAGMENTS = (
    "hidden",
    "ground_truth",
    "truth",
    "true_",
    "_true",
    "node_mapping",
    "edge_mapping",
    "motif_instance_id",
    "answer",
    "label_target",
)


def build_method_query_view(query_motifs_path: Any, output_path: Any, edge_map_output_path: Optional[Any] = None) -> Path:
    """Build an inference-safe method-facing query view from full query motifs."""
    source = Path(query_motifs_path)
    output = Path(output_path)
    if edge_map_output_path is None:
        edge_map_output = output.parent / "query_edge_id_map.evaluation_only.json"
    else:
        edge_map_output = Path(edge_map_output_path)
    with open(source, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected {source} to contain a JSON list of queries.")
    sanitized = []
    edge_maps = []
    for query in payload:
        sanitized_query, edge_map = sanitize_query_with_edge_map(query)
        sanitized.append(sanitized_query)
        edge_maps.append(edge_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)
        f.write("\n")
    edge_map_payload = {
        "evaluation_only": True,
        "must_not_be_loaded_by_methods": True,
        "source_query_motifs": str(source),
        "method_queries": str(output),
        "description": (
            "Query-local edge ids used in method_queries.json are mapped back to canonical edge ids "
            "only for evaluation/audit utilities. Method runners and method loaders must not read this file."
        ),
        "queries": edge_maps,
    }
    with open(edge_map_output, "w", encoding="utf-8") as f:
        json.dump(edge_map_payload, f, indent=2)
        f.write("\n")
    return output


def sanitize_query(query: Dict[str, Any]) -> Dict[str, Any]:
    """Return only fields that methods may use at inference time."""
    sanitized, _ = sanitize_query_with_edge_map(query)
    return sanitized


def sanitize_query_with_edge_map(query: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return an inference-safe query plus an evaluation-only edge-id map."""
    if not isinstance(query, dict):
        raise ValueError(f"Query must be a JSON object, got {type(query).__name__}.")
    output: Dict[str, Any] = {}
    edge_id_map: Dict[str, str] = {}
    edge_counter = 0
    for key in ALLOWED_TOP_LEVEL_FIELDS:
        if key not in query or is_forbidden_field(key):
            continue
        value = query[key]
        if key == "metadata":
            value = sanitize_metadata(value)
            if value in ({}, [], None):
                continue
        elif key == "description" and isinstance(value, str):
            value = sanitize_text(value)
        elif key in {"edges", "visible_edges"}:
            value, new_edge_map, edge_counter = sanitize_edges(value, edge_counter)
            edge_id_map.update(new_edge_map)
        output[key] = value
    edge_map_payload = {
        "query_id": str(query.get("query_id", output.get("query_id", ""))),
        "evaluation_only": True,
        "edge_id_map": edge_id_map,
    }
    return output, edge_map_payload


def sanitize_edges(value: Any, start_idx: int = 0) -> Tuple[Any, Dict[str, str], int]:
    """Replace canonical edge ids with query-local ids while preserving topology."""
    if not isinstance(value, list):
        return value, {}, start_idx
    edge_id_map: Dict[str, str] = {}
    sanitized_edges: List[Any] = []
    edge_idx = start_idx
    for item in value:
        if not isinstance(item, dict):
            sanitized_edges.append(item)
            continue
        edge: Dict[str, Any] = {}
        for key, field_value in item.items():
            if is_forbidden_field(str(key)):
                continue
            if key == "edge_id":
                local_id = f"qe{edge_idx}"
                edge["edge_id"] = local_id
                if field_value is not None:
                    edge_id_map[local_id] = str(field_value)
                edge_idx += 1
            else:
                edge[key] = field_value
        if "edge_id" not in edge:
            edge["edge_id"] = f"qe{edge_idx}"
            edge_idx += 1
        sanitized_edges.append(edge)
    return sanitized_edges, edge_id_map, edge_idx
    return output


def load_method_queries(dataset_dir: Any) -> List[Dict[str, Any]]:
    """Load method-facing queries, generating them from full queries if absent."""
    dataset_path = Path(dataset_dir)
    method_queries_path = dataset_path / "method_queries.json"
    full_queries_path = dataset_path / "query_motifs.json"
    if not method_queries_path.exists():
        print(
            f"[query_view] WARNING: {method_queries_path} does not exist; "
            f"generating sanitized method-facing queries from {full_queries_path}.",
            file=sys.stderr,
            flush=True,
        )
        build_method_query_view(full_queries_path, method_queries_path)
    with open(method_queries_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected {method_queries_path} to contain a JSON list of queries.")
    return payload


def sanitize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if is_forbidden_field(str(key)):
                continue
            cleaned = sanitize_metadata(item)
            if cleaned is not None:
                output[key] = cleaned
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            cleaned = sanitize_metadata(item)
            if cleaned is not None:
                output.append(cleaned)
        return output
    return value


def sanitize_text(value: str) -> str:
    replacements = {
        "hidden_nodes": "withheld nodes",
        "hidden_edges": "withheld edges",
        "node_mapping": "query mapping",
        "edge_mapping": "query mapping",
        "ground truth": "evaluation reference",
        "ground_truth": "evaluation reference",
        "true_nodes": "evaluation nodes",
        "true_edges": "evaluation edges",
        "true_partitions": "evaluation partitions",
        "motif_instance_id": "query id",
    }
    output = value
    for src, dst in replacements.items():
        output = output.replace(src, dst)
        output = output.replace(src.upper(), dst)
    return output


def is_forbidden_field(key: str) -> bool:
    normalized = key.lower()
    return normalized in FORBIDDEN_EXACT_FIELDS or any(fragment in normalized for fragment in FORBIDDEN_FIELD_FRAGMENTS)


def resolve_dataset_dir(dataset: str) -> Path:
    path = Path(dataset)
    if path.exists() or path.is_absolute() or "/" in dataset:
        return path.expanduser()
    return PROJECT_ROOT / "data" / "processed" / dataset


def build_for_dataset(dataset: str) -> Path:
    dataset_dir = resolve_dataset_dir(dataset)
    return build_method_query_view(dataset_dir / "query_motifs.json", dataset_dir / "method_queries.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build method-facing no-leakage query views.")
    parser.add_argument("datasets", nargs="+", help="Dataset names, processed dirs, or all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets: Iterable[str]
    if args.datasets == ["all"]:
        processed_root = PROJECT_ROOT / "data" / "processed"
        datasets = sorted(path.name for path in processed_root.iterdir() if path.is_dir())
    else:
        datasets = args.datasets
    for dataset in datasets:
        dataset_dir = resolve_dataset_dir(dataset)
        query_path = dataset_dir / "query_motifs.json"
        if not query_path.exists():
            print(f"[query_view] skip {dataset}: missing {query_path}", flush=True)
            continue
        output = build_for_dataset(str(dataset_dir))
        print(f"[query_view] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
