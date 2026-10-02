import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.assembly_helpers import (  # noqa: E402
    build_anchor_only_candidates,
    build_pair_candidates_from_alignment,
    build_pair_candidates_from_compatibility,
    build_random_cross_partition_candidates,
)
from methods.baselines.common import StandardGraphDataset  # noqa: E402
from methods.comot import (  # noqa: E402
    anchor_local_expansion,
    build_compatibility_graph_standard,
    candidate,
    clean_node_partitions,
    dedupe_candidates,
    prioritized_neighbors,
)

DATASETS = ["amlworld", "elliptic", "dblp", "web_google"]
SEEDS = [0, 1, 2]
METHOD = "comot"
AML_TAG = "five_party"
VARIANTS = [
    "full_comot",
    "no_compatibility_filtering",
    "no_structural_constraints",
    "anchor_only_assembly",
    "random_cross_partition_assembly",
]

BY_SEED_COLUMNS = [
    "dataset",
    "method",
    "runner",
    "seed",
    "status",
    "num_queries",
    "num_ground_truth",
    "num_candidates",
    "candidate_purity",
    "motif_recall_all",
    "motif_type_coverage",
    "avg_candidate_nodes",
    "avg_candidate_edges",
    "avg_candidate_size",
    "runtime",
    "candidate_type_distribution",
    "source",
    "notes",
]

SUMMARY_COLUMNS = [
    "dataset",
    "method",
    "runner",
    "num_seeds",
    "completed_seeds",
    "status",
    "num_candidates_mean",
    "num_candidates_std",
    "candidate_purity_mean",
    "candidate_purity_std",
    "motif_recall_all_mean",
    "motif_recall_all_std",
    "motif_type_coverage_mean",
    "motif_type_coverage_std",
    "avg_candidate_size_mean",
    "avg_candidate_size_std",
    "runtime_mean",
    "runtime_std",
    "dominant_candidate_types",
    "notes",
]

DIST_COLUMNS = [
    "dataset",
    "method",
    "runner",
    "seed",
    "candidate_type",
    "num_candidates",
    "fraction",
]

VARIANT_BY_SEED_COLUMNS = [
    "dataset",
    "method",
    "runner",
    "seed",
    "assembly_variant",
    "status",
    "num_candidates",
    "candidate_purity",
    "motif_recall_all",
    "motif_type_coverage",
    "avg_candidate_size",
    "source",
    "notes",
]

VARIANT_SUMMARY_COLUMNS = [
    "dataset",
    "method",
    "assembly_variant",
    "num_seeds",
    "completed_seeds",
    "status",
    "num_candidates",
    "purity",
    "motif_recall",
    "type_coverage",
    "avg_size",
    "notes",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.6f}"
    return str(value)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def as_str_set(values: Iterable[Any]) -> Set[str]:
    return {str(value) for value in values if str(value) not in {"", "None", "nan"}}


def load_ground_truth(dataset: str) -> Tuple[List[Dict[str, Any]], Dict[str, Set[str]], Dict[str, str]]:
    data_dir = PROJECT_ROOT / "data" / "processed" / dataset
    gt_rows = load_json(data_dir / "ground_truth.json")
    queries = load_json(data_dir / "query_motifs.json")
    query_type = {str(row["query_id"]): str(row.get("motif_type", "unknown")) for row in queries}

    edge_to_motifs: Dict[str, Set[str]] = defaultdict(set)
    motif_to_type: Dict[str, str] = {}
    for row in gt_rows:
        motif_id = str(row.get("motif_instance_id") or row.get("query_id"))
        motif_to_type[motif_id] = query_type.get(str(row.get("query_id")), "unknown")
        for edge_id in row.get("true_edges", []):
            edge_to_motifs[str(edge_id)].add(motif_id)
    return gt_rows, edge_to_motifs, motif_to_type


def load_aml_candidates(run_dir: Path) -> Tuple[List[Dict[str, Any]], Path]:
    candidate_dir = run_dir / f"semotif_candidates_{AML_TAG}"
    candidates: List[Dict[str, Any]] = []
    for name in ["chain_candidates.csv", "fan_candidates.csv", "cycle_candidates.csv"]:
        path = candidate_dir / name
        if not path.exists():
            continue
        for _, row in pd.read_csv(path).iterrows():
            candidates.append({
                "candidate_type": str(row.get("candidate_type", name.replace("_candidates.csv", "").upper())),
                "nodes": as_str_set(parse_json_list(row.get("nodes_json"))),
                "edges": as_str_set(parse_json_list(row.get("tx_ids_json"))),
                "score": row.get("score_mean"),
            })
    if not candidates:
        raise FileNotFoundError(f"No AMLWorld candidate CSVs found in {candidate_dir}")
    return candidates, candidate_dir


def load_standard_candidates(run_dir: Path) -> Tuple[List[Dict[str, Any]], Path]:
    path = run_dir / "comot_standard_candidates.jsonl"
    if not path.exists():
        raise FileNotFoundError(str(path))
    candidates: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            candidates.append({
                "candidate_type": str(item.get("candidate_type", "unknown")),
                "nodes": as_str_set(item.get("node_ids", [])),
                "edges": as_str_set(item.get("edge_ids", [])),
                "score": item.get("score"),
            })
    return candidates, path


def dataframe_to_candidates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        candidates.append({
            "candidate_type": str(row.get("candidate_type", "unknown")),
            "nodes": as_str_set(parse_json_list(row.get("nodes_json"))),
            "edges": as_str_set(parse_json_list(row.get("tx_ids_json"))),
            "score": row.get("score_mean"),
        })
    return candidates


def normalize_standard_items(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for item in items:
        candidates.append({
            "candidate_type": str(item.get("candidate_type", "unknown")),
            "nodes": as_str_set(item.get("node_ids", [])),
            "edges": as_str_set(item.get("edge_ids", [])),
            "score": item.get("score"),
        })
    return candidates


def load_candidates(dataset: str, run_dir: Path) -> Tuple[List[Dict[str, Any]], str, Path]:
    if dataset == "amlworld" and (run_dir / f"semotif_candidates_{AML_TAG}").exists():
        candidates, source = load_aml_candidates(run_dir)
        return candidates, "semotif_amlworld", source
    candidates, source = load_standard_candidates(run_dir)
    return candidates, "comot_standard", source


def load_run_config(run_dir: Path, dataset: str) -> Dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    config.setdefault("_run", {})["dataset"] = dataset
    config.setdefault("data", {})["dataset_dir"] = str(PROJECT_ROOT / "data" / "processed" / dataset)
    return config


def standard_dataset_and_compatibility(dataset: str, run_dir: Path) -> Tuple[StandardGraphDataset, Dict[str, Any], Dict[str, Any]]:
    config = load_run_config(run_dir, dataset)
    ds = StandardGraphDataset(dataset_name=dataset, project_root=PROJECT_ROOT, config=config)
    compatibility = build_compatibility_graph_standard(ds, config)
    return ds, compatibility, config


def evaluate_candidates(
    dataset: str,
    seed: int,
    candidates: Sequence[Dict[str, Any]],
    runner: str,
    source: Path,
    gt_rows: Sequence[Dict[str, Any]],
    edge_to_motifs: Dict[str, Set[str]],
    motif_to_type: Dict[str, str],
    runtime: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    hit_count = 0
    hit_motifs: Set[str] = set()
    hit_types: Set[str] = set()
    type_counts: Counter[str] = Counter()
    node_sizes: List[int] = []
    edge_sizes: List[int] = []

    for item in candidates:
        candidate_type = str(item["candidate_type"])
        type_counts[candidate_type] += 1
        nodes = set(item["nodes"])
        edges = set(item["edges"])
        node_sizes.append(len(nodes))
        edge_sizes.append(len(edges))

        candidate_hits: Set[str] = set()
        for edge_id in edges:
            candidate_hits.update(edge_to_motifs.get(edge_id, set()))
        if candidate_hits:
            hit_count += 1
            hit_motifs.update(candidate_hits)
            for motif_id in candidate_hits:
                if motif_id in motif_to_type:
                    hit_types.add(motif_to_type[motif_id])

    num_candidates = len(candidates)
    motif_types = set(motif_to_type.values())
    row = {
        "dataset": dataset,
        "method": METHOD,
        "runner": runner,
        "seed": seed,
        "status": "completed",
        "num_queries": len(gt_rows),
        "num_ground_truth": len(gt_rows),
        "num_candidates": num_candidates,
        "candidate_purity": hit_count / num_candidates if num_candidates else 0.0,
        "motif_recall_all": len(hit_motifs) / len(gt_rows) if gt_rows else 0.0,
        "motif_type_coverage": len(hit_types) / len(motif_types) if motif_types else 0.0,
        "avg_candidate_nodes": mean(node_sizes) if node_sizes else 0.0,
        "avg_candidate_edges": mean(edge_sizes) if edge_sizes else 0.0,
        "avg_candidate_size": mean((n + e) / 2.0 for n, e in zip(node_sizes, edge_sizes)) if node_sizes else 0.0,
        "runtime": runtime,
        "candidate_type_distribution": json.dumps(dict(sorted(type_counts.items())), sort_keys=True),
        "source": rel(source),
        "notes": "Candidate assembly evaluated by edge overlap against ground_truth.json.",
    }

    dist_rows = []
    for candidate_type, count in sorted(type_counts.items()):
        dist_rows.append({
            "dataset": dataset,
            "method": METHOD,
            "runner": runner,
            "seed": seed,
            "candidate_type": candidate_type,
            "num_candidates": count,
            "fraction": count / num_candidates if num_candidates else 0.0,
        })
    return row, dist_rows


def evaluate_variant_candidates(
    dataset: str,
    seed: int,
    variant: str,
    runner: str,
    candidates: Sequence[Dict[str, Any]],
    source: str,
    notes: str,
    gt_rows: Sequence[Dict[str, Any]],
    edge_to_motifs: Dict[str, Set[str]],
    motif_to_type: Dict[str, str],
) -> Dict[str, Any]:
    row, _ = evaluate_candidates(
        dataset=dataset,
        seed=seed,
        candidates=candidates,
        runner=runner,
        source=PROJECT_ROOT / source if source != "N/A" else PROJECT_ROOT,
        gt_rows=gt_rows,
        edge_to_motifs=edge_to_motifs,
        motif_to_type=motif_to_type,
        runtime=None,
    )
    return {
        "dataset": dataset,
        "method": METHOD,
        "runner": runner,
        "seed": seed,
        "assembly_variant": variant,
        "status": "completed",
        "num_candidates": row["num_candidates"],
        "candidate_purity": row["candidate_purity"],
        "motif_recall_all": row["motif_recall_all"],
        "motif_type_coverage": row["motif_type_coverage"],
        "avg_candidate_size": row["avg_candidate_size"],
        "source": source,
        "notes": notes,
    }


def unavailable_variant_row(dataset: str, seed: int, variant: str, runner: str, reason: str) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "method": METHOD,
        "runner": runner,
        "seed": seed,
        "assembly_variant": variant,
        "status": "not_available",
        "num_candidates": None,
        "candidate_purity": None,
        "motif_recall_all": None,
        "motif_type_coverage": None,
        "avg_candidate_size": None,
        "source": "N/A",
        "notes": reason,
    }


def amlworld_variant_candidates(
    run_dir: Path,
    seed: int,
    variant: str,
    full_candidates: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str, str]:
    if variant == "full_comot":
        return list(full_candidates), rel(run_dir / f"semotif_candidates_{AML_TAG}"), "Existing full CoMot/SeMotif candidate pool."

    orchestrator_dir = run_dir / f"semotif_orchestrator_{AML_TAG}"
    partition_dir = run_dir / f"semotif_partitioned_{AML_TAG}"
    data_dir = PROJECT_ROOT / "data" / "processed" / "amlworld"

    if variant == "no_compatibility_filtering":
        source = orchestrator_dir / "alignment_pairs_scored.csv"
        if not source.exists():
            raise FileNotFoundError(str(source))
        df = build_pair_candidates_from_alignment(source)
        return dataframe_to_candidates(df), rel(source), "Uses saved pre-filter alignment pairs directly as pair candidates."

    if variant == "no_structural_constraints":
        source = orchestrator_dir / "compatibility_graph_edges.csv"
        if not source.exists():
            raise FileNotFoundError(str(source))
        df = build_pair_candidates_from_compatibility(source)
        return dataframe_to_candidates(df), rel(source), "Uses retained compatibility graph edges directly, bypassing chain/fan/cycle constraints."

    if variant == "anchor_only_assembly":
        tx_path = partition_dir / "transactions_partitioned.csv"
        query_path = data_dir / "method_queries.json"
        if not tx_path.exists() or not query_path.exists():
            raise FileNotFoundError(f"{tx_path}; {query_path}")
        df = build_anchor_only_candidates(tx_path, query_path)
        return dataframe_to_candidates(df), f"{rel(tx_path)}; {rel(query_path)}", "Expands only from method-facing anchor nodes."

    if variant == "random_cross_partition_assembly":
        tx_path = partition_dir / "transactions_partitioned.csv"
        if not tx_path.exists():
            raise FileNotFoundError(str(tx_path))
        df = build_random_cross_partition_candidates(tx_path, target_count=len(full_candidates), seed=seed)
        return dataframe_to_candidates(df), rel(tx_path), "Fixed-seed random visible cross-partition edge sample matched to full candidate count."

    raise ValueError(f"Unknown variant: {variant}")


def standard_no_compatibility_candidates(
    queries: Sequence[Dict[str, Any]],
    compatibility: Dict[str, Any],
    max_neighbors: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Public-dataset standard runs do not persist AMLWorld-style pre-filter
    # alignment pairs. This bounded raw-neighborhood pool is the recoverable
    # no-filter counterpart: it expands the anchor incident edge pool before
    # applying the stricter retained-edge cap used by no_structural_constraints.
    raw_cap = max(max_neighbors * 4, max_neighbors)
    for query in queries:
        for anchor in [str(node) for node in query.get("anchor_nodes", [])]:
            if anchor not in compatibility["undirected_adj"]:
                continue
            for neighbor, edge_id in prioritized_neighbors(anchor, compatibility, raw_cap, directed=False):
                rows.append(candidate(query, "raw_anchor_edge", [anchor, neighbor], [edge_id], [anchor]))
    return normalize_standard_items(dedupe_candidates(rows))


def standard_no_structural_candidates(
    queries: Sequence[Dict[str, Any]],
    compatibility: Dict[str, Any],
    max_neighbors: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for query in queries:
        for anchor in [str(node) for node in query.get("anchor_nodes", [])]:
            if anchor not in compatibility["undirected_adj"]:
                continue
            for neighbor, edge_id in prioritized_neighbors(anchor, compatibility, max_neighbors, directed=False):
                rows.append(candidate(query, "compatibility_edge", [anchor, neighbor], [edge_id], [anchor]))
    return normalize_standard_items(dedupe_candidates(rows))


def standard_anchor_only_candidates(
    queries: Sequence[Dict[str, Any]],
    compatibility: Dict[str, Any],
    max_neighbors: int,
    expansion_radius: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for query in queries:
        for anchor in [str(node) for node in query.get("anchor_nodes", [])]:
            if anchor in compatibility["undirected_adj"]:
                rows.extend(anchor_local_expansion(query, anchor, compatibility, max_neighbors, expansion_radius))
    return normalize_standard_items(dedupe_candidates(rows))


def is_cross_partition_edge(src: str, dst: str, edge_id: str, compatibility: Dict[str, Any]) -> bool:
    edge_parts = {part for part in compatibility["edge_to_partitions"].get(edge_id, set()) if part != "UNKNOWN"}
    if len(edge_parts) > 1:
        return True
    src_parts = clean_node_partitions(src, compatibility)
    dst_parts = clean_node_partitions(dst, compatibility)
    return bool(src_parts and dst_parts and src_parts.isdisjoint(dst_parts))


def standard_random_cross_partition_candidates(
    queries: Sequence[Dict[str, Any]],
    compatibility: Dict[str, Any],
    target_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    edge_rows = [
        (edge_id, src, dst)
        for edge_id, (src, dst) in compatibility["edge_lookup"].items()
        if is_cross_partition_edge(src, dst, edge_id, compatibility)
    ]
    if not edge_rows:
        edge_rows = [(edge_id, src, dst) for edge_id, (src, dst) in compatibility["edge_lookup"].items()]
    if not edge_rows or not queries:
        return []

    sample_size = min(target_count, len(edge_rows))
    sample = rng.sample(edge_rows, sample_size)
    rows = []
    for idx, (edge_id, src, dst) in enumerate(sample):
        query = queries[idx % len(queries)]
        rows.append(candidate(query, "random_cross_partition", [src, dst], [edge_id], query.get("anchor_nodes", [])))
    return normalize_standard_items(dedupe_candidates(rows))


def standard_variant_candidates(
    dataset_name: str,
    run_dir: Path,
    seed: int,
    variant: str,
    full_candidates: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str, str]:
    if variant == "full_comot":
        return list(full_candidates), rel(run_dir / "comot_standard_candidates.jsonl"), "Existing full CoMot standard candidate pool."

    dataset, compatibility, config = standard_dataset_and_compatibility(dataset_name, run_dir)
    return standard_variant_candidates_from_context(dataset_name, run_dir, seed, variant, full_candidates, dataset, compatibility, config)


def standard_variant_candidates_from_context(
    dataset_name: str,
    run_dir: Path,
    seed: int,
    variant: str,
    full_candidates: Sequence[Dict[str, Any]],
    dataset: StandardGraphDataset,
    compatibility: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str, str]:
    if variant == "full_comot":
        return list(full_candidates), rel(run_dir / "comot_standard_candidates.jsonl"), "Existing full CoMot standard candidate pool."

    standard_cfg = config.get("comot", {}).get("standard", {})
    max_neighbors = int(standard_cfg.get("max_neighbors_per_anchor", 24))
    expansion_radius = int(standard_cfg.get("expansion_radius", 2))
    queries = dataset.load_query_motifs()

    if variant == "no_compatibility_filtering":
        return (
            standard_no_compatibility_candidates(queries, compatibility, max_neighbors),
            rel(PROJECT_ROOT / "data" / "processed" / dataset_name / "edges.csv"),
            "Standard runner has no persisted pre-filter alignment file; this uses a bounded raw anchor-neighborhood edge pool before the retained-edge cap.",
        )
    if variant == "no_structural_constraints":
        return (
            standard_no_structural_candidates(queries, compatibility, max_neighbors),
            rel(PROJECT_ROOT / "data" / "processed" / dataset_name / "edges.csv"),
            "Uses anchor-incident compatibility edges directly, bypassing motif-shape assembly constraints.",
        )
    if variant == "anchor_only_assembly":
        return (
            standard_anchor_only_candidates(queries, compatibility, max_neighbors, expansion_radius),
            rel(PROJECT_ROOT / "data" / "processed" / dataset_name / "method_queries.json"),
            "Uses only method-facing anchors and local radius expansion.",
        )
    if variant == "random_cross_partition_assembly":
        return (
            standard_random_cross_partition_candidates(queries, compatibility, len(full_candidates), seed),
            rel(PROJECT_ROOT / "data" / "processed" / dataset_name / "partitions.json"),
            "Fixed-seed random cross-partition edge sample matched to full candidate count.",
        )
    raise ValueError(f"Unknown variant: {variant}")


def missing_row(dataset: str, seed: int, reason: str) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "method": METHOD,
        "runner": "unknown",
        "seed": seed,
        "status": "needs_run",
        "num_queries": None,
        "num_ground_truth": None,
        "num_candidates": None,
        "candidate_purity": None,
        "motif_recall_all": None,
        "motif_type_coverage": None,
        "avg_candidate_nodes": None,
        "avg_candidate_edges": None,
        "avg_candidate_size": None,
        "runtime": None,
        "candidate_type_distribution": None,
        "source": "N/A",
        "notes": reason,
    }


def numeric_values(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            values.append(float(value))
    return values


def mean_std(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[Any, Any]:
    values = numeric_values(rows, key)
    if not values:
        return None, None
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(by_seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        rows = [row for row in by_seed_rows if row["dataset"] == dataset and row["status"] == "completed"]
        all_rows = [row for row in by_seed_rows if row["dataset"] == dataset]
        if not rows:
            output.append({
                "dataset": dataset,
                "method": METHOD,
                "runner": "unknown",
                "num_seeds": len(all_rows),
                "completed_seeds": 0,
                "status": "needs_run",
                "notes": "No completed candidate assembly artifacts found.",
            })
            continue
        aggregate: Dict[str, Any] = {
            "dataset": dataset,
            "method": METHOD,
            "runner": ",".join(sorted({str(row["runner"]) for row in rows})),
            "num_seeds": len(all_rows),
            "completed_seeds": len(rows),
            "status": "completed" if len(rows) == len(all_rows) else "needs_run",
        }
        for key in [
            "num_candidates",
            "candidate_purity",
            "motif_recall_all",
            "motif_type_coverage",
            "avg_candidate_size",
            "runtime",
        ]:
            avg, sd = mean_std(rows, key)
            aggregate[f"{key}_mean"] = avg
            aggregate[f"{key}_std"] = sd

        type_counter: Counter[str] = Counter()
        for row in rows:
            for candidate_type, count in json.loads(row["candidate_type_distribution"]).items():
                type_counter[str(candidate_type)] += int(count)
        aggregate["dominant_candidate_types"] = json.dumps(dict(type_counter.most_common(5)), sort_keys=True)
        aggregate["notes"] = "Aggregated over completed CoMot seeds 0/1/2; purity/recall use edge overlap against ground_truth.json."
        output.append(aggregate)
    return output


def aggregate_variant_rows(by_seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            all_rows = [
                row for row in by_seed_rows
                if row["dataset"] == dataset and row["assembly_variant"] == variant
            ]
            rows = [row for row in all_rows if row["status"] == "completed"]
            if not rows:
                output.append({
                    "dataset": dataset,
                    "method": METHOD,
                    "assembly_variant": variant,
                    "num_seeds": len(all_rows),
                    "completed_seeds": 0,
                    "status": "not_available",
                    "num_candidates": None,
                    "purity": None,
                    "motif_recall": None,
                    "type_coverage": None,
                    "avg_size": None,
                    "notes": "; ".join(sorted({str(row.get("notes", "")) for row in all_rows if row.get("notes")})),
                })
                continue
            num_candidates, _ = mean_std(rows, "num_candidates")
            purity, _ = mean_std(rows, "candidate_purity")
            recall, _ = mean_std(rows, "motif_recall_all")
            coverage, _ = mean_std(rows, "motif_type_coverage")
            avg_size, _ = mean_std(rows, "avg_candidate_size")
            output.append({
                "dataset": dataset,
                "method": METHOD,
                "assembly_variant": variant,
                "num_seeds": len(all_rows),
                "completed_seeds": len(rows),
                "status": "completed" if len(rows) == len(all_rows) else "needs_run",
                "num_candidates": num_candidates,
                "purity": purity,
                "motif_recall": recall,
                "type_coverage": coverage,
                "avg_size": avg_size,
                "notes": rows[0].get("notes", ""),
            })
    return output


def write_csv(rows: Sequence[Dict[str, Any]], path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in columns})


def latex_escape(value: Any) -> str:
    text = fmt(value)
    for src, dst in {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}.items():
        text = text.replace(src, dst)
    return text


def write_latex(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    columns = [
        ("dataset", "Dataset"),
        ("num_candidates_mean", "Candidates"),
        ("candidate_purity_mean", "Purity"),
        ("motif_recall_all_mean", "Recall"),
        ("motif_type_coverage_mean", "TypeCov"),
        ("avg_candidate_size_mean", "Avg. size"),
        ("runtime_mean", "Runtime"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        " & ".join(label for _, label in columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(key)) for key, _ in columns) + r" \\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Candidate assembly analysis across four public datasets. Values are averaged over CoMot seeds 0--2.}",
        r"\label{tab:candidate_assembly_4datasets}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_variant_latex(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    columns = [
        ("dataset", "Dataset"),
        ("assembly_variant", "Variant"),
        ("num_candidates", "\\#Cand."),
        ("purity", "Purity"),
        ("motif_recall", "Recall"),
        ("type_coverage", "TypeCov"),
        ("avg_size", "Avg. size"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        " & ".join(label for _, label in columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(key)) for key, _ in columns) + r" \\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Candidate assembly variant analysis across four public datasets. Values are averaged over CoMot seeds 0--2.}",
        r"\label{tab:candidate_assembly_variants_4datasets}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build four-dataset CoMot candidate assembly analysis tables.")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--summary_dir", default="outputs/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = PROJECT_ROOT / args.output_root
    summary_dir = PROJECT_ROOT / args.summary_dir
    by_seed_rows: List[Dict[str, Any]] = []
    dist_rows: List[Dict[str, Any]] = []
    variant_by_seed_rows: List[Dict[str, Any]] = []

    for dataset in DATASETS:
        gt_rows, edge_to_motifs, motif_to_type = load_ground_truth(dataset)
        for seed in SEEDS:
            run_dir = output_root / dataset / METHOD / f"seed_{seed}"
            try:
                candidates, runner, source = load_candidates(dataset, run_dir)
                runtime_path = run_dir / "runtime.json"
                runtime = load_json(runtime_path).get("duration_sec") if runtime_path.exists() else None
                row, distribution = evaluate_candidates(
                    dataset=dataset,
                    seed=seed,
                    candidates=candidates,
                    runner=runner,
                    source=source,
                    gt_rows=gt_rows,
                    edge_to_motifs=edge_to_motifs,
                    motif_to_type=motif_to_type,
                    runtime=runtime,
                )
                by_seed_rows.append(row)
                dist_rows.extend(distribution)
                standard_context = None
                if dataset != "amlworld":
                    standard_context = standard_dataset_and_compatibility(dataset, run_dir)
                for variant in VARIANTS:
                    try:
                        if dataset == "amlworld":
                            variant_candidates, source, notes = amlworld_variant_candidates(
                                run_dir=run_dir,
                                seed=seed,
                                variant=variant,
                                full_candidates=candidates,
                            )
                        else:
                            std_dataset, std_compatibility, std_config = standard_context
                            variant_candidates, source, notes = standard_variant_candidates_from_context(
                                dataset_name=dataset,
                                run_dir=run_dir,
                                seed=seed,
                                variant=variant,
                                full_candidates=candidates,
                                dataset=std_dataset,
                                compatibility=std_compatibility,
                                config=std_config,
                            )
                        variant_by_seed_rows.append(evaluate_variant_candidates(
                            dataset=dataset,
                            seed=seed,
                            variant=variant,
                            runner=runner,
                            candidates=variant_candidates,
                            source=source,
                            notes=notes,
                            gt_rows=gt_rows,
                            edge_to_motifs=edge_to_motifs,
                            motif_to_type=motif_to_type,
                        ))
                    except FileNotFoundError as exc:
                        variant_by_seed_rows.append(unavailable_variant_row(dataset, seed, variant, runner, str(exc)))
            except FileNotFoundError as exc:
                by_seed_rows.append(missing_row(dataset, seed, str(exc)))
                for variant in VARIANTS:
                    variant_by_seed_rows.append(unavailable_variant_row(dataset, seed, variant, "unknown", str(exc)))

    summary_rows = aggregate_rows(by_seed_rows)
    variant_summary_rows = aggregate_variant_rows(variant_by_seed_rows)

    by_seed_path = summary_dir / "candidate_assembly_4datasets_by_seed.csv"
    summary_path = summary_dir / "candidate_assembly_4datasets.csv"
    latex_path = summary_dir / "candidate_assembly_4datasets_latex.tex"
    dist_path = summary_dir / "candidate_assembly_type_distribution_4datasets.csv"
    variant_by_seed_path = summary_dir / "candidate_assembly_variants_4datasets_by_seed.csv"
    variant_summary_path = summary_dir / "candidate_assembly_variants_4datasets.csv"
    variant_latex_path = summary_dir / "candidate_assembly_variants_4datasets_latex.tex"

    write_csv(by_seed_rows, by_seed_path, BY_SEED_COLUMNS)
    write_csv(summary_rows, summary_path, SUMMARY_COLUMNS)
    write_csv(dist_rows, dist_path, DIST_COLUMNS)
    write_latex(summary_rows, latex_path)
    write_csv(variant_by_seed_rows, variant_by_seed_path, VARIANT_BY_SEED_COLUMNS)
    write_csv(variant_summary_rows, variant_summary_path, VARIANT_SUMMARY_COLUMNS)
    write_variant_latex(variant_summary_rows, variant_latex_path)

    # Backward-compatible paper-mapping aliases for the renamed analysis.
    write_csv(variant_summary_rows, summary_dir / "candidate_generation.csv", VARIANT_SUMMARY_COLUMNS)
    write_variant_latex(variant_summary_rows, summary_dir / "candidate_generation_latex.tex")

    print(json.dumps({
        "summary": rel(summary_path),
        "by_seed": rel(by_seed_path),
        "distribution": rel(dist_path),
        "latex": rel(latex_path),
        "variant_summary": rel(variant_summary_path),
        "variant_by_seed": rel(variant_by_seed_path),
        "variant_latex": rel(variant_latex_path),
        "statuses": {row["dataset"]: row["status"] for row in summary_rows},
        "variant_statuses": {
            f"{row['dataset']}:{row['assembly_variant']}": row["status"]
            for row in variant_summary_rows
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
