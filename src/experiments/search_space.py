import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["amlworld", "elliptic", "dblp", "web_google"]
METHOD = "comot"
SEED = 0
AML_TAG = "five_party"
SCALE_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]

SEARCH_COLUMNS = [
    "dataset",
    "raw_edges",
    "boundary_vertices",
    "evidence_records",
    "compatibility_edges",
    "assembled_candidates",
    "topk_candidates",
]
SCALE_COLUMNS = [
    "dataset",
    "scale_ratio",
    "raw_edges",
    "evidence_records",
    "compatibility_edges",
    "assembled_candidates",
    "runtime",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


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


def write_csv(rows: Sequence[Dict[str, Any]], path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in columns})


def parse_tx_ids(value: Any) -> List[int]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            return []
    if not isinstance(parsed, list):
        return []
    output = []
    for item in parsed:
        try:
            output.append(int(item))
        except (TypeError, ValueError):
            continue
    return output


def count_candidate_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return count_csv_rows(path)


def aml_run_dir(output_root: Path, seed: int) -> Path:
    return output_root / "amlworld" / METHOD / f"seed_{seed}"


def aml_boundary_vertices(evidence_dir: Path) -> int:
    vertices: Set[str] = set()
    for path in sorted(evidence_dir.glob("*_evidences.csv")):
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node = row.get("evidence_node_id") or row.get("local_account_id")
                if node:
                    vertices.add(str(node))
    return len(vertices)


def aml_assembled_candidates(candidate_dir: Path) -> int:
    summary_path = candidate_dir / "candidate_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        total = sum(int(summary.get(key, 0)) for key in ["num_chain_candidates", "num_fan_candidates", "num_cycle_candidates"])
        if total:
            return total
    return sum(count_candidate_rows(candidate_dir / name) for name in ["chain_candidates.csv", "fan_candidates.csv", "cycle_candidates.csv"])


def aml_search_row(output_root: Path, seed: int) -> Dict[str, Any]:
    run_dir = aml_run_dir(output_root, seed)
    global_summary = load_json(run_dir / "semotif_global" / "global_summary.json")
    orchestrator_stats = load_json(run_dir / f"semotif_orchestrator_{AML_TAG}" / "orchestrator_stats.json")
    candidate_dir = run_dir / f"semotif_candidates_{AML_TAG}"
    assembled = aml_assembled_candidates(candidate_dir)
    topk = min(1000, count_csv_rows(run_dir / f"semotif_candidates_{AML_TAG}_reranked" / "candidates_reranked.csv"))
    return {
        "dataset": "amlworld",
        "raw_edges": int(global_summary.get("num_transactions") or count_csv_rows(PROJECT_ROOT / "data" / "processed" / "amlworld" / "edges.csv")),
        "boundary_vertices": aml_boundary_vertices(run_dir / f"semotif_local_evidences_{AML_TAG}"),
        "evidence_records": int(orchestrator_stats.get("total_rows_read", 0)),
        "compatibility_edges": int(orchestrator_stats.get("kept_edges", 0)),
        "assembled_candidates": assembled,
        "topk_candidates": topk,
    }


def public_partition_stats(dataset: str) -> Tuple[int, int, Set[str]]:
    processed_dir = PROJECT_ROOT / "data" / "processed" / dataset
    partitions = load_json(processed_dir / "partitions.json")
    edge_parts: Dict[str, Set[str]] = defaultdict(set)
    evidence_records = 0
    for part in partitions:
        partition_id = str(part.get("partition_id", ""))
        for edge_id in part.get("visible_edges", []):
            edge_parts[str(edge_id)].add(partition_id)
            evidence_records += 1
    compatibility_edges = {edge_id for edge_id, parts in edge_parts.items() if len(parts) > 1}
    return evidence_records, len(compatibility_edges), compatibility_edges


def public_boundary_vertices(dataset: str, compatibility_edges: Set[str]) -> int:
    if not compatibility_edges:
        return 0
    edges_path = PROJECT_ROOT / "data" / "processed" / dataset / "edges.csv"
    vertices: Set[str] = set()
    for chunk in pd.read_csv(edges_path, dtype=str, usecols=["edge_id", "src", "dst"], chunksize=200000):
        rows = chunk[chunk["edge_id"].astype(str).isin(compatibility_edges)]
        if rows.empty:
            continue
        vertices.update(rows["src"].astype(str).tolist())
        vertices.update(rows["dst"].astype(str).tolist())
    return len(vertices)


def public_search_row(dataset: str, output_root: Path, seed: int) -> Dict[str, Any]:
    run_dir = output_root / dataset / METHOD / f"seed_{seed}"
    summary = load_json(run_dir / "comot_standard_summary.json")
    metrics = load_json(run_dir / "metrics.json")
    evidence_records, compatibility_edge_count, compatibility_edges = public_partition_stats(dataset)
    return {
        "dataset": dataset,
        "raw_edges": count_csv_rows(PROJECT_ROOT / "data" / "processed" / dataset / "edges.csv"),
        "boundary_vertices": public_boundary_vertices(dataset, compatibility_edges),
        "evidence_records": evidence_records,
        "compatibility_edges": compatibility_edge_count,
        "assembled_candidates": int(summary.get("num_candidates", count_csv_rows(run_dir / "comot_standard_candidates.jsonl"))),
        "topk_candidates": int(metrics.get("efficiency", {}).get("num_candidates", 0)),
    }


def threshold_for_ratio(full_raw_edges: int, ratio: float) -> int:
    return max(0, int(full_raw_edges * ratio))


def count_aml_evidence_records_at_scale(evidence_dir: Path, max_tx_id: int) -> int:
    total = 0
    for path in sorted(evidence_dir.glob("*_evidences.csv")):
        for chunk in pd.read_csv(path, usecols=["tx_id"], chunksize=200000):
            total += int((pd.to_numeric(chunk["tx_id"], errors="coerce") < max_tx_id).sum())
    return total


def count_aml_compat_edges_at_scale(edge_path: Path, max_tx_id: int) -> int:
    total = 0
    for chunk in pd.read_csv(edge_path, usecols=["tx_id_sender", "tx_id_receiver"], chunksize=200000):
        sender = pd.to_numeric(chunk["tx_id_sender"], errors="coerce")
        receiver = pd.to_numeric(chunk["tx_id_receiver"], errors="coerce")
        total += int(((sender < max_tx_id) & (receiver < max_tx_id)).sum())
    return total


def count_candidate_file_at_scale(path: Path, max_tx_id: int) -> int:
    if not path.exists():
        return 0
    total = 0
    for chunk in pd.read_csv(path, usecols=["tx_ids_json"], chunksize=50000):
        for value in chunk["tx_ids_json"]:
            tx_ids = parse_tx_ids(value)
            if tx_ids and all(tx_id < max_tx_id for tx_id in tx_ids):
                total += 1
    return total


def count_aml_candidates_at_scale(candidate_dir: Path, max_tx_id: int) -> int:
    return sum(
        count_candidate_file_at_scale(candidate_dir / name, max_tx_id)
        for name in ["chain_candidates.csv", "fan_candidates.csv", "cycle_candidates.csv"]
    )


def aml_scale_rows(output_root: Path, seed: int, ratios: Sequence[float]) -> List[Dict[str, Any]]:
    run_dir = aml_run_dir(output_root, seed)
    full_raw_edges = int(load_json(run_dir / "semotif_global" / "global_summary.json").get("num_transactions"))
    full_runtime = float(load_json(run_dir / "runtime.json").get("duration_sec"))
    evidence_dir = run_dir / f"semotif_local_evidences_{AML_TAG}"
    compat_path = run_dir / f"semotif_orchestrator_{AML_TAG}" / "compatibility_graph_edges.csv"
    candidate_dir = run_dir / f"semotif_candidates_{AML_TAG}"
    rows = []
    for ratio in ratios:
        max_tx_id = threshold_for_ratio(full_raw_edges, ratio)
        raw_edges = max_tx_id
        evidence_records = count_aml_evidence_records_at_scale(evidence_dir, max_tx_id)
        compatibility_edges = count_aml_compat_edges_at_scale(compat_path, max_tx_id)
        assembled_candidates = count_aml_candidates_at_scale(candidate_dir, max_tx_id)
        runtime = full_runtime if math.isclose(ratio, 1.0) else None
        rows.append({
            "dataset": "amlworld",
            "scale_ratio": ratio,
            "raw_edges": raw_edges,
            "evidence_records": evidence_records,
            "compatibility_edges": compatibility_edges,
            "assembled_candidates": assembled_candidates,
            "runtime": runtime,
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build search-space reduction and AMLWorld scale-extension data.")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--summary_dir", default="outputs/summary")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scale_ratios", default=",".join(str(x) for x in SCALE_RATIOS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = PROJECT_ROOT / args.output_root
    summary_dir = PROJECT_ROOT / args.summary_dir
    ratios = [float(item) for item in args.scale_ratios.replace(",", " ").split() if item.strip()]

    search_rows = [aml_search_row(output_root, args.seed)]
    for dataset in ["elliptic", "dblp", "web_google"]:
        search_rows.append(public_search_row(dataset, output_root, args.seed))

    scale_rows = aml_scale_rows(output_root, args.seed, ratios)

    search_path = summary_dir / "search_space_reduction_4datasets.csv"
    scale_path = summary_dir / "scalability_amlworld_scale.csv"
    metadata_path = summary_dir / "search_space_scalability_metadata.json"
    write_csv(search_rows, search_path, SEARCH_COLUMNS)
    write_csv(scale_rows, scale_path, SCALE_COLUMNS)

    metadata = {
        "seed": args.seed,
        "outputs": {
            "search_space_reduction": rel(search_path),
            "amlworld_scale": rel(scale_path),
        },
        "field_definitions": {
            "raw_edges": "Processed edge count, or AMLWorld transaction count from semotif_global/global_summary.json.",
            "boundary_vertices": "AMLWorld: unique local evidence vertices. Public datasets: endpoints of visible edges that appear in more than one partition.",
            "evidence_records": "AMLWorld: local evidence rows read by orchestrator. Public datasets: sum of partition visible-edge records.",
            "compatibility_edges": "AMLWorld: kept orchestrator compatibility graph edges. Public datasets: unique visible edges shared by more than one partition.",
            "assembled_candidates": "Candidate pool size emitted by CoMot candidate assembly before the selected inspection prefix.",
            "topk_candidates": "Top-1000 inspection prefix size when the assembled pool is at least 1000.",
            "scale_runtime": "Verified end-to-end AMLWorld runtime from runtime.json. Partial scale ratios are N/A unless separately rerun.",
        },
        "notes": [
            "No processed dataset files are modified.",
            "The AMLWorld scale counts are recovered from existing full-run artifacts by tx_id prefix at the requested scale ratios.",
            "Only scale_ratio=1.0 has a verified end-to-end runtime in the current repository.",
            "Partial-scale runtime values are N/A to avoid reporting artifact-recovery time as pipeline runtime.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "search_space_reduction": rel(search_path),
        "amlworld_scale": rel(scale_path),
        "metadata": rel(metadata_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
