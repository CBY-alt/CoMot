import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.query_view import build_method_query_view


Edge = Tuple[str, str, str]


def backup_once(path: Path) -> Path:
    backup = path.with_name(path.stem + ".original" + path.suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def read_edges(path: Path) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, List[Tuple[str, str]]], Dict[str, List[Tuple[str, str]]]]:
    edge_lookup: Dict[str, Tuple[str, str]] = {}
    out_adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    in_adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edge_id = str(row["edge_id"])
            src = str(row["src"])
            dst = str(row["dst"])
            edge_lookup[edge_id] = (src, dst)
            out_adj[src].append((dst, edge_id))
            in_adj[dst].append((src, edge_id))
    for adj in (out_adj, in_adj):
        for node in adj:
            adj[node].sort(key=lambda item: (sort_key(item[0]), sort_key(item[1])))
    return edge_lookup, out_adj, in_adj


def read_nodes(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [str(row["node_id"]) for row in csv.DictReader(f)]


def read_time_steps(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    time_steps: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "node_id" not in reader.fieldnames:
            return {}
        time_col = "time_step" if "time_step" in reader.fieldnames else None
        if not time_col:
            return {}
        for row in reader:
            try:
                time_steps[str(row["node_id"])] = int(float(row[time_col]))
            except (TypeError, ValueError):
                continue
    return time_steps


def read_illicit_nodes(nodes_path: Path, labels_path: Path) -> Set[str]:
    illicit: Set[str] = set()
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("object_type", "")).lower() != "node":
                    continue
                label = str(row.get("label", "")).lower()
                label_name = str(row.get("label_name", "")).lower()
                if label in {"1", "illicit", "true", "risk", "suspicious"} or "illicit" in label_name:
                    illicit.add(str(row.get("object_id", "")))
    if illicit:
        return illicit
    with open(nodes_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = str(row.get("label", "")).lower()
            if label in {"1", "illicit", "true", "risk", "suspicious"}:
                illicit.add(str(row["node_id"]))
    return illicit


def stable_hash_int(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:12], 16)


def assign_hybrid_partitions(nodes: Iterable[str], time_steps: Dict[str, int], partition_num: int) -> Dict[str, str]:
    assignments = {}
    for node in nodes:
        time_component = time_steps.get(node, 0)
        bucket = (time_component + stable_hash_int(node)) % partition_num
        assignments[node] = f"P{bucket:03d}"
    return assignments


def write_partitions(path: Path, nodes: List[str], edge_lookup: Dict[str, Tuple[str, str]], assignments: Dict[str, str], partition_num: int) -> None:
    visible_nodes: Dict[str, List[str]] = {f"P{i:03d}": [] for i in range(partition_num)}
    visible_edges: Dict[str, List[str]] = {f"P{i:03d}": [] for i in range(partition_num)}
    for node in nodes:
        visible_nodes[assignments[node]].append(node)
    for edge_id, (src, dst) in edge_lookup.items():
        src_part = assignments.get(src)
        dst_part = assignments.get(dst)
        if src_part:
            visible_edges[src_part].append(edge_id)
        if dst_part and dst_part != src_part:
            visible_edges[dst_part].append(edge_id)
    payload = [
        {
            "partition_id": f"P{i:03d}",
            "visible_nodes": sorted(visible_nodes[f"P{i:03d}"], key=sort_key),
            "visible_edges": sorted(visible_edges[f"P{i:03d}"], key=sort_key),
            "hidden_nodes": [],
            "hidden_edges": [],
        }
        for i in range(partition_num)
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def collect_candidates(
    illicit_nodes: Set[str],
    out_adj: Dict[str, List[Tuple[str, str]]],
    in_adj: Dict[str, List[Tuple[str, str]]],
    edge_lookup: Dict[str, Tuple[str, str]],
    assignments: Dict[str, str],
) -> List[Dict[str, Any]]:
    pair_to_edge = {(src, dst): edge_id for edge_id, (src, dst) in edge_lookup.items()}
    candidates: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, Tuple[str, ...]]] = set()
    for center in sorted(illicit_nodes, key=sort_key):
        outgoing = [(dst, eid) for dst, eid in out_adj.get(center, []) if dst != center]
        incoming = [(src, eid) for src, eid in in_adj.get(center, []) if src != center]

        if len(outgoing) >= 2:
            add_candidate(candidates, seen, "fan_out", [eid for _, eid in outgoing[:3]], edge_lookup, assignments, center)
        if len(incoming) >= 2:
            add_candidate(candidates, seen, "fan_in", [eid for _, eid in incoming[:3]], edge_lookup, assignments, center)
        if incoming and outgoing:
            for src, in_eid in incoming[:8]:
                for dst, out_eid in outgoing[:8]:
                    if src != dst:
                        add_candidate(candidates, seen, "directed_chain", [in_eid, out_eid], edge_lookup, assignments, center)
                        break
                if any(c["center"] == center and c["motif_type"] == "directed_chain" for c in candidates[-2:]):
                    break

        cycle_edges = find_cycle_edges(center, outgoing, out_adj, pair_to_edge)
        if cycle_edges:
            add_candidate(candidates, seen, "cycle", cycle_edges, edge_lookup, assignments, center)

    candidates.sort(key=lambda item: (not item["is_cross_partition"], item["motif_type"], sort_key(item["center"]), item["edge_ids"]))
    return candidates


def add_candidate(
    candidates: List[Dict[str, Any]],
    seen: Set[Tuple[str, Tuple[str, ...]]],
    motif_type: str,
    edge_ids: List[str],
    edge_lookup: Dict[str, Tuple[str, str]],
    assignments: Dict[str, str],
    center: str,
) -> None:
    edge_ids = sorted(set(edge_ids), key=sort_key)
    key = (motif_type, tuple(edge_ids))
    if key in seen:
        return
    seen.add(key)
    nodes = sorted({node for edge_id in edge_ids for node in edge_lookup[edge_id]}, key=sort_key)
    if len(nodes) < 2:
        return
    partitions = sorted({assignments[node] for node in nodes if node in assignments})
    if not partitions:
        return
    candidates.append(
        {
            "motif_type": motif_type,
            "edge_ids": edge_ids,
            "nodes": nodes,
            "partitions": partitions,
            "center": center,
            "is_cross_partition": len(partitions) >= 2,
        }
    )


def find_cycle_edges(
    center: str,
    outgoing: List[Tuple[str, str]],
    out_adj: Dict[str, List[Tuple[str, str]]],
    pair_to_edge: Dict[Tuple[str, str], str],
) -> List[str]:
    for dst, out_eid in outgoing[:50]:
        back = pair_to_edge.get((dst, center))
        if back:
            return [out_eid, back]
    for mid, first_eid in outgoing[:30]:
        for dst, second_eid in out_adj.get(mid, [])[:30]:
            closing = pair_to_edge.get((dst, center))
            if closing and dst not in {center, mid}:
                return [first_eid, second_eid, closing]
    return []


def select_candidates(candidates: List[Dict[str, Any]], max_instances: int) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_type[item["motif_type"]].append(item)
    selected: List[Dict[str, Any]] = []
    motif_order = ["directed_chain", "fan_in", "fan_out", "cycle"]
    while len(selected) < max_instances:
        progressed = False
        for motif_type in motif_order:
            bucket = by_type.get(motif_type, [])
            if bucket:
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) >= max_instances:
                    break
        if not progressed:
            break
    return selected


def build_queries_and_ground_truth(
    selected: List[Dict[str, Any]],
    edge_lookup: Dict[str, Tuple[str, str]],
    assignments: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    queries = []
    ground_truth = []
    for idx, item in enumerate(selected):
        motif_id = f"ELLIPTIC_MOTIF_{idx:06d}"
        true_nodes = sorted(item["nodes"], key=sort_key)
        true_edges = sorted(item["edge_ids"], key=sort_key)
        true_partitions = sorted({assignments[node] for node in true_nodes if node in assignments})
        node_to_q = {node: f"q{i}" for i, node in enumerate(true_nodes)}
        query_edges = [
            {"src": node_to_q[edge_lookup[edge_id][0]], "dst": node_to_q[edge_lookup[edge_id][1]], "edge_id": edge_id}
            for edge_id in true_edges
        ]
        anchor_nodes = choose_anchor_nodes(item["center"], true_nodes, true_edges, edge_lookup, assignments)
        hidden_nodes = [node for node in true_nodes if node not in set(anchor_nodes)]
        hidden_edges = [edge_id for edge_id in true_edges if is_cross_edge(edge_id, edge_lookup, assignments)]
        if not hidden_edges:
            hidden_edges = [true_edges[0]]
        queries.append(
            {
                "query_id": motif_id,
                "motif_type": item["motif_type"],
                "nodes": [node_to_q[node] for node in true_nodes],
                "edges": query_edges,
                "anchor_nodes": anchor_nodes,
                "hidden_nodes": hidden_nodes,
                "hidden_edges": hidden_edges,
                "description": (
                    f"Elliptic {item['motif_type']} motif instance {motif_id}; "
                    "query topology is a directed anonymized transaction-neighborhood subgraph."
                ),
            }
        )
        ground_truth.append(
            {
                "query_id": motif_id,
                "true_nodes": true_nodes,
                "true_edges": true_edges,
                "true_partitions": true_partitions,
                "motif_instance_id": motif_id,
            }
        )
    return queries, ground_truth


def choose_anchor_nodes(center: str, true_nodes: List[str], true_edges: List[str], edge_lookup: Dict[str, Tuple[str, str]], assignments: Dict[str, str]) -> List[str]:
    cross_sources = [edge_lookup[edge_id][0] for edge_id in true_edges if is_cross_edge(edge_id, edge_lookup, assignments)]
    candidates = cross_sources or ([center] if center in true_nodes else true_nodes)
    anchor = sorted(set(candidates), key=sort_key)[0]
    if len(true_nodes) > 1 and len({anchor}) == len(true_nodes):
        anchor = true_nodes[0]
    return [anchor]


def is_cross_edge(edge_id: str, edge_lookup: Dict[str, Tuple[str, str]], assignments: Dict[str, str]) -> bool:
    src, dst = edge_lookup[edge_id]
    return assignments.get(src) != assignments.get(dst)


def update_metadata(path: Path, query_count: int, partition_num: int, before_ratio: float, after_ratio: float) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    metadata["num_motif_instances"] = query_count
    metadata["has_ground_truth"] = query_count > 0
    metadata["motif_construction"] = {
        "source": "processed Elliptic graph",
        "center_nodes": "illicit-labeled transaction nodes",
        "motif_types": ["directed_chain", "fan_in", "fan_out", "cycle"],
        "sampling": "deterministic candidates from directed k-hop transaction neighborhoods; cross-partition candidates preferred",
        "max_instances": query_count,
    }
    metadata["partition_construction"] = {
        "partition_num": partition_num,
        "strategy": "hash/time hybrid partition: P[(time_step + md5(node_id)) mod partition_num]",
        "reason": "pure time-step partitions produced zero cross-partition GT motifs for the previous illicit ego construction",
    }
    metadata["cross_partition_repair"] = {
        "before_ratio": before_ratio,
        "after_ratio": after_ratio,
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def repair(dataset_dir: Path, partition_num: int, max_instances: int) -> Dict[str, Any]:
    query_path = dataset_dir / "query_motifs.json"
    gt_path = dataset_dir / "ground_truth.json"
    partitions_path = dataset_dir / "partitions.json"
    metadata_path = dataset_dir / "metadata.json"
    backup_once(query_path)
    backup_once(gt_path)
    backup_once(partitions_path)

    source_gt = json.loads(gt_path.read_text(encoding="utf-8"))
    before_ratio = cross_ratio(source_gt)

    nodes = read_nodes(dataset_dir / "nodes.csv")
    time_steps = read_time_steps(dataset_dir / "node_features.csv")
    illicit_nodes = read_illicit_nodes(dataset_dir / "nodes.csv", dataset_dir / "labels.csv")
    if not illicit_nodes:
        raise ValueError("No illicit nodes found in labels.csv or nodes.csv; cannot construct Elliptic illicit-centered motifs.")
    edge_lookup, out_adj, in_adj = read_edges(dataset_dir / "edges.csv")
    assignments = assign_hybrid_partitions(nodes, time_steps, partition_num)
    write_partitions(partitions_path, nodes, edge_lookup, assignments, partition_num)

    candidates = collect_candidates(illicit_nodes, out_adj, in_adj, edge_lookup, assignments)
    selected = select_candidates(candidates, max_instances)
    if not selected:
        raise ValueError("No motif candidates could be constructed from illicit neighborhoods.")
    queries, ground_truth = build_queries_and_ground_truth(selected, edge_lookup, assignments)
    query_path.write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8")
    gt_path.write_text(json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8")
    build_method_query_view(query_path, dataset_dir / "method_queries.json")
    after_ratio = cross_ratio(ground_truth)
    update_metadata(metadata_path, len(queries), partition_num, before_ratio, after_ratio)

    motif_counter = Counter(query["motif_type"] for query in queries)
    return {
        "query_count": len(queries),
        "ground_truth_count": len(ground_truth),
        "query_ids_match": {q["query_id"] for q in queries} == {g["query_id"] for g in ground_truth},
        "cross_partition_ratio_before": before_ratio,
        "cross_partition_ratio_after": after_ratio,
        "motif_type_distribution": dict(sorted(motif_counter.items())),
        "anchor_stats": summarize([len(q["anchor_nodes"]) for q in queries]),
        "hidden_node_stats": summarize([len(q["hidden_nodes"]) for q in queries]),
        "hidden_edge_stats": summarize([len(q["hidden_edges"]) for q in queries]),
        "raw_basis": "processed graph rebuilt from data/processed/elliptic; raw files may be used to reproduce the processed graph, but this repair preserves existing node/edge ids and labels",
        "outputs": {
            "partitions": str(partitions_path),
            "query_motifs": str(query_path),
            "ground_truth": str(gt_path),
            "method_queries": str(dataset_dir / "method_queries.json"),
            "partitions_backup": str(partitions_path.with_name("partitions.original.json")),
            "query_backup": str(query_path.with_name("query_motifs.original.json")),
            "ground_truth_backup": str(gt_path.with_name("ground_truth.original.json")),
        },
    }


def cross_ratio(gt: List[Dict[str, Any]]) -> float:
    return sum(1 for item in gt if len(item.get("true_partitions", [])) >= 2) / len(gt) if gt else 0.0


def summarize(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def sort_key(value: Any) -> Tuple[int, Any]:
    text = str(value)
    if text.startswith("e") and text[1:].isdigit():
        return (0, int(text[1:]))
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair Elliptic into a cross-partition motif recovery task.")
    parser.add_argument("--dataset_dir", default="data/processed/elliptic")
    parser.add_argument("--partition_num", type=int, default=5)
    parser.add_argument("--max_instances", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = repair(Path(args.dataset_dir), partition_num=args.partition_num, max_instances=args.max_instances)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
