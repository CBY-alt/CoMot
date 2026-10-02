import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.query_view import build_method_query_view


BACKUP_SUFFIX = ".before_protocol_fix"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def backup_before_protocol_fix(path: Path) -> Path:
    backup = path.with_name(path.stem + BACKUP_SUFFIX + path.suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def protocol_source_path(dataset_dir: Path, filename: str) -> Path:
    path = dataset_dir / filename
    backup = path.with_name(path.stem + BACKUP_SUFFIX + path.suffix)
    return backup if backup.exists() else path


def sort_key(value: Any) -> Tuple[int, Any]:
    text = str(value)
    if text.startswith("e") and text[1:].isdigit():
        return (0, int(text[1:]))
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def load_edges(dataset_dir: Path, directed: bool) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Set[str]], Dict[str, List[Tuple[str, str]]]]:
    edge_lookup: Dict[str, Tuple[str, str]] = {}
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    incident_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    with open(dataset_dir / "edges.csv", "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edge_id = str(row["edge_id"])
            src = str(row["src"])
            dst = str(row["dst"])
            edge_lookup[edge_id] = (src, dst)
            adjacency[src].add(dst)
            adjacency[dst].add(src)
            incident_edges[src].append((edge_id, dst))
            incident_edges[dst].append((edge_id, src))
    for node in list(adjacency):
        adjacency[node] = set(sorted(adjacency[node], key=sort_key))
        incident_edges[node] = sorted(incident_edges[node], key=lambda item: sort_key(item[1]))
    return edge_lookup, adjacency, incident_edges


def load_partitions(dataset_dir: Path) -> Dict[str, str]:
    partitions = load_json(dataset_dir / "partitions.json")
    node_to_partition: Dict[str, str] = {}
    for part in partitions:
        part_id = str(part["partition_id"])
        for node in part.get("visible_nodes", []):
            node_to_partition[str(node)] = part_id
    return node_to_partition


def is_cross_edge(edge_id: str, edge_lookup: Dict[str, Tuple[str, str]], node_to_partition: Dict[str, str]) -> bool:
    src, dst = edge_lookup[edge_id]
    return bool(node_to_partition.get(src)) and bool(node_to_partition.get(dst)) and node_to_partition.get(src) != node_to_partition.get(dst)


def edge_in_anchor_onehop(edge_id: str, anchor: str, edge_lookup: Dict[str, Tuple[str, str]], adjacency: Dict[str, Set[str]]) -> bool:
    src, dst = edge_lookup[edge_id]
    closure = set(adjacency.get(anchor, set()))
    closure.add(anchor)
    return src in closure and dst in closure


def edge_incident_to_anchor(edge_id: str, anchor: str, edge_lookup: Dict[str, Tuple[str, str]]) -> bool:
    src, dst = edge_lookup[edge_id]
    return src == anchor or dst == anchor


def cn_positive(edge_id: str, edge_lookup: Dict[str, Tuple[str, str]], adjacency: Dict[str, Set[str]]) -> bool:
    src, dst = edge_lookup[edge_id]
    return bool(adjacency.get(src, set()) & adjacency.get(dst, set()))


def true_nodes_from_edges(edge_ids: Iterable[str], edge_lookup: Dict[str, Tuple[str, str]]) -> List[str]:
    nodes: Set[str] = set()
    for edge_id in edge_ids:
        src, dst = edge_lookup[edge_id]
        nodes.add(src)
        nodes.add(dst)
    return sorted(nodes, key=sort_key)


def edge_rows(edge_ids: Iterable[str], edge_lookup: Dict[str, Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    rows = [(edge_id, edge_lookup[edge_id][0], edge_lookup[edge_id][1]) for edge_id in edge_ids]
    return sorted(rows, key=lambda row: sort_key(row[0]))


def q_edges(edge_ids: Iterable[str], edge_lookup: Dict[str, Tuple[str, str]], node_to_q: Dict[str, str]) -> List[Dict[str, str]]:
    return [
        {"src": node_to_q[src], "dst": node_to_q[dst], "edge_id": edge_id}
        for edge_id, src, dst in edge_rows(edge_ids, edge_lookup)
    ]


def choose_protocol_view(
    true_edges: Sequence[str],
    edge_lookup: Dict[str, Tuple[str, str]],
    adjacency: Dict[str, Set[str]],
    node_to_partition: Dict[str, str],
    allow_local: bool = False,
) -> Optional[Tuple[List[str], List[str], Dict[str, float]]]:
    true_nodes = true_nodes_from_edges(true_edges, edge_lookup)
    if len(true_nodes) < 3 or len(true_edges) < 2:
        return None

    best: Optional[Tuple[float, str, List[str], Dict[str, float]]] = None
    for anchor in true_nodes:
        hidden_nodes = [node for node in true_nodes if node != anchor]
        if not hidden_nodes:
            continue
        nonlocal_edges = [edge_id for edge_id in true_edges if not edge_in_anchor_onehop(edge_id, anchor, edge_lookup, adjacency)]
        cross_nonlocal = [edge_id for edge_id in nonlocal_edges if is_cross_edge(edge_id, edge_lookup, node_to_partition)]
        cross_edges = [edge_id for edge_id in true_edges if is_cross_edge(edge_id, edge_lookup, node_to_partition)]
        if cross_nonlocal:
            hidden_edges = cross_nonlocal
        elif nonlocal_edges:
            hidden_edges = nonlocal_edges
        elif allow_local and cross_edges:
            hidden_edges = cross_edges
        elif allow_local:
            hidden_edges = list(true_edges)
        else:
            continue
        hidden_edges = sorted(set(hidden_edges), key=sort_key)
        if not hidden_edges:
            continue
        local_ratio = sum(edge_in_anchor_onehop(eid, anchor, edge_lookup, adjacency) for eid in hidden_edges) / len(hidden_edges)
        incident_ratio = sum(edge_incident_to_anchor(eid, anchor, edge_lookup) for eid in hidden_edges) / len(hidden_edges)
        cn_ratio = sum(cn_positive(eid, edge_lookup, adjacency) for eid in hidden_edges) / len(hidden_edges)
        cross_ratio = sum(is_cross_edge(eid, edge_lookup, node_to_partition) for eid in hidden_edges) / len(hidden_edges)
        # Prefer non-local, cross-partition hidden subgraphs; penalize CN-positive and anchor-incident edges.
        score = (1.0 - local_ratio) * 5.0 + cross_ratio * 2.0 - incident_ratio - cn_ratio * 0.5 + len(hidden_edges) * 0.02
        if best is None or score > best[0]:
            best = (
                score,
                anchor,
                hidden_edges,
                {
                    "hidden_edge_anchor_onehop_closure_ratio": local_ratio,
                    "hidden_edge_anchor_incident_ratio": incident_ratio,
                    "cn_positive_hidden_edge_ratio": cn_ratio,
                    "hidden_edge_cross_partition_ratio": cross_ratio,
                },
            )
    if best is None:
        return None
    _, anchor, hidden_edges, stats = best
    return [anchor], hidden_edges, stats


def build_query_and_gt(
    query_id: str,
    motif_type: str,
    true_edges: Sequence[str],
    anchor_nodes: Sequence[str],
    hidden_edges: Sequence[str],
    edge_lookup: Dict[str, Tuple[str, str]],
    node_to_partition: Dict[str, str],
    description: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    true_edges_sorted = sorted(set(true_edges), key=sort_key)
    true_nodes = true_nodes_from_edges(true_edges_sorted, edge_lookup)
    true_partitions = sorted({node_to_partition[node] for node in true_nodes if node in node_to_partition})
    hidden_nodes = [node for node in true_nodes if node not in set(anchor_nodes)]
    node_to_q = {node: f"q{i}" for i, node in enumerate(true_nodes)}
    query = {
        "query_id": query_id,
        "motif_type": motif_type,
        "nodes": [node_to_q[node] for node in true_nodes],
        "edges": q_edges(true_edges_sorted, edge_lookup, node_to_q),
        "anchor_nodes": list(anchor_nodes),
        "hidden_nodes": hidden_nodes,
        "hidden_edges": sorted(set(hidden_edges), key=sort_key),
        "description": description,
    }
    gt = {
        "query_id": query_id,
        "true_nodes": true_nodes,
        "true_edges": true_edges_sorted,
        "true_partitions": true_partitions,
        "motif_instance_id": query_id,
    }
    return query, gt


def valid_query(query: Dict[str, Any], gt: Dict[str, Any]) -> bool:
    return (
        len(gt["true_nodes"]) >= 3
        and len(gt["true_edges"]) >= 2
        and len(gt["true_partitions"]) >= 2
        and len(query["hidden_edges"]) >= 1
        and len(query["anchor_nodes"]) < len(gt["true_nodes"])
        and len(query["hidden_nodes"]) >= 1
    )


def source_items(dataset_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    queries = load_json(dataset_dir / "query_motifs.json")
    gt_by_id = {item["query_id"]: item for item in load_json(dataset_dir / "ground_truth.json")}
    return queries, gt_by_id


def candidate_from_existing(
    item: Dict[str, Any],
    motif_type: str,
    edge_lookup: Dict[str, Tuple[str, str]],
    adjacency: Dict[str, Set[str]],
    node_to_partition: Dict[str, str],
    allow_local: bool,
) -> Optional[Tuple[List[str], List[str], List[str], Dict[str, float]]]:
    true_edges = [str(edge_id) for edge_id in item.get("true_edges", []) if str(edge_id) in edge_lookup]
    view = choose_protocol_view(true_edges, edge_lookup, adjacency, node_to_partition, allow_local=allow_local)
    if view is None:
        return None
    anchors, hidden_edges, stats = view
    return true_edges, anchors, hidden_edges, stats


def make_from_existing(
    dataset_name: str,
    source_queries: List[Dict[str, Any]],
    source_gt: List[Dict[str, Any]],
    target_counts: Dict[str, int],
    edge_lookup: Dict[str, Tuple[str, str]],
    adjacency: Dict[str, Set[str]],
    node_to_partition: Dict[str, str],
    allow_local_types: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    queries: List[Dict[str, Any]] = []
    gts: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    used_edges: Set[Tuple[str, ...]] = set()
    query_type_by_id = {str(query["query_id"]): str(query.get("motif_type", "")) for query in source_queries}
    for item in source_gt:
        source_type = str(item.get("motif_type") or query_type_by_id.get(str(item.get("query_id")), "") or item.get("query_id") or "")
        # Current repaired files store motif type in query ids, so use explicit prefixes when possible.
        if dataset_name == "dblp":
            for known in ("triangle", "cycle4", "star", "dense_subgraph", "bridge_motif"):
                if known in source_type:
                    source_type = known
                    break
        else:
            for known in ("directed_chain", "directed_star", "feed_forward", "directed_cycle", "dense_hyperlink_block"):
                if known in source_type:
                    source_type = known
                    break
        if source_type not in target_counts or counts[source_type] >= target_counts[source_type]:
            continue
        candidate = candidate_from_existing(
            item,
            source_type,
            edge_lookup,
            adjacency,
            node_to_partition,
            allow_local=source_type in allow_local_types,
        )
        if candidate is None:
            continue
        true_edges, anchors, hidden_edges, _ = candidate
        edge_key = tuple(sorted(true_edges, key=sort_key))
        if edge_key in used_edges:
            continue
        query_id = f"{dataset_name.upper()}_PROTOCOL_{source_type}_{counts[source_type]:06d}"
        query, gt = build_query_and_gt(
            query_id,
            source_type,
            true_edges,
            anchors,
            hidden_edges,
            edge_lookup,
            node_to_partition,
            f"{dataset_name} protocol-fixed {source_type} recovery query.",
        )
        if not valid_query(query, gt):
            continue
        queries.append(query)
        gts.append(gt)
        counts[source_type] += 1
        used_edges.add(edge_key)
    return queries, gts, counts


def sample_bridge_motifs(
    edge_lookup: Dict[str, Tuple[str, str]],
    adjacency: Dict[str, Set[str]],
    incident_edges: Dict[str, List[Tuple[str, str]]],
    node_to_partition: Dict[str, str],
    start_index: int,
    target: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    nodes = sorted(adjacency, key=sort_key)
    queries: List[Dict[str, Any]] = []
    gts: List[Dict[str, Any]] = []
    used: Set[Tuple[str, ...]] = set()
    attempts = 0
    while len(queries) < target and attempts < target * 5000:
        attempts += 1
        a = rng.choice(nodes)
        if len(incident_edges.get(a, [])) < 1:
            continue
        e1, b = rng.choice(incident_edges[a])
        b_neighbors = [item for item in incident_edges.get(b, []) if item[1] != a]
        if not b_neighbors:
            continue
        e2, c = rng.choice(b_neighbors)
        c_neighbors = [item for item in incident_edges.get(c, []) if item[1] not in {a, b}]
        if not c_neighbors:
            continue
        e3, d = rng.choice(c_neighbors)
        true_edges = sorted({e1, e2, e3}, key=sort_key)
        if len(true_edges) < 3:
            continue
        true_nodes = true_nodes_from_edges(true_edges, edge_lookup)
        if len(true_nodes) < 4:
            continue
        parts = {node_to_partition.get(node) for node in true_nodes if node_to_partition.get(node)}
        if len(parts) < 2:
            continue
        view = choose_protocol_view(true_edges, edge_lookup, adjacency, node_to_partition, allow_local=False)
        if view is None:
            continue
        anchors, hidden_edges, stats = view
        if stats["hidden_edge_anchor_onehop_closure_ratio"] >= 1.0:
            continue
        key = tuple(true_edges)
        if key in used:
            continue
        query_id = f"DBLP_PROTOCOL_bridge_motif_{start_index + len(queries):06d}"
        query, gt = build_query_and_gt(
            query_id,
            "bridge_motif",
            true_edges,
            anchors,
            hidden_edges,
            edge_lookup,
            node_to_partition,
            "DBLP protocol-fixed bridge motif crossing collaboration partitions.",
        )
        if not valid_query(query, gt):
            continue
        queries.append(query)
        gts.append(gt)
        used.add(key)
    return queries, gts


def sample_directed_chain_motifs(
    edge_lookup: Dict[str, Tuple[str, str]],
    adjacency: Dict[str, Set[str]],
    node_to_partition: Dict[str, str],
    start_index: int,
    target: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed + 17)
    out_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for edge_id, (src, dst) in edge_lookup.items():
        out_edges[src].append((edge_id, dst))
    for node in out_edges:
        out_edges[node] = sorted(out_edges[node], key=lambda item: sort_key(item[1]))
    starts = sorted(out_edges, key=sort_key)
    queries: List[Dict[str, Any]] = []
    gts: List[Dict[str, Any]] = []
    used: Set[Tuple[str, ...]] = set()
    attempts = 0
    while len(queries) < target and attempts < target * 8000:
        attempts += 1
        a = rng.choice(starts)
        if not out_edges.get(a):
            continue
        e1, b = rng.choice(out_edges[a])
        if not out_edges.get(b):
            continue
        b_options = [(eid, node) for eid, node in out_edges[b] if node != a]
        if not b_options:
            continue
        e2, c = rng.choice(b_options)
        if not out_edges.get(c):
            continue
        c_options = [(eid, node) for eid, node in out_edges[c] if node not in {a, b}]
        if not c_options:
            continue
        e3, d = rng.choice(c_options)
        true_edges = sorted({e1, e2, e3}, key=sort_key)
        if len(true_edges) < 3:
            continue
        true_nodes = true_nodes_from_edges(true_edges, edge_lookup)
        if len(true_nodes) < 4:
            continue
        parts = {node_to_partition.get(node) for node in true_nodes if node_to_partition.get(node)}
        if len(parts) < 2:
            continue
        view = choose_protocol_view(true_edges, edge_lookup, adjacency, node_to_partition, allow_local=False)
        if view is None:
            continue
        anchors, hidden_edges, stats = view
        if stats["hidden_edge_anchor_onehop_closure_ratio"] >= 1.0:
            continue
        key = tuple(true_edges)
        if key in used:
            continue
        query_id = f"WEB_GOOGLE_PROTOCOL_directed_chain_generated_{start_index + len(queries):06d}"
        query, gt = build_query_and_gt(
            query_id,
            "directed_chain",
            true_edges,
            anchors,
            hidden_edges,
            edge_lookup,
            node_to_partition,
            "Web-Google protocol-fixed directed chain crossing hyperlink partitions.",
        )
        if not valid_query(query, gt):
            continue
        queries.append(query)
        gts.append(gt)
        used.add(key)
    return queries, gts


def repair_dblp(dataset_dir: Path, seed: int) -> Dict[str, Any]:
    edge_lookup, adjacency, incident_edges = load_edges(dataset_dir, directed=False)
    node_to_partition = load_partitions(dataset_dir)
    before_queries = load_json(protocol_source_path(dataset_dir, "query_motifs.json"))
    before_gt = load_json(protocol_source_path(dataset_dir, "ground_truth.json"))
    before_stats = protocol_stats(before_queries, before_gt, edge_lookup, adjacency, node_to_partition)

    target_counts = {"bridge_motif": 652, "star": 250, "cycle4": 200, "dense_subgraph": 150, "triangle": 50}
    existing_queries, existing_gt, counts = make_from_existing(
        "dblp",
        before_queries,
        before_gt,
        target_counts,
        edge_lookup,
        adjacency,
        node_to_partition,
        allow_local_types={"triangle"},
    )
    bridge_needed = max(0, target_counts["bridge_motif"] - counts["bridge_motif"])
    bridge_queries, bridge_gt = sample_bridge_motifs(
        edge_lookup,
        adjacency,
        incident_edges,
        node_to_partition,
        counts["bridge_motif"],
        bridge_needed,
        seed,
    )
    queries = existing_queries + bridge_queries
    gts = existing_gt + bridge_gt

    # Keep target size stable at 1000 when possible; deterministic sort improves reproducibility.
    paired = sorted(zip(queries, gts), key=lambda pair: (pair[0]["motif_type"], pair[0]["query_id"]))
    queries = [pair[0] for pair in paired[:1000]]
    gts = [pair[1] for pair in paired[:1000]]
    after_stats = protocol_stats(queries, gts, edge_lookup, adjacency, node_to_partition)
    write_repaired_dataset(dataset_dir, queries, gts)
    return {"before": before_stats, "after": after_stats, "query_count": len(queries)}


def repair_web_google(dataset_dir: Path, seed: int) -> Dict[str, Any]:
    edge_lookup, adjacency, _incident_edges = load_edges(dataset_dir, directed=True)
    node_to_partition = load_partitions(dataset_dir)
    before_queries = load_json(protocol_source_path(dataset_dir, "query_motifs.json"))
    before_gt = load_json(protocol_source_path(dataset_dir, "ground_truth.json"))
    before_stats = protocol_stats(before_queries, before_gt, edge_lookup, adjacency, node_to_partition)
    target_counts = {
        "directed_chain": 250,
        "directed_star": 250,
        "directed_cycle": 200,
        "dense_hyperlink_block": 200,
        "feed_forward": 100,
    }
    queries, gts, counts = make_from_existing(
        "web_google",
        before_queries,
        before_gt,
        target_counts,
        edge_lookup,
        adjacency,
        node_to_partition,
        allow_local_types={"feed_forward"},
    )
    chain_needed = max(0, 1000 - len(queries))
    if chain_needed:
        chain_queries, chain_gt = sample_directed_chain_motifs(
            edge_lookup,
            adjacency,
            node_to_partition,
            start_index=counts["directed_chain"],
            target=chain_needed,
            seed=seed,
        )
        queries.extend(chain_queries)
        gts.extend(chain_gt)
    paired = sorted(zip(queries, gts), key=lambda pair: (pair[0]["motif_type"], pair[0]["query_id"]))
    queries = [pair[0] for pair in paired[:1000]]
    gts = [pair[1] for pair in paired[:1000]]
    after_stats = protocol_stats(queries, gts, edge_lookup, adjacency, node_to_partition)
    write_repaired_dataset(dataset_dir, queries, gts)
    return {"before": before_stats, "after": after_stats, "query_count": len(queries), "accepted_counts": dict(counts)}


def write_repaired_dataset(dataset_dir: Path, queries: List[Dict[str, Any]], gts: List[Dict[str, Any]]) -> None:
    for name in ("query_motifs.json", "ground_truth.json", "method_queries.json"):
        backup_before_protocol_fix(dataset_dir / name)
    write_json(dataset_dir / "query_motifs.json", queries)
    write_json(dataset_dir / "ground_truth.json", gts)
    build_method_query_view(dataset_dir / "query_motifs.json", dataset_dir / "method_queries.json")


def protocol_stats(
    queries: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    edge_lookup: Dict[str, Tuple[str, str]],
    adjacency: Dict[str, Set[str]],
    node_to_partition: Dict[str, str],
) -> Dict[str, Any]:
    gt_by_id = {item["query_id"]: item for item in gts}
    motif_counter: Counter[str] = Counter()
    true_nodes_counts: List[int] = []
    true_edges_counts: List[int] = []
    anchor_counts: List[int] = []
    hidden_node_counts: List[int] = []
    hidden_edge_counts: List[int] = []
    partition_counts: List[int] = []
    total_hidden_edges = 0
    anchor_incident = 0
    onehop = 0
    cn_pos = 0
    onehop_queries = 0
    for query in queries:
        qid = query["query_id"]
        gt = gt_by_id.get(qid, {})
        motif_counter[str(query.get("motif_type", "unknown"))] += 1
        true_nodes = [str(node) for node in gt.get("true_nodes", [])]
        true_edges = [str(edge) for edge in gt.get("true_edges", []) if str(edge) in edge_lookup]
        anchors = [str(node) for node in query.get("anchor_nodes", [])]
        hidden_edges = [str(edge) for edge in query.get("hidden_edges", []) if str(edge) in edge_lookup]
        true_nodes_counts.append(len(true_nodes))
        true_edges_counts.append(len(true_edges))
        anchor_counts.append(len(anchors))
        hidden_node_counts.append(len(query.get("hidden_nodes", [])))
        hidden_edge_counts.append(len(hidden_edges))
        partition_counts.append(len(set(gt.get("true_partitions", []))))
        q_onehop = 0
        for edge_id in hidden_edges:
            total_hidden_edges += 1
            if any(edge_incident_to_anchor(edge_id, anchor, edge_lookup) for anchor in anchors):
                anchor_incident += 1
            if any(edge_in_anchor_onehop(edge_id, anchor, edge_lookup, adjacency) for anchor in anchors):
                onehop += 1
                q_onehop += 1
            if cn_positive(edge_id, edge_lookup, adjacency):
                cn_pos += 1
        if hidden_edges and (q_onehop / len(hidden_edges) >= 0.75):
            onehop_queries += 1

    def mean(values: Sequence[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "query_count": len(queries),
        "motif_type_distribution": dict(sorted(motif_counter.items())),
        "avg_true_nodes": mean(true_nodes_counts),
        "avg_true_edges": mean(true_edges_counts),
        "avg_anchor_nodes": mean(anchor_counts),
        "avg_hidden_nodes": mean(hidden_node_counts),
        "avg_hidden_edges": mean(hidden_edge_counts),
        "avg_partitions": mean(partition_counts),
        "cross_partition_ratio": sum(1 for value in partition_counts if value >= 2) / len(partition_counts) if partition_counts else 0.0,
        "hidden_edge_anchor_incident_ratio": anchor_incident / total_hidden_edges if total_hidden_edges else 0.0,
        "hidden_edge_anchor_onehop_closure_ratio": onehop / total_hidden_edges if total_hidden_edges else 0.0,
        "cn_positive_hidden_edge_ratio": cn_pos / total_hidden_edges if total_hidden_edges else 0.0,
        "one_hop_link_prediction_style_query_ratio": onehop_queries / len(queries) if queries else 0.0,
    }


def write_report(dataset: str, report: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{dataset}_protocol_fix_report.csv"
    md_path = output_dir / f"{dataset}_protocol_fix_report.md"
    rows = []
    before = report["before"]
    after = report["after"]
    for key in [
        "query_count",
        "motif_type_distribution",
        "avg_true_nodes",
        "avg_true_edges",
        "avg_anchor_nodes",
        "avg_hidden_nodes",
        "avg_hidden_edges",
        "avg_partitions",
        "cross_partition_ratio",
        "hidden_edge_anchor_incident_ratio",
        "hidden_edge_anchor_onehop_closure_ratio",
        "cn_positive_hidden_edge_ratio",
        "one_hop_link_prediction_style_query_ratio",
    ]:
        rows.append(
            {
                "dataset": dataset,
                "metric": key,
                "before": json.dumps(before.get(key), sort_keys=True) if isinstance(before.get(key), dict) else before.get(key),
                "after": json.dumps(after.get(key), sort_keys=True) if isinstance(after.get(key), dict) else after.get(key),
            }
        )
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "metric", "before", "after"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {dataset} Protocol Fix Report",
        "",
        "This report compares protocol statistics before and after regenerating instance-level queries from existing graph edges. No new non-existent edges were generated and no main experiments were rerun.",
        "",
        "| Metric | Before | After |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['before']} | {row['after']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lower `hidden_edge_anchor_onehop_closure_ratio` and `one_hop_link_prediction_style_query_ratio` indicate less degeneration into one-hop link prediction.",
            "- `method_queries.json` was regenerated as a method-facing query view excluding evaluation-only GT fields.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix DBLP/Web-Google query protocols away from one-hop link prediction.")
    parser.add_argument("--dataset", choices=["dblp", "web_google", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ["dblp", "web_google"] if args.dataset == "all" else [args.dataset]
    output_dir = PROJECT_ROOT / args.output_dir
    summaries: Dict[str, Any] = {}
    for dataset in datasets:
        dataset_dir = PROJECT_ROOT / "data" / "processed" / dataset
        if dataset == "dblp":
            report = repair_dblp(dataset_dir, args.seed)
        else:
            report = repair_web_google(dataset_dir, args.seed)
        write_report(dataset, report, output_dir)
        summaries[dataset] = report
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
