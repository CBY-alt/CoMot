import hashlib
import math
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from methods.base_method import BaseMethod
from methods.baselines.common import elapsed, node_partition_map, now, prediction, save_predictions


class CSGMBaseline(BaseMethod):
    """Collaborative scatter-gather mining baseline adapted to standard queries."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        return None

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        self.predictions = csgm_predictions(
            dataset=dataset,
            max_predictions=int(cfg.get("max_predictions", 1000)),
            bfs_depth=int(cfg.get("bfs_depth", 2)),
            max_frontier=int(cfg.get("max_frontier", 64)),
            max_candidate_nodes=int(cfg.get("max_candidate_nodes", 18)),
            minhash_dim=int(cfg.get("minhash_dim", 16)),
            directed=bool(cfg.get("directed", False)),
            partnership_mode=str(cfg.get("partnership_mode", "bilateral")),
            min_bilateral_strength=float(cfg.get("min_bilateral_strength", 0.0)),
            max_anchors_per_party=int(cfg.get("max_anchors_per_party", 0)),
            amlworld_max_candidate_nodes=int(cfg.get("amlworld_max_candidate_nodes", 4)),
            amlworld_max_anchors_per_party=int(cfg.get("amlworld_max_anchors_per_party", 1)),
        )
        for item in self.predictions:
            item["metadata"]["paper"] = (
                "Tian Z, Ding Y, Yu X, et al. Towards collaborative anti-money laundering "
                "among financial institutions. Proceedings of the ACM Web Conference 2025."
            )
            item["metadata"]["implementation_note"] = (
                "adapted collaborative scatter-gather mining over observable partitioned evidence; "
                "privacy-preserving Bloom-filter communication is represented by deterministic set signatures"
            )
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions

        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)


def csgm_predictions(
    dataset: Any,
    max_predictions: int,
    bfs_depth: int,
    max_frontier: int,
    max_candidate_nodes: int,
    minhash_dim: int,
    directed: bool,
    partnership_mode: str,
    min_bilateral_strength: float,
    max_anchors_per_party: int,
    amlworld_max_candidate_nodes: int,
    amlworld_max_anchors_per_party: int,
) -> List[Dict[str, Any]]:
    start = now()
    dataset_name = str(getattr(dataset, "dataset_name", "")).lower()
    if dataset_name == "amlworld":
        max_candidate_nodes = min(max_candidate_nodes, max(2, amlworld_max_candidate_nodes))
        max_anchors_per_party = max(1, amlworld_max_anchors_per_party)
    adjacency, edge_lookup = dataset.graph(directed=directed)
    partitions = node_partition_map(dataset, max_nodes_per_partition=None)
    primary_rows = []
    extra_rows = []

    for query in dataset.load_query_motifs():
        query_id = str(query.get("query_id", "CSGM_query"))
        anchors = [str(node) for node in query.get("anchor_nodes", []) if str(node) in adjacency]
        if not anchors:
            continue
        target_nodes = bounded_target_size(query, max_candidate_nodes)
        target_edges = max(1, len(query.get("edges", [])))
        motif_type = str(query.get("motif_type", "")).lower()

        local_sets = scatter_anchor_sets(
            anchors=anchors,
            adjacency=adjacency,
            edge_lookup=edge_lookup,
            partitions=partitions,
            bfs_depth=max(1, bfs_depth),
            max_frontier=max_frontier,
            minhash_dim=max(4, minhash_dim),
        )
        allowed_partitions: Optional[Set[str]] = None
        bilateral_strength = 1.0
        if partnership_mode.lower() == "bilateral":
            scope = select_bilateral_scope(anchors, local_sets, adjacency, partitions)
            if scope is None or scope["strength"] < min_bilateral_strength:
                continue
            allowed_partitions = set(scope["partitions"])
            bilateral_strength = float(scope["strength"])
            anchors = [node for node in anchors if partitions.get(node) in allowed_partitions]
            anchors = limit_anchors_per_party(anchors, adjacency, partitions, max_anchors_per_party)
            if not anchors:
                continue
        candidates = gather_candidates(
            anchors=anchors,
            local_sets=local_sets,
            adjacency=adjacency,
            edge_lookup=edge_lookup,
            partitions=partitions,
            target_nodes=target_nodes,
            target_edges=target_edges,
            motif_type=motif_type,
            max_frontier=max_frontier,
            allowed_partitions=allowed_partitions,
        )
        if not candidates:
            continue

        scored = []
        for nodes, source in candidates:
            if allowed_partitions:
                nodes = [node for node in nodes if partitions.get(str(node)) in allowed_partitions]
            edges = select_csgm_edges(nodes, edge_lookup, partitions, anchors, target_edges, allowed_partitions=allowed_partitions)
            if not edges:
                continue
            score = csgm_score(
                nodes=nodes,
                edges=edges,
                anchors=anchors,
                partitions=partitions,
                adjacency=adjacency,
                target_nodes=target_nodes,
                target_edges=target_edges,
                motif_type=motif_type,
                source=source,
                bilateral_strength=bilateral_strength,
            )
            scored.append((score, query_id, tuple(sorted(set(nodes))), tuple(edges), source))
        if not scored:
            continue

        scored.sort(key=lambda row: (-row[0], row[2], row[3]))
        primary_rows.append(scored[0])
        extra_rows.extend(scored[1:])

    dedup: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], Tuple[float, str]] = {}
    for score, query_id, nodes, edges, source in primary_rows + extra_rows:
        key = (query_id, nodes, edges)
        if key not in dedup or score > dedup[key][0]:
            dedup[key] = (score, source)

    ranked_primary = [row for row in primary_rows if (row[1], row[2], row[3]) in dedup]
    ranked_extra = [
        (score, query_id, nodes, edges, source)
        for (query_id, nodes, edges), (score, source) in dedup.items()
        if (score, query_id, nodes, edges, source) not in ranked_primary
    ]
    ranked_primary.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    ranked_extra.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    selected = (ranked_primary + ranked_extra)[:max_predictions]
    runtime = elapsed(start)

    return [
        prediction(
            query_id=query_id,
            nodes=nodes,
            edges=edges,
            score=score,
            runtime=runtime,
            metadata={
                "method": "CSGM",
                "task": "collaborative_scatter_gather_mining",
                "source": source,
                "implementation_status": "adapted",
            },
        )
        for score, query_id, nodes, edges, source in selected
    ]


def bounded_target_size(query: Dict[str, Any], max_candidate_nodes: int) -> int:
    query_nodes = [node for node in query.get("nodes", []) if isinstance(node, str)]
    if not query_nodes:
        return min(4, max_candidate_nodes)
    return max(2, min(int(max_candidate_nodes), len(query_nodes)))


def scatter_anchor_sets(
    anchors: List[str],
    adjacency: Dict[str, Set[str]],
    edge_lookup: Dict[Tuple[str, str], List[str]],
    partitions: Dict[str, str],
    bfs_depth: int,
    max_frontier: int,
    minhash_dim: int,
) -> List[Dict[str, Any]]:
    local_sets = []
    anchor_set = set(anchors)
    shared_counter: Counter[str] = Counter()
    for anchor in anchors:
        for neighbor in adjacency.get(anchor, set()):
            shared_counter[neighbor] += 1

    for anchor in anchors:
        visited = {anchor}
        queue = deque([(anchor, 0)])
        frontier = []
        boundary_edges = set()
        while queue and len(frontier) < max_frontier:
            node, depth = queue.popleft()
            if depth >= bfs_depth:
                continue
            neighbors = sorted(
                adjacency.get(node, set()),
                key=lambda item: (
                    -neighbor_priority(item, anchor, adjacency, partitions, shared_counter, anchor_set),
                    item,
                ),
            )[:max_frontier]
            for neighbor in neighbors:
                for edge_id in edge_lookup.get((node, neighbor), []):
                    if is_cross_partition(node, neighbor, partitions) or node in anchor_set or neighbor in anchor_set:
                        boundary_edges.add(str(edge_id))
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                frontier.append(neighbor)
                queue.append((neighbor, depth + 1))
                if len(frontier) >= max_frontier:
                    break

        edge_sig = minhash_signature(boundary_edges, minhash_dim)
        local_sets.append(
            {
                "anchor": anchor,
                "nodes": visited,
                "frontier": frontier,
                "edges": boundary_edges,
                "signature": edge_sig,
                "partition": partitions.get(anchor, ""),
            }
        )
    return local_sets


def select_bilateral_scope(
    anchors: List[str],
    local_sets: List[Dict[str, Any]],
    adjacency: Dict[str, Set[str]],
    partitions: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    anchor_parts = [partitions.get(anchor, "") for anchor in anchors if partitions.get(anchor, "")]
    if not anchor_parts:
        return None
    primary = Counter(anchor_parts).most_common(1)[0][0]
    partner_scores: Counter[str] = Counter()

    for anchor in anchors:
        if partitions.get(anchor) != primary:
            partner_scores[partitions.get(anchor, "")] += 2.0
        for neighbor in adjacency.get(anchor, set()):
            part = partitions.get(neighbor, "")
            if part and part != primary:
                partner_scores[part] += 1.0

    for idx, left in enumerate(local_sets):
        for right in local_sets[idx + 1 :]:
            left_part = str(left.get("partition", ""))
            right_part = str(right.get("partition", ""))
            if not left_part or not right_part or left_part == right_part:
                continue
            similarity = signature_similarity(left.get("signature", []), right.get("signature", []))
            overlap = jaccard(set(left.get("frontier", [])), set(right.get("frontier", [])))
            if left_part == primary:
                partner_scores[right_part] += 2.0 * similarity + overlap
            elif right_part == primary:
                partner_scores[left_part] += 2.0 * similarity + overlap

    partner_scores.pop("", None)
    partner_scores.pop(primary, None)
    if not partner_scores:
        return None
    partner, strength = sorted(partner_scores.items(), key=lambda item: (-item[1], item[0]))[0]
    normalized = float(strength / max(1, len(anchors)))
    return {"partitions": {primary, partner}, "strength": normalized}


def limit_anchors_per_party(
    anchors: List[str],
    adjacency: Dict[str, Set[str]],
    partitions: Dict[str, str],
    max_anchors_per_party: int,
) -> List[str]:
    if max_anchors_per_party <= 0:
        return anchors
    by_partition: Dict[str, List[str]] = {}
    for anchor in anchors:
        by_partition.setdefault(partitions.get(anchor, ""), []).append(anchor)
    selected = []
    for part in sorted(by_partition):
        ranked = sorted(by_partition[part], key=lambda node: (-len(adjacency.get(node, set())), node))
        selected.extend(ranked[:max_anchors_per_party])
    return selected


def gather_candidates(
    anchors: List[str],
    local_sets: List[Dict[str, Any]],
    adjacency: Dict[str, Set[str]],
    edge_lookup: Dict[Tuple[str, str], List[str]],
    partitions: Dict[str, str],
    target_nodes: int,
    target_edges: int,
    motif_type: str,
    max_frontier: int,
    allowed_partitions: Optional[Set[str]] = None,
) -> List[Tuple[List[str], str]]:
    candidates: List[Tuple[List[str], str]] = []
    anchor_set = set(anchors)
    shared_counter: Counter[str] = Counter()
    for local_set in local_sets:
        shared_counter.update(str(node) for node in local_set.get("frontier", []))

    shared_nodes = sorted(
        (node for node, count in shared_counter.items() if node not in anchor_set and partition_allowed(node, partitions, allowed_partitions)),
        key=lambda node: (-shared_counter[node], -len(adjacency.get(node, set()) & anchor_set), node),
    )
    if shared_nodes:
        nodes = list(dict.fromkeys(anchors[: max(1, target_nodes - 1)] + shared_nodes[: max(1, target_nodes - len(anchors[: target_nodes]))]))
        candidates.append((fill_candidate(nodes, adjacency, partitions, anchor_set, target_nodes, motif_type, max_frontier), "shared_anchor_gather"))

    similar_pairs = []
    for idx, left in enumerate(local_sets):
        for right in local_sets[idx + 1 :]:
            if left.get("partition") and right.get("partition") and left.get("partition") == right.get("partition"):
                continue
            similarity = signature_similarity(left.get("signature", []), right.get("signature", []))
            node_overlap = jaccard(set(left.get("frontier", [])), set(right.get("frontier", [])))
            score = 0.65 * similarity + 0.35 * node_overlap
            similar_pairs.append((score, left, right))
    similar_pairs.sort(key=lambda item: -item[0])
    for score, left, right in similar_pairs[: max(2, min(8, len(similar_pairs)))]:
        nodes = list(anchor_set | set(left.get("frontier", [])[: max(1, target_nodes // 2)]) | set(right.get("frontier", [])[: max(1, target_nodes // 2)]))
        nodes = [node for node in nodes if partition_allowed(str(node), partitions, allowed_partitions)]
        candidates.append((fill_candidate(nodes, adjacency, partitions, anchor_set, target_nodes, motif_type, max_frontier), f"similar_set_gather:{score:.4f}"))

    for local_set in local_sets[: max(1, min(len(local_sets), 12))]:
        anchor = str(local_set["anchor"])
        ranked = sorted(
            [node for node in local_set.get("frontier", []) if partition_allowed(str(node), partitions, allowed_partitions)],
            key=lambda node: (-neighbor_priority(str(node), anchor, adjacency, partitions, shared_counter, anchor_set), str(node)),
        )
        nodes = [anchor] + [str(node) for node in ranked[: max(1, target_nodes - 1)]]
        candidates.append((fill_candidate(nodes, adjacency, partitions, anchor_set, target_nodes, motif_type, max_frontier), "single_institution_scatter"))

    return unique_candidates(candidates)


def fill_candidate(
    nodes: List[str],
    adjacency: Dict[str, Set[str]],
    partitions: Dict[str, str],
    anchor_set: Set[str],
    target_nodes: int,
    motif_type: str,
    max_frontier: int,
) -> List[str]:
    node_list = list(dict.fromkeys(str(node) for node in nodes))
    node_set = set(node_list)
    ranked_neighbors = []
    seeds = [node for node in node_list if node in anchor_set] or node_list[:1]
    for seed in seeds:
        ranked_neighbors.extend(
            sorted(
                adjacency.get(seed, set()),
                key=lambda node: (
                    -shape_priority(node, adjacency, set(node_list), motif_type),
                    -cross_partition_priority(seed, node, partitions),
                    node,
                ),
            )[:max_frontier]
        )
    for node in ranked_neighbors:
        if len(node_list) >= target_nodes:
            break
        if node not in node_set:
            node_set.add(node)
            node_list.append(node)
    return node_list[:target_nodes]


def select_csgm_edges(
    nodes: Iterable[str],
    edge_lookup: Dict[Tuple[str, str], List[str]],
    partitions: Dict[str, str],
    anchors: List[str],
    target_edges: int,
    allowed_partitions: Optional[Set[str]] = None,
) -> List[str]:
    node_list = list(dict.fromkeys(str(node) for node in nodes))
    node_set = set(node_list)
    anchor_set = set(anchors)
    rows = []
    seen_edges = set()
    for src in node_list:
        for dst in node_list:
            if src == dst or dst not in node_set:
                continue
            if not partition_allowed(src, partitions, allowed_partitions) or not partition_allowed(dst, partitions, allowed_partitions):
                continue
            for edge_id in edge_lookup.get((src, dst), []):
                edge_id = str(edge_id)
                if edge_id in seen_edges:
                    continue
                seen_edges.add(edge_id)
                score = 0.0
                score += 1.0 if src in anchor_set or dst in anchor_set else 0.0
                score += 0.8 if is_cross_partition(src, dst, partitions) else 0.0
                score += 0.1 * (len(edge_lookup.get((src, dst), [])))
                rows.append((score, edge_id, src, dst))
    rows.sort(key=lambda row: (-row[0], row[1]))
    limit = max(1, int(target_edges))
    return [edge_id for _, edge_id, _, _ in rows[:limit]]


def csgm_score(
    nodes: List[str],
    edges: List[str],
    anchors: List[str],
    partitions: Dict[str, str],
    adjacency: Dict[str, Set[str]],
    target_nodes: int,
    target_edges: int,
    motif_type: str,
    source: str,
    bilateral_strength: float = 1.0,
) -> float:
    node_set = set(nodes)
    edge_count = len(edges)
    edge_match = min(edge_count, target_edges) / max(edge_count, target_edges, 1)
    size_match = 1.0 - abs(len(node_set) - target_nodes) / max(len(node_set), target_nodes, 1)
    anchor_coverage = len(node_set & set(anchors)) / max(1, min(len(anchors), target_nodes))
    parts = {partitions.get(node) for node in node_set if partitions.get(node)}
    partition_span = 1.0 if len(parts) == 2 else 0.35 if len(parts) == 1 else 0.0
    cross_edges = sum(1 for src in node_set for dst in adjacency.get(src, set()) & node_set if src < dst and is_cross_partition(src, dst, partitions))
    cross_ratio = min(1.0, cross_edges / max(1, edge_count))
    degrees = [len(adjacency.get(node, set()) & node_set) for node in node_set]
    motif_shape = motif_shape_score(degrees, edge_count, len(node_set), motif_type)
    source_bonus = 0.05 if source.startswith("similar_set") else 0.08 if source == "shared_anchor_gather" else 0.0
    return (
        0.27 * edge_match
        + 0.18 * size_match
        + 0.17 * anchor_coverage
        + 0.16 * partition_span
        + 0.12 * cross_ratio
        + 0.10 * motif_shape
        + 0.05 * min(1.0, bilateral_strength)
        + source_bonus
    )


def motif_shape_score(degrees: List[int], edge_count: int, size: int, motif_type: str) -> float:
    if size <= 1:
        return 0.0
    max_degree = max(degrees) if degrees else 0
    density = edge_count / max(1.0, size * (size - 1) / 2.0)
    if "fan" in motif_type or "star" in motif_type:
        return max_degree / max(1, size - 1)
    if "cycle" in motif_type:
        return min(1.0, edge_count / max(1, size))
    if "dense" in motif_type:
        return min(1.0, density)
    return min(1.0, edge_count / max(1, size - 1))


def neighbor_priority(
    node: str,
    anchor: str,
    adjacency: Dict[str, Set[str]],
    partitions: Dict[str, str],
    shared_counter: Counter[str],
    anchor_set: Set[str],
) -> float:
    degree = len(adjacency.get(node, set()))
    shared = shared_counter.get(node, 0)
    anchor_links = len(adjacency.get(node, set()) & anchor_set)
    return (
        math.log1p(degree)
        + 1.25 * cross_partition_priority(anchor, node, partitions)
        + 0.85 * shared
        + 0.95 * anchor_links
    )


def shape_priority(node: str, adjacency: Dict[str, Set[str]], current: Set[str], motif_type: str) -> float:
    local_links = len(adjacency.get(node, set()) & current)
    degree = len(adjacency.get(node, set()))
    if "fan" in motif_type or "star" in motif_type:
        return 1.3 * local_links + 0.2 * math.log1p(degree)
    if "dense" in motif_type or "cycle" in motif_type:
        return 0.8 * local_links + 0.4 * math.log1p(degree)
    return 0.6 * local_links + 0.3 * math.log1p(degree)


def is_cross_partition(src: str, dst: str, partitions: Dict[str, str]) -> bool:
    left = partitions.get(src)
    right = partitions.get(dst)
    return bool(left and right and left != right)


def cross_partition_priority(src: str, dst: str, partitions: Dict[str, str]) -> float:
    return 1.0 if is_cross_partition(src, dst, partitions) else 0.0


def partition_allowed(node: str, partitions: Dict[str, str], allowed_partitions: Optional[Set[str]]) -> bool:
    if not allowed_partitions:
        return True
    return partitions.get(str(node)) in allowed_partitions


def minhash_signature(items: Iterable[str], dim: int) -> List[int]:
    item_list = [str(item) for item in items]
    if not item_list:
        return [0] * dim
    signature = []
    for idx in range(dim):
        values = [stable_hash(f"{idx}:{item}") for item in item_list]
        signature.append(min(values))
    return signature


def stable_hash(value: str) -> int:
    return int(hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest(), 16)


def signature_similarity(left: List[int], right: List[int]) -> float:
    if not left or not right:
        return 0.0
    width = min(len(left), len(right))
    return sum(1 for idx in range(width) if left[idx] == right[idx]) / width


def jaccard(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def unique_candidates(candidates: List[Tuple[List[str], str]]) -> List[Tuple[List[str], str]]:
    output = []
    seen = set()
    for nodes, source in candidates:
        key = tuple(sorted(set(nodes)))
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        output.append((list(key), source))
    return output
