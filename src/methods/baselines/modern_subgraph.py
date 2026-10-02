import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from methods.base_method import BaseMethod
from methods.baselines.common import elapsed, node_partition_map, now, prediction, save_predictions


class NeuGNBaseline(BaseMethod):
    """NeuGN-inspired neural-guided subgraph navigation baseline.

    This is an adapted baseline for the CoMot partial-observability protocol:
    it uses deterministic structural signatures to guide local search from
    method-facing anchors, then returns ranked candidate subgraphs.
    """

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        return None

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        self.predictions = query_guided_subgraph_predictions(
            dataset=dataset,
            method_name="NeuGN",
            max_predictions=int(cfg.get("max_predictions", 1000)),
            max_neighbors=int(cfg.get("max_neighbors", 24)),
            beam_width=int(cfg.get("beam_width", 8)),
            max_candidate_nodes=int(cfg.get("max_candidate_nodes", 8)),
            directed=bool(cfg.get("directed", False)),
            mode="neural_navigation",
        )
        for item in self.predictions:
            item["metadata"]["paper"] = "Neural Graph Navigation for Intelligent Subgraph Matching"
            item["metadata"]["implementation_note"] = "adapted neural-guided local navigation over observable evidence"
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)


class TPABBaseline(BaseMethod):
    """Topology-aware top-k subgraph matching baseline adapted to anchors."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        return None

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        self.predictions = query_guided_subgraph_predictions(
            dataset=dataset,
            method_name="TPAB",
            max_predictions=int(cfg.get("max_predictions", 1000)),
            max_neighbors=int(cfg.get("max_neighbors", 28)),
            beam_width=int(cfg.get("beam_width", 10)),
            max_candidate_nodes=int(cfg.get("max_candidate_nodes", 8)),
            directed=bool(cfg.get("directed", False)),
            mode="topology_aware_topk",
        )
        for item in self.predictions:
            item["metadata"]["paper"] = "Efficient Pruned Top-K Subgraph Matching with Topology-Aware Bounds"
            item["metadata"]["implementation_note"] = "adapted topology-aware bounded local top-k search"
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)


class ISONETBaseline(BaseMethod):
    """ISONET-inspired neural subgraph retrieval baseline."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        return None

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        self.predictions = query_guided_subgraph_predictions(
            dataset=dataset,
            method_name="ISONET",
            max_predictions=int(cfg.get("max_predictions", 1000)),
            max_neighbors=int(cfg.get("max_neighbors", 24)),
            beam_width=int(cfg.get("beam_width", 8)),
            max_candidate_nodes=int(cfg.get("max_candidate_nodes", 8)),
            directed=bool(cfg.get("directed", False)),
            mode="ISONET_edge_alignment",
        )
        for item in self.predictions:
            item["metadata"]["paper"] = "ISONET: Interpretable Neural Subgraph Matching for Graph Retrieval"
            item["metadata"]["implementation_note"] = "adapted edge-alignment subgraph retrieval scorer"
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)


def query_guided_subgraph_predictions(
    dataset: Any,
    method_name: str,
    max_predictions: int,
    max_neighbors: int,
    beam_width: int,
    max_candidate_nodes: int,
    directed: bool,
    mode: str,
) -> List[Dict[str, Any]]:
    start = now()
    adjacency, edge_lookup = dataset.graph(directed=directed)
    partitions = node_partition_map(dataset, max_nodes_per_partition=None)
    rows = []

    for query in dataset.load_query_motifs():
        query_id = str(query.get("query_id", f"{method_name}_query"))
        anchors = [str(node) for node in query.get("anchor_nodes", []) if str(node) in adjacency]
        if not anchors:
            continue
        target_nodes = bounded_target_size(query, max_candidate_nodes)
        target_edges = max(1, len(query.get("edges", [])))
        motif_type = str(query.get("motif_type", "")).lower()

        for anchor in anchors[: max(1, beam_width)]:
            for nodes in local_candidate_node_sets(
                anchor=anchor,
                adjacency=adjacency,
                partitions=partitions,
                target_nodes=target_nodes,
                target_edges=target_edges,
                motif_type=motif_type,
                max_neighbors=max_neighbors,
                beam_width=beam_width,
                mode=mode,
            ):
                edge_ids = edge_ids_for_node_set(nodes, edge_lookup)
                if not edge_ids:
                    continue
                score = score_candidate(
                    nodes=nodes,
                    edge_ids=edge_ids,
                    anchor=anchor,
                    adjacency=adjacency,
                    partitions=partitions,
                    target_nodes=target_nodes,
                    target_edges=target_edges,
                    motif_type=motif_type,
                    mode=mode,
                )
                rows.append((score, query_id, tuple(sorted(nodes)), tuple(edge_ids), anchor))

    dedup: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], Tuple[float, str]] = {}
    for score, query_id, nodes, edge_ids, anchor in rows:
        key = (query_id, nodes, edge_ids)
        if key not in dedup or score > dedup[key][0]:
            dedup[key] = (score, anchor)

    ranked = [(score, query_id, nodes, edge_ids, anchor) for (query_id, nodes, edge_ids), (score, anchor) in dedup.items()]
    ranked.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    runtime = elapsed(start)
    return [
        prediction(
            query_id=query_id,
            nodes=nodes,
            edges=edge_ids,
            score=score,
            runtime=runtime,
            metadata={
                "method": method_name,
                "task": "query_guided_subgraph_retrieval",
                "anchor": anchor,
                "implementation_status": "adapted",
            },
        )
        for score, query_id, nodes, edge_ids, anchor in ranked[:max_predictions]
    ]


def bounded_target_size(query: Dict[str, Any], max_candidate_nodes: int) -> int:
    query_nodes = [node for node in query.get("nodes", []) if isinstance(node, str)]
    if not query_nodes:
        return min(4, max_candidate_nodes)
    return max(2, min(int(max_candidate_nodes), len(query_nodes)))


def local_candidate_node_sets(
    anchor: str,
    adjacency: Dict[str, Set[str]],
    partitions: Dict[str, str],
    target_nodes: int,
    target_edges: int,
    motif_type: str,
    max_neighbors: int,
    beam_width: int,
    mode: str,
) -> Iterable[List[str]]:
    ranked = sorted(
        adjacency.get(anchor, set()),
        key=lambda node: (-neighbor_priority(node, anchor, adjacency, partitions), node),
    )[:max_neighbors]
    if not ranked:
        return
    if mode == "topology_aware_topk":
        yield from topology_bound_node_sets(anchor, ranked, adjacency, partitions, target_nodes, target_edges, motif_type, beam_width)
        return

    seen = set()
    windows = max(1, min(beam_width, len(ranked)))
    for offset in range(windows):
        nodes = [anchor] + ranked[offset : offset + max(1, target_nodes - 1)]
        if len(nodes) < target_nodes:
            nodes = [anchor] + ranked[: max(1, target_nodes - 1)]
        key = tuple(sorted(set(nodes)))
        if len(key) >= 2 and key not in seen:
            seen.add(key)
            yield list(key)

    for first in ranked[:beam_width]:
        second_hop = [
            node
            for node in sorted(
                adjacency.get(first, set()),
                key=lambda node: (-neighbor_priority(node, first, adjacency, partitions), node),
            )
            if node != anchor
        ][: max(1, target_nodes - 2)]
        nodes = [anchor, first] + second_hop
        key = tuple(sorted(set(nodes)))
        if len(key) >= 2 and key not in seen:
            seen.add(key)
            yield list(key)


def topology_bound_node_sets(
    anchor: str,
    ranked: List[str],
    adjacency: Dict[str, Set[str]],
    partitions: Dict[str, str],
    target_nodes: int,
    target_edges: int,
    motif_type: str,
    beam_width: int,
) -> Iterable[List[str]]:
    seen = set()
    local = [anchor] + ranked[: max(target_nodes + beam_width, target_nodes)]
    scored_neighbors = sorted(
        ranked,
        key=lambda node: (
            -topology_local_bound([anchor, node], adjacency, target_edges, motif_type),
            -neighbor_priority(node, anchor, adjacency, partitions),
            node,
        ),
    )
    windows = max(1, min(beam_width, len(scored_neighbors)))
    for offset in range(windows):
        nodes = [anchor] + scored_neighbors[offset : offset + max(1, target_nodes - 1)]
        if len(nodes) < target_nodes:
            nodes = [anchor] + scored_neighbors[: max(1, target_nodes - 1)]
        key = tuple(sorted(set(nodes)))
        if len(key) >= 2 and key not in seen:
            seen.add(key)
            yield list(key)

    for first in scored_neighbors[:beam_width]:
        candidates = [
            node
            for node in sorted(
                adjacency.get(first, set()) & set(local),
                key=lambda node: (
                    -topology_local_bound([anchor, first, node], adjacency, target_edges, motif_type),
                    -neighbor_priority(node, first, adjacency, partitions),
                    node,
                ),
            )
            if node != anchor
        ]
        nodes = [anchor, first] + candidates[: max(1, target_nodes - 2)]
        key = tuple(sorted(set(nodes)))
        if len(key) >= 2 and key not in seen:
            seen.add(key)
            yield list(key)

    for first in scored_neighbors[:beam_width]:
        second_hop = [
            node
            for node in sorted(
                adjacency.get(first, set()),
                key=lambda node: (
                    -topology_local_bound([anchor, first, node], adjacency, target_edges, motif_type),
                    -neighbor_priority(node, first, adjacency, partitions),
                    node,
                ),
            )
            if node != anchor
        ][: max(1, target_nodes - 2)]
        nodes = [anchor, first] + second_hop
        key = tuple(sorted(set(nodes)))
        if len(key) >= 2 and key not in seen:
            seen.add(key)
            yield list(key)


def topology_local_bound(nodes: List[str], adjacency: Dict[str, Set[str]], target_edges: int, motif_type: str) -> float:
    node_set = set(nodes)
    edge_upper = sum(1 for src in node_set for dst in adjacency.get(src, set()) if dst in node_set and src != dst) / 2.0
    edge_gap = abs(edge_upper - target_edges)
    density = edge_upper / max(1.0, len(node_set) * (len(node_set) - 1) / 2.0)
    if "star" in motif_type or "fan" in motif_type:
        hub = max((len(adjacency.get(node, set()) & node_set) for node in node_set), default=0)
        return 0.60 * hub + 0.30 * density - 0.10 * edge_gap
    return 0.70 * min(edge_upper, target_edges) + 0.20 * density - 0.10 * edge_gap


def neighbor_priority(node: str, anchor: str, adjacency: Dict[str, Set[str]], partitions: Dict[str, str]) -> float:
    degree = len(adjacency.get(node, set()))
    two_hop = sum(len(adjacency.get(neighbor, set())) for neighbor in list(adjacency.get(node, set()))[:16])
    cross = 1.0 if partitions.get(node) and partitions.get(anchor) and partitions.get(node) != partitions.get(anchor) else 0.0
    return math.log1p(degree) + 0.10 * math.log1p(two_hop) + 0.75 * cross


def edge_ids_for_node_set(nodes: Iterable[str], edge_lookup: Dict[Tuple[str, str], List[str]]) -> List[str]:
    node_list = list(dict.fromkeys(str(node) for node in nodes))
    edge_ids = set()
    for src in node_list:
        for dst in node_list:
            if src == dst:
                continue
            edge_ids.update(str(edge_id) for edge_id in edge_lookup.get((src, dst), []))
    return sorted(edge_ids)


def score_candidate(
    nodes: List[str],
    edge_ids: List[str],
    anchor: str,
    adjacency: Dict[str, Set[str]],
    partitions: Dict[str, str],
    target_nodes: int,
    target_edges: int,
    motif_type: str,
    mode: str,
) -> float:
    size = max(1, len(nodes))
    edge_count = len(edge_ids)
    density = edge_count / max(1.0, size * (size - 1))
    if edge_count >= target_edges:
        edge_match = target_edges / max(edge_count, target_edges)
    else:
        edge_match = edge_count / max(1, target_edges)
    size_match = 1.0 - abs(size - target_nodes) / max(size, target_nodes, 1)
    parts = {partitions.get(node) for node in nodes if partitions.get(node)}
    partition_span = min(1.0, len(parts) / 2.0)
    degrees = [len(adjacency.get(node, set()) & set(nodes)) for node in nodes]
    max_local_degree = max(degrees) if degrees else 0
    fan_score = max_local_degree / max(1, size - 1)
    chain_score = min(1.0, edge_count / max(1, size - 1))
    cycle_score = 1.0 if edge_count >= size else edge_count / max(1, size)
    if "fan" in motif_type or "star" in motif_type:
        motif_shape = fan_score
    elif "cycle" in motif_type:
        motif_shape = cycle_score
    elif "chain" in motif_type or "bridge" in motif_type:
        motif_shape = chain_score
    else:
        motif_shape = 0.5 * density + 0.5 * chain_score

    if mode == "neural_navigation":
        anchor_degree = len(adjacency.get(anchor, set()))
        navigation = math.log1p(anchor_degree) / 10.0 + 0.25 * motif_shape
        return 0.34 * edge_match + 0.22 * size_match + 0.22 * partition_span + 0.22 * navigation
    if mode == "topology_aware_topk":
        upper_bound = 0.55 * edge_match + 0.25 * size_match + 0.20 * motif_shape
        return upper_bound + 0.18 * partition_span + 0.05 * density
    if mode == "ISONET_edge_alignment":
        edge_alignment = min(edge_count, target_edges) / max(edge_count, target_edges, 1)
        degree_balance = 1.0 / (1.0 + np_std(degrees))
        return 0.42 * edge_alignment + 0.22 * degree_balance + 0.20 * motif_shape + 0.16 * partition_span
    return edge_match + size_match + partition_span


def np_std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
