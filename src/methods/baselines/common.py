import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from data.query_view import load_method_queries


Prediction = Dict[str, Any]


class StandardGraphDataset:
    """Lightweight reader for CoMot standard processed datasets."""

    def __init__(self, dataset_name: str, project_root: Path, config: Dict[str, Any]):
        self.dataset_name = dataset_name
        self.project_root = Path(project_root)
        self.config = config
        data_cfg = config.get("data", {})
        dataset_dir = data_cfg.get("dataset_dir", f"data/processed/{dataset_name}")
        self.dataset_dir = self._resolve(dataset_dir)
        self.method_queries_path = self.dataset_dir / "method_queries.json"
        self.full_query_motifs_path = self.dataset_dir / "query_motifs.json"
        self.max_edges = data_cfg.get("max_edges")
        self.max_nodes = data_cfg.get("max_nodes")

        self.nodes_df: Optional[pd.DataFrame] = None
        self.edges_df: Optional[pd.DataFrame] = None
        self.ground_truth: Optional[List[Dict[str, Any]]] = None
        self.query_motifs: Optional[List[Dict[str, Any]]] = None
        self.partitions: Optional[List[Dict[str, Any]]] = None

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    def load_nodes(self) -> pd.DataFrame:
        if self.nodes_df is None:
            df = pd.read_csv(self.dataset_dir / "nodes.csv", dtype=str)
            if self.max_nodes not in (None, "all"):
                df = df.head(int(self.max_nodes)).copy()
            self.nodes_df = df
        return self.nodes_df

    def load_edges(self) -> pd.DataFrame:
        if self.edges_df is None:
            if self.max_edges in (None, "all"):
                df = pd.read_csv(self.dataset_dir / "edges.csv", dtype=str)
            else:
                df = pd.read_csv(self.dataset_dir / "edges.csv", dtype=str, nrows=int(self.max_edges))
            node_ids = set(self.load_nodes()["node_id"].astype(str))
            df = df[df["src"].astype(str).isin(node_ids) & df["dst"].astype(str).isin(node_ids)].copy()
            self.edges_df = df
        return self.edges_df

    def load_ground_truth(self) -> List[Dict[str, Any]]:
        if self.ground_truth is None:
            with open(self.dataset_dir / "ground_truth.json", "r", encoding="utf-8") as f:
                self.ground_truth = json.load(f)
        return self.ground_truth

    def load_query_motifs(self) -> List[Dict[str, Any]]:
        if self.query_motifs is None:
            self.query_motifs = load_method_queries(self.dataset_dir)
        return self.query_motifs

    def load_partitions(self) -> List[Dict[str, Any]]:
        if self.partitions is None:
            with open(self.dataset_dir / "partitions.json", "r", encoding="utf-8") as f:
                self.partitions = json.load(f)
        return self.partitions

    def graph(self, directed: bool = False):
        edges = self.load_edges()
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        edge_lookup: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for row in edges[["edge_id", "src", "dst"]].itertuples(index=False):
            src = str(row.src)
            dst = str(row.dst)
            edge_id = str(row.edge_id)
            adjacency[src].add(dst)
            adjacency.setdefault(dst, set())
            edge_lookup[(src, dst)].append(edge_id)
            if not directed:
                adjacency[dst].add(src)
                edge_lookup[(dst, src)].append(edge_id)
        return adjacency, edge_lookup


def now() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return float(time.perf_counter() - start)


def prediction(query_id: str, nodes: Iterable[str], edges: Iterable[str], score: float, runtime: float, metadata: Dict[str, Any]) -> Prediction:
    return {
        "query_id": str(query_id),
        "predicted_nodes": [str(x) for x in nodes],
        "predicted_edges": [str(x) for x in edges],
        "score": float(score),
        "runtime": float(runtime),
        "metadata": metadata,
    }


def save_predictions(predictions: List[Prediction], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item) + "\n")
    return path


def load_json_or_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text) or {}
    return json.loads(text)


def top_existing_edge_predictions(
    dataset: StandardGraphDataset,
    score_fn,
    method_name: str,
    max_predictions: int,
    directed: bool = False,
) -> List[Prediction]:
    start = now()
    adjacency, _ = dataset.graph(directed=directed)
    rows = []
    for row in dataset.load_edges()[["edge_id", "src", "dst"]].itertuples(index=False):
        src = str(row.src)
        dst = str(row.dst)
        score = score_fn(src, dst, adjacency)
        if score > 0:
            rows.append((score, str(row.edge_id), src, dst))
    rows.sort(key=lambda x: (-x[0], x[1]))
    runtime = elapsed(start)
    return [
        prediction(
            query_id=f"{method_name}_edge_recovery",
            nodes=[src, dst],
            edges=[edge_id],
            score=score,
            runtime=runtime,
            metadata={"method": method_name, "task": "existing_edge_ranking"},
        )
        for score, edge_id, src, dst in rows[:max_predictions]
    ]


def anchor_or_edge_predictions(
    dataset: StandardGraphDataset,
    score_fn,
    method_name: str,
    max_predictions: int,
    directed: bool = False,
) -> List[Prediction]:
    anchored = anchor_predictions(dataset, score_fn, method_name, max_predictions, directed=directed)
    if anchored:
        return anchored
    return top_existing_edge_predictions(dataset, score_fn, method_name, max_predictions, directed=directed)


def anchor_link_prediction_predictions(
    dataset: StandardGraphDataset,
    score_fn,
    method_name: str,
    max_predictions: int,
    directed: bool = False,
    radius: int = 2,
    max_nodes_per_query: int = 240,
) -> List[Prediction]:
    """Rank candidate node pairs near query anchors by a Liben-Nowell-style score.

    The original link-prediction setting ranks node pairs by proximity. In this
    recovery protocol we keep that pair-ranking semantics, but restrict the
    candidate universe to a small anchor-centered neighborhood so the baseline
    remains query-facing and tractable on large graphs.
    """
    start = now()
    adjacency, edge_lookup = dataset.graph(directed=directed)
    rows = []
    seen = set()
    for query_index, query in enumerate(dataset.load_query_motifs()):
        anchors = [str(node) for node in query.get("anchor_nodes", []) if str(node) in adjacency]
        if not anchors:
            continue
        candidate_nodes = anchor_neighborhood_nodes(adjacency, anchors, radius=radius, max_nodes=max_nodes_per_query)
        for src, dst, distance_hint in candidate_node_pairs(candidate_nodes, directed=directed):
            edge_ids = edge_lookup.get((src, dst), [])
            if not edge_ids:
                continue
            score = score_fn(src, dst, adjacency)
            if score <= 0:
                continue
            for edge_id in edge_ids:
                key = (str(query.get("query_id", method_name)), str(edge_id))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    (
                        score,
                        distance_hint,
                        query_index,
                        str(edge_id),
                        src,
                        dst,
                        str(query.get("query_id", method_name)),
                    )
                )

    rows.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    runtime = elapsed(start)
    return [
        prediction(
            query_id=query_id,
            nodes=[src, dst],
            edges=[edge_id],
            score=score,
            runtime=runtime,
            metadata={
                "method": method_name,
                "task": "anchor_constrained_link_prediction",
                "candidate_radius": radius,
                "candidate_distance_hint": distance_hint,
            },
        )
        for score, distance_hint, _, edge_id, src, dst, query_id in rows[:max_predictions]
    ]


def anchor_neighborhood_nodes(
    adjacency: Dict[str, Set[str]],
    anchors: List[str],
    radius: int,
    max_nodes: int,
) -> List[Tuple[str, int]]:
    distances: Dict[str, int] = {}
    frontier = [(anchor, 0) for anchor in anchors]
    for anchor in anchors:
        distances.setdefault(anchor, 0)
    idx = 0
    while idx < len(frontier):
        node, dist = frontier[idx]
        idx += 1
        if dist >= radius:
            continue
        for neighbor in sorted(adjacency.get(node, set())):
            next_dist = dist + 1
            if neighbor in distances and distances[neighbor] <= next_dist:
                continue
            distances[neighbor] = next_dist
            frontier.append((neighbor, next_dist))
            if len(distances) >= max_nodes:
                break
        if len(distances) >= max_nodes:
            break
    return sorted(distances.items(), key=lambda item: (item[1], item[0]))


def candidate_node_pairs(nodes_with_distance: List[Tuple[str, int]], directed: bool) -> Iterable[Tuple[str, str, int]]:
    nodes = [node for node, _ in nodes_with_distance]
    distances = {node: distance for node, distance in nodes_with_distance}
    if directed:
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                yield src, dst, distances.get(src, 0) + distances.get(dst, 0)
    else:
        for idx, src in enumerate(nodes):
            for dst in nodes[idx + 1:]:
                yield src, dst, distances.get(src, 0) + distances.get(dst, 0)


def anchor_predictions(
    dataset: StandardGraphDataset,
    score_fn,
    method_name: str,
    max_predictions: int,
    directed: bool = False,
) -> List[Prediction]:
    start = now()
    adjacency, edge_lookup = dataset.graph(directed=directed)
    rows = []
    for query in dataset.load_query_motifs():
        anchors = [str(node) for node in query.get("anchor_nodes", []) if str(node) in adjacency]
        if not anchors:
            continue
        seen_edges = set()
        for anchor in anchors:
            for candidate in adjacency.get(anchor, set()):
                score = score_fn(anchor, candidate, adjacency)
                edge_ids = edge_lookup.get((anchor, candidate), [])
                for edge_id in edge_ids:
                    if edge_id in seen_edges:
                        continue
                    seen_edges.add(edge_id)
                    rows.append((score, str(edge_id), anchor, candidate, str(query.get("query_id", method_name))))

    rows.sort(key=lambda x: (-x[0], x[1]))
    runtime = elapsed(start)
    return [
        prediction(
            query_id=query_id,
            nodes=[anchor, candidate],
            edges=[edge_id],
            score=score,
            runtime=runtime,
            metadata={"method": method_name, "task": "anchor_based_link_prediction"},
        )
        for score, edge_id, anchor, candidate, query_id in rows[:max_predictions]
    ]


def common_neighbors_score(src: str, dst: str, adjacency: Dict[str, Set[str]]) -> float:
    return float(len(adjacency.get(src, set()) & adjacency.get(dst, set())))


def adamic_adar_score(src: str, dst: str, adjacency: Dict[str, Set[str]]) -> float:
    total = 0.0
    for node in adjacency.get(src, set()) & adjacency.get(dst, set()):
        deg = len(adjacency.get(node, set()))
        if deg > 1:
            total += 1.0 / math.log(deg)
    return float(total)


def deterministic_vector(key: str, dim: int, seed: int) -> List[float]:
    rng = random.Random(f"{seed}:{key}")
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0:
        return vec
    return [x / norm for x in vec]


def dot(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def add_inplace(a: List[float], b: List[float], weight: float = 1.0) -> None:
    for i, value in enumerate(b):
        a[i] += weight * value


def random_walk_embeddings(
    adjacency: Dict[str, Set[str]],
    dim: int,
    walk_length: int,
    walks_per_node: int,
    window_size: int,
    seed: int,
    p: float = 1.0,
    q: float = 1.0,
    max_nodes: Optional[int] = None,
) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    nodes = sorted(adjacency.keys())
    if max_nodes not in (None, "all"):
        nodes = nodes[:int(max_nodes)]
    node_set = set(nodes)
    embeddings = {node: [0.0] * dim for node in nodes}

    for _ in range(walks_per_node):
        order = list(nodes)
        rng.shuffle(order)
        for start_node in order:
            walk = biased_walk(start_node, adjacency, walk_length, rng, p=p, q=q, allowed=node_set)
            for idx, node in enumerate(walk):
                if node not in embeddings:
                    continue
                lo = max(0, idx - window_size)
                hi = min(len(walk), idx + window_size + 1)
                for ctx in walk[lo:idx] + walk[idx + 1:hi]:
                    add_inplace(embeddings[node], deterministic_vector(ctx, dim, seed), 1.0)

    return {node: normalize(vec) for node, vec in embeddings.items()}


def biased_walk(start_node: str, adjacency: Dict[str, Set[str]], walk_length: int, rng: random.Random, p: float, q: float, allowed: Set[str]) -> List[str]:
    walk = [start_node]
    prev = None
    current = start_node
    for _ in range(max(0, walk_length - 1)):
        candidates = [n for n in adjacency.get(current, set()) if n in allowed]
        if not candidates:
            break
        if prev is None:
            nxt = rng.choice(sorted(candidates))
        else:
            weighted = []
            for cand in candidates:
                if cand == prev:
                    weight = 1.0 / max(p, 1e-8)
                elif cand in adjacency.get(prev, set()):
                    weight = 1.0
                else:
                    weight = 1.0 / max(q, 1e-8)
                weighted.append((cand, weight))
            nxt = weighted_choice(weighted, rng)
        walk.append(nxt)
        prev, current = current, nxt
    return walk


def weighted_choice(items: List[Tuple[str, float]], rng: random.Random) -> str:
    total = sum(weight for _, weight in items)
    pick = rng.random() * total
    acc = 0.0
    for item, weight in items:
        acc += weight
        if acc >= pick:
            return item
    return items[-1][0]


def embedding_edge_predictions(dataset: StandardGraphDataset, embeddings: Dict[str, List[float]], method_name: str, max_predictions: int) -> List[Prediction]:
    start = now()
    rows = []
    for row in dataset.load_edges()[["edge_id", "src", "dst"]].itertuples(index=False):
        src = str(row.src)
        dst = str(row.dst)
        if src not in embeddings or dst not in embeddings:
            continue
        rows.append((dot(embeddings[src], embeddings[dst]), str(row.edge_id), src, dst))
    rows.sort(key=lambda x: (-x[0], x[1]))
    runtime = elapsed(start)
    return [
        prediction(
            query_id=f"{method_name}_embedding_edge_recovery",
            nodes=[src, dst],
            edges=[edge_id],
            score=score,
            runtime=runtime,
            metadata={"method": method_name, "task": "embedding_existing_edge_ranking"},
        )
        for score, edge_id, src, dst in rows[:max_predictions]
    ]


def node_partition_map(dataset: StandardGraphDataset, max_nodes_per_partition: Optional[int] = None) -> Dict[str, str]:
    loaded_nodes = set(dataset.load_nodes()["node_id"].astype(str))
    mapping: Dict[str, str] = {}
    for partition in dataset.load_partitions():
        partition_id = str(partition.get("partition_id", ""))
        count = 0
        for node in partition.get("visible_nodes", []):
            node = str(node)
            if node in loaded_nodes:
                mapping[node] = partition_id
                count += 1
                if max_nodes_per_partition not in (None, "all") and count >= int(max_nodes_per_partition):
                    break
    return mapping


def node_labels(dataset: StandardGraphDataset) -> Dict[str, str]:
    nodes = dataset.load_nodes()
    if "label" not in nodes.columns:
        return {}
    return {str(row.node_id): str(row.label) for row in nodes[["node_id", "label"]].itertuples(index=False)}


def structural_signature(
    node: str,
    adjacency: Dict[str, Set[str]],
    labels: Dict[str, str],
    dim: int,
    seed: int,
    mode: str,
) -> List[float]:
    neighbors = adjacency.get(node, set())
    degree = len(neighbors)
    neighbor_degrees = [len(adjacency.get(neighbor, set())) for neighbor in neighbors]
    mean_neighbor_degree = sum(neighbor_degrees) / len(neighbor_degrees) if neighbor_degrees else 0.0
    max_neighbor_degree = max(neighbor_degrees) if neighbor_degrees else 0.0
    two_hop = set()
    for neighbor in neighbors:
        two_hop.update(adjacency.get(neighbor, set()))
    two_hop.discard(node)

    base = [
        math.log1p(degree),
        math.log1p(mean_neighbor_degree),
        math.log1p(max_neighbor_degree),
        math.log1p(len(two_hop)),
    ]
    if mode == "final":
        label_vec = deterministic_vector(labels.get(node, "unknown"), max(1, dim - len(base)), seed)
        vec = base + label_vec
    elif mode == "regal":
        bins = [0, 0, 0, 0]
        for value in neighbor_degrees:
            if value <= 1:
                bins[0] += 1
            elif value <= 4:
                bins[1] += 1
            elif value <= 16:
                bins[2] += 1
            else:
                bins[3] += 1
        vec = base + [math.log1p(x) for x in bins]
    elif mode == "bright":
        bridge_degree = sum(1 for neighbor in neighbors if labels.get(neighbor) != labels.get(node))
        vec = base + [math.log1p(bridge_degree)]
    elif mode == "HLOT":
        local_edges = 0
        wedges = 0
        for neighbor in neighbors:
            shared = adjacency.get(neighbor, set()) & neighbors
            local_edges += len(shared)
            wedges += len(adjacency.get(neighbor, set()) - {node})
        local_edges = local_edges / 2.0
        clustering = (2.0 * local_edges / (degree * (degree - 1))) if degree > 1 else 0.0
        label_bridge = sum(1 for neighbor in neighbors if labels.get(neighbor) != labels.get(node))
        vec = base + [
            math.log1p(local_edges),
            math.log1p(wedges),
            clustering,
            math.log1p(label_bridge),
        ]
    else:
        vec = base

    if len(vec) < dim:
        vec.extend([0.0] * (dim - len(vec)))
    return normalize(vec[:dim])


def alignment_predictions(
    dataset: StandardGraphDataset,
    method_name: str,
    max_predictions: int,
    seed: int,
    dim: int = 16,
    directed: bool = False,
    mode: str = "final",
    max_nodes_per_partition: Optional[int] = 2000,
) -> List[Prediction]:
    start = now()
    adjacency, edge_lookup = dataset.graph(directed=directed)
    partitions = node_partition_map(dataset, max_nodes_per_partition=max_nodes_per_partition)
    labels = node_labels(dataset)
    nodes_by_partition: Dict[str, List[str]] = defaultdict(list)
    for node in sorted(adjacency.keys()):
        partition_id = partitions.get(node)
        if partition_id:
            nodes_by_partition[partition_id].append(node)

    signatures = {
        node: structural_signature(node, adjacency, labels, dim=dim, seed=seed, mode=mode)
        for nodes in nodes_by_partition.values()
        for node in nodes
    }

    rows = []
    partition_ids = sorted(nodes_by_partition)
    for idx, left_pid in enumerate(partition_ids):
        for right_pid in partition_ids[idx + 1:]:
            for left in nodes_by_partition[left_pid]:
                best = None
                left_vec = signatures[left]
                for right in nodes_by_partition[right_pid]:
                    score = dot(left_vec, signatures[right])
                    if best is None or score > best[0] or (score == best[0] and right < best[1]):
                        best = (score, right)
                if best is None:
                    continue
                score, right = best
                edge_ids = edge_lookup.get((left, right), [])
                rows.append((score, left, right, left_pid, right_pid, edge_ids))

    rows.sort(key=lambda x: (-x[0], x[1], x[2]))
    runtime = elapsed(start)
    return [
        prediction(
            query_id=f"{method_name}_partition_alignment",
            nodes=[left, right],
            edges=edge_ids[:1],
            score=score,
            runtime=runtime,
            metadata={
                "method": method_name,
                "task": "partition_node_alignment",
                "left_partition": left_pid,
                "right_partition": right_pid,
                "implementation_status": "simplified",
            },
        )
        for score, left, right, left_pid, right_pid, edge_ids in rows[:max_predictions]
    ]


def seal_style_predictions(
    dataset: StandardGraphDataset,
    method_name: str,
    max_predictions: int,
    directed: bool = False,
    debug: bool = False,
    candidate_edge_limit: Optional[int] = None,
) -> List[Prediction]:
    start = now()
    adjacency, _ = dataset.graph(directed=directed)
    rows = []
    edge_iter = dataset.load_edges()[["edge_id", "src", "dst"]].itertuples(index=False)
    limit = int(candidate_edge_limit) if candidate_edge_limit not in (None, "all") else None
    if debug and limit is None:
        limit = 5000
    for idx, row in enumerate(edge_iter):
        if limit is not None and idx >= limit:
            break
        src = str(row.src)
        dst = str(row.dst)
        src_neighbors = adjacency.get(src, set())
        dst_neighbors = adjacency.get(dst, set())
        common = src_neighbors & dst_neighbors
        enclosing_nodes = set(common)
        enclosing_nodes.update([src, dst])
        enclosing_edges = 0
        for node in enclosing_nodes:
            enclosing_edges += sum(1 for neighbor in adjacency.get(node, set()) if neighbor in enclosing_nodes)
        if not directed:
            enclosing_edges = enclosing_edges // 2
        score = (
            2.0 * math.log1p(len(common))
            + 0.5 * math.log1p(enclosing_edges)
            + 0.1 * math.log1p(len(src_neighbors) + len(dst_neighbors))
        )
        rows.append((score, str(row.edge_id), src, dst, len(common), enclosing_edges, len(enclosing_nodes)))

    rows.sort(key=lambda x: (-x[0], x[1]))
    runtime = elapsed(start)
    return [
        prediction(
            query_id=f"{method_name}_link_prediction",
            nodes=[src, dst],
            edges=[edge_id],
            score=score,
            runtime=runtime,
            metadata={
                "method": method_name,
                "task": "enclosing_subgraph_link_prediction",
                "implementation_status": "simplified",
                "num_common_neighbors": num_common,
                "enclosing_edges": enclosing_edges,
                "enclosing_nodes": enclosing_nodes,
                "debug": debug,
            },
        )
        for score, edge_id, src, dst, num_common, enclosing_edges, enclosing_nodes in rows[:max_predictions]
    ]
