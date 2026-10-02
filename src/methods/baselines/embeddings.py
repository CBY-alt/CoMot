"""Baseline implementations grouped by method family."""

# deepwalk
from pathlib import Path
from typing import Any, Dict, List

from methods.base_method import BaseMethod
from methods.baselines.common import (
    embedding_edge_predictions,
    random_walk_embeddings,
    save_predictions,
)


class DeepWalkBaseline(BaseMethod):
    """Simplified DeepWalk baseline using random-walk co-occurrence random indexing."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.embeddings = {}
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        cfg = self.config.get("baseline", {})
        adjacency, _ = dataset.graph(directed=bool(cfg.get("directed", False)))
        self.embeddings = random_walk_embeddings(
            adjacency=adjacency,
            dim=int(cfg.get("embedding_dim", 64)),
            walk_length=int(cfg.get("walk_length", 20)),
            walks_per_node=int(cfg.get("walks_per_node", 4)),
            window_size=int(cfg.get("window_size", 5)),
            seed=self.seed,
            p=1.0,
            q=1.0,
            max_nodes=cfg.get("embedding_max_nodes", 50000),
        )
        return self.embeddings

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        if not self.embeddings:
            self.fit(dataset)
        self.predictions = embedding_edge_predictions(
            dataset=dataset,
            embeddings=self.embeddings,
            method_name="deepwalk",
            max_predictions=int(cfg.get("max_predictions", 1000)),
        )
        for item in self.predictions:
            item["metadata"]["implementation_status"] = "simplified"
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)


# node2vec
from pathlib import Path
from typing import Any, Dict, List

from methods.base_method import BaseMethod


class Node2VecBaseline(BaseMethod):
    """Simplified node2vec baseline using biased random-walk co-occurrence random indexing."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.embeddings = {}
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        cfg = self.config.get("baseline", {})
        adjacency, _ = dataset.graph(directed=bool(cfg.get("directed", False)))
        self.embeddings = random_walk_embeddings(
            adjacency=adjacency,
            dim=int(cfg.get("embedding_dim", 64)),
            walk_length=int(cfg.get("walk_length", 20)),
            walks_per_node=int(cfg.get("walks_per_node", 4)),
            window_size=int(cfg.get("window_size", 5)),
            seed=self.seed,
            p=float(cfg.get("p", 1.0)),
            q=float(cfg.get("q", 1.0)),
            max_nodes=cfg.get("embedding_max_nodes", 50000),
        )
        return self.embeddings

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        if not self.embeddings:
            self.fit(dataset)
        self.predictions = embedding_edge_predictions(
            dataset=dataset,
            embeddings=self.embeddings,
            method_name="node2vec",
            max_predictions=int(cfg.get("max_predictions", 1000)),
        )
        for item in self.predictions:
            item["metadata"]["implementation_status"] = "simplified"
            item["metadata"]["p"] = float(cfg.get("p", 1.0))
            item["metadata"]["q"] = float(cfg.get("q", 1.0))
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)


# graphsage
from pathlib import Path
from typing import Any, Dict, List

from methods.base_method import BaseMethod
from methods.baselines.common import (
    deterministic_vector,
    dot,
    normalize,
    prediction,
    now,
    elapsed,
)


class GraphSAGEBaseline(BaseMethod):
    """Simplified unsupervised GraphSAGE-style mean aggregation baseline."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.embeddings = {}
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        cfg = self.config.get("baseline", {})
        dim = int(cfg.get("embedding_dim", 64))
        layers = int(cfg.get("layers", 2))
        max_nodes = cfg.get("embedding_max_nodes", 50000)
        adjacency, _ = dataset.graph(directed=bool(cfg.get("directed", False)))
        nodes = sorted(adjacency.keys())
        if max_nodes not in (None, "all"):
            nodes = nodes[:int(max_nodes)]
        node_set = set(nodes)
        embeddings = {node: normalize(deterministic_vector(node, dim, self.seed)) for node in nodes}
        for _ in range(layers):
            new_embeddings = {}
            for node in nodes:
                vec = list(embeddings[node])
                neigh = [n for n in adjacency.get(node, []) if n in node_set]
                if neigh:
                    for idx in range(dim):
                        vec[idx] += sum(embeddings[n][idx] for n in neigh) / len(neigh)
                new_embeddings[node] = normalize(vec)
            embeddings = new_embeddings
        self.embeddings = embeddings
        return self.embeddings

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        if not self.embeddings:
            self.fit(dataset)
        start = now()
        rows = []
        for row in dataset.load_edges()[["edge_id", "src", "dst"]].itertuples(index=False):
            src = str(row.src)
            dst = str(row.dst)
            if src in self.embeddings and dst in self.embeddings:
                rows.append((dot(self.embeddings[src], self.embeddings[dst]), str(row.edge_id), src, dst))
        rows.sort(key=lambda x: (-x[0], x[1]))
        runtime = elapsed(start)
        self.predictions = [
            prediction(
                query_id="graphsage_unsupervised_edge_recovery",
                nodes=[src, dst],
                edges=[edge_id],
                score=score,
                runtime=runtime,
                metadata={
                    "method": "graphsage",
                    "implementation_status": "simplified",
                    "training": "unsupervised mean aggregation with deterministic structural seed features",
                },
            )
            for score, edge_id, src, dst in rows[:int(cfg.get("max_predictions", 1000))]
        ]
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)


# seal
from pathlib import Path
from typing import Any, Dict, List

from methods.base_method import BaseMethod
from methods.baselines.common import seal_style_predictions


class SealBaseline(BaseMethod):
    """Simplified SEAL-style enclosing-subgraph link prediction baseline."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        return None

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        self.predictions = seal_style_predictions(
            dataset=dataset,
            method_name="seal",
            max_predictions=int(cfg.get("max_predictions", 1000)),
            directed=bool(cfg.get("directed", False)),
            debug=bool(cfg.get("debug", False)),
            candidate_edge_limit=cfg.get("candidate_edge_limit"),
        )
        for item in self.predictions:
            item["metadata"]["paper"] = "Link Prediction Based on Graph Neural Networks"
            item["metadata"]["implementation_note"] = "simplified enclosing-subgraph feature scorer; no GNN training"
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)


# subgnn
from pathlib import Path
from typing import Any, Dict, List

from methods.base_method import BaseMethod


class SubGNNBaseline(BaseMethod):
    """Simplified SubGNN-style subgraph representation baseline."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        return None

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        self.predictions = simplified_subgnn_predictions(
            dataset=dataset,
            max_predictions=int(cfg.get("max_predictions", 1000)),
            max_seed_nodes=int(cfg.get("max_seed_nodes", 5000)),
            radius=int(cfg.get("radius", 1)),
            directed=bool(cfg.get("directed", False)),
        )
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)


def simplified_subgnn_predictions(dataset: Any, max_predictions: int, max_seed_nodes: int, radius: int, directed: bool) -> List[Dict[str, Any]]:
    start = now()
    adjacency, edge_lookup = dataset.graph(directed=directed)
    rows = []
    for seed_node in sorted(adjacency.keys())[:max_seed_nodes]:
        nodes = ego_nodes(seed_node, adjacency, radius=radius)
        if len(nodes) < 2:
            continue
        edges = []
        internal_edges = 0
        for src in nodes:
            for dst in adjacency.get(src, set()):
                if dst in nodes:
                    internal_edges += 1
                    edges.extend(edge_lookup.get((src, dst), [])[:1])
        if not directed:
            internal_edges = internal_edges // 2
        density = internal_edges / max(1, len(nodes) * (len(nodes) - 1))
        embedding = normalize([float(len(nodes)), float(internal_edges), float(len(adjacency.get(seed_node, set()))), density])
        reference = normalize([4.0, 4.0, 3.0, 0.3])
        score = dot(embedding, reference)
        rows.append((score, seed_node, sorted(nodes), sorted(set(edges))))

    rows.sort(key=lambda x: (-x[0], x[1]))
    runtime = elapsed(start)
    return [
        prediction(
            query_id="subgnn_subgraph_recovery",
            nodes=nodes,
            edges=edges,
            score=score,
            runtime=runtime,
            metadata={
                "method": "subgnn",
                "task": "subgraph_representation_scoring",
                "seed_node": seed_node,
                "radius": radius,
                "implementation_status": "simplified_fallback",
            },
        )
        for score, seed_node, nodes, edges in rows[:max_predictions]
    ]


def ego_nodes(seed_node: str, adjacency: Dict[str, set], radius: int) -> set:
    visited = {seed_node}
    frontier = {seed_node}
    for _ in range(max(0, radius)):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(adjacency.get(node, set()))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return visited

