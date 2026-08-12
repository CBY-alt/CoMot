from pathlib import Path
from typing import Any, Dict, List

from methods.base_method import BaseMethod
from methods.baselines.common import dot, elapsed, normalize, now, prediction, save_predictions


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
