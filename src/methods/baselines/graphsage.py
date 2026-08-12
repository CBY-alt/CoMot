from pathlib import Path
from typing import Any, Dict, List

from methods.base_method import BaseMethod
from methods.baselines.common import (
    deterministic_vector,
    dot,
    normalize,
    save_predictions,
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
