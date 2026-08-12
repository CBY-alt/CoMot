from pathlib import Path
from typing import Any, Dict, List

from methods.base_method import BaseMethod
from methods.baselines.common import (
    embedding_edge_predictions,
    random_walk_embeddings,
    save_predictions,
)


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
