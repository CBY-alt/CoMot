from pathlib import Path
from typing import Any, Dict, List

from methods.base_method import BaseMethod
from methods.baselines.common import save_predictions, seal_style_predictions


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
