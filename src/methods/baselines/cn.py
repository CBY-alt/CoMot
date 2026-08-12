from pathlib import Path
from typing import Any, Dict, List

from methods.base_method import BaseMethod
from methods.baselines.common import (
    anchor_link_prediction_predictions,
    common_neighbors_score,
    save_predictions,
)


class CommonNeighborsBaseline(BaseMethod):
    """Common Neighbors link-prediction baseline adapted to query anchors."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        return None

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        self.predictions = anchor_link_prediction_predictions(
            dataset=dataset,
            score_fn=common_neighbors_score,
            method_name="cn",
            max_predictions=int(cfg.get("max_predictions", 1000)),
            directed=bool(cfg.get("directed", False)),
            radius=int(cfg.get("candidate_radius", 2)),
            max_nodes_per_query=int(cfg.get("max_nodes_per_query", 240)),
        )
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)
