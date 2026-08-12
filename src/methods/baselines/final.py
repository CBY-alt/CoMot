from pathlib import Path
from typing import Any, Dict, List

from methods.base_method import BaseMethod
from methods.baselines.common import alignment_predictions, save_predictions


class FinalBaseline(BaseMethod):
    """Simplified FINAL-style attributed partition alignment baseline."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        return None

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        self.predictions = alignment_predictions(
            dataset=dataset,
            method_name="final",
            max_predictions=int(cfg.get("max_predictions", 1000)),
            seed=self.seed,
            dim=int(cfg.get("signature_dim", 16)),
            directed=bool(cfg.get("directed", False)),
            mode="final",
            max_nodes_per_partition=cfg.get("max_nodes_per_partition", 500),
        )
        for item in self.predictions:
            item["metadata"]["paper"] = "FINAL: Fast Attributed Network Alignment"
            item["metadata"]["implementation_note"] = "simplified structural/attribute signature alignment"
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)
