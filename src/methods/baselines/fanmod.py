import json
import shlex
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

from methods.base_method import BaseMethod
from methods.baselines.common import elapsed, now, prediction, save_predictions


class FanmodBaseline(BaseMethod):
    """FANMOD wrapper with a simplified frequent motif mining fallback."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        return None

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        if bool(cfg.get("use_external", False)):
            external = self._try_external(dataset)
            if external is not None:
                self.predictions = external
                save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
                return self.predictions
        self.predictions = simplified_fanmod_predictions(
            dataset=dataset,
            max_predictions=int(cfg.get("max_predictions", 1000)),
            max_seed_nodes=int(cfg.get("max_seed_nodes", 5000)),
            directed=bool(cfg.get("directed", False)),
        )
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)

    def _try_external(self, dataset: Any) -> Optional[List[Dict[str, Any]]]:
        cfg = self.config.get("baseline", {})
        tool_path = cfg.get("tool_path")
        command_template = cfg.get("command_template")
        if not tool_path or not command_template:
            return None
        input_path = self.output_dir / "fanmod_input.edgelist"
        output_path = self.output_dir / "fanmod_external_predictions.jsonl"
        export_fanmod_edgelist(dataset, input_path)
        command = command_template.format(
            tool_path=tool_path,
            input_path=input_path,
            output_path=output_path,
            motif_size=int(cfg.get("motif_size", 3)),
        )
        subprocess.run(shlex.split(command), cwd=self.project_root, check=True)
        if not output_path.exists():
            return None
        return load_prediction_jsonl(output_path)


def simplified_fanmod_predictions(dataset: Any, max_predictions: int, max_seed_nodes: int, directed: bool) -> List[Dict[str, Any]]:
    start = now()
    adjacency, edge_lookup = dataset.graph(directed=directed)
    rows = []
    for node in sorted(adjacency.keys())[:max_seed_nodes]:
        neighbors = sorted(adjacency.get(node, set()))
        if len(neighbors) >= 2:
            for left, right in combinations(neighbors[:25], 2):
                motif_nodes = [node, left, right]
                edges = []
                edges.extend(edge_lookup.get((node, left), [])[:1])
                edges.extend(edge_lookup.get((node, right), [])[:1])
                if right in adjacency.get(left, set()):
                    edges.extend(edge_lookup.get((left, right), [])[:1])
                    motif_type = "triangle"
                    score = 3.0
                else:
                    motif_type = "wedge"
                    score = 2.0
                rows.append((score, motif_type, motif_nodes, sorted(set(edges))))
        if len(neighbors) >= 3:
            motif_nodes = [node] + neighbors[:3]
            edges = []
            for neighbor in neighbors[:3]:
                edges.extend(edge_lookup.get((node, neighbor), [])[:1])
            rows.append((float(len(neighbors)), "star", motif_nodes, sorted(set(edges))))

    rows.sort(key=lambda x: (-x[0], x[1], x[2]))
    runtime = elapsed(start)
    return [
        prediction(
            query_id=f"fanmod_{motif_type}",
            nodes=nodes,
            edges=edges,
            score=score,
            runtime=runtime,
            metadata={
                "method": "fanmod",
                "task": "frequent_motif_mining",
                "motif_type": motif_type,
                "implementation_status": "simplified_fallback",
            },
        )
        for score, motif_type, nodes, edges in rows[:max_predictions]
    ]


def export_fanmod_edgelist(dataset: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset.load_edges()[["src", "dst"]].itertuples(index=False):
            f.write(f"{row.src} {row.dst}\n")
    return path


def load_prediction_jsonl(path: Path) -> List[Dict[str, Any]]:
    predictions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions
