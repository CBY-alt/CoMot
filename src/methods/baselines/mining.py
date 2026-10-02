"""Baseline implementations grouped by method family."""

# fanmod
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


# gspan
from pathlib import Path
from typing import Any, Dict, List, Optional

from methods.base_method import BaseMethod


class GSpanBaseline(BaseMethod):
    """gSpan wrapper with standard-format export and simplified fallback mining."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        cfg = self.config.get("baseline", {})
        export_path = self.output_dir / cfg.get("gspan_input", "gspan_input.txt")
        export_gspan_input(dataset, export_path, max_edges=cfg.get("export_max_edges", 100000))
        return export_path

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        input_path = self.fit(dataset)
        if bool(cfg.get("use_external", False)):
            external = self._try_external(input_path)
            if external is not None:
                self.predictions = external
                save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
                return self.predictions
        self.predictions = simplified_gspan_predictions(
            dataset=dataset,
            max_predictions=int(cfg.get("max_predictions", 1000)),
            min_support=int(cfg.get("min_support", 2)),
            max_edges=int(cfg.get("fallback_max_edges", 50000)),
        )
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)

    def _try_external(self, input_path: Path) -> Optional[List[Dict[str, Any]]]:
        cfg = self.config.get("baseline", {})
        tool_path = cfg.get("tool_path")
        command_template = cfg.get("command_template")
        if not tool_path or not command_template:
            return None
        output_path = self.output_dir / "gspan_external_predictions.jsonl"
        command = command_template.format(
            tool_path=tool_path,
            input_path=input_path,
            output_path=output_path,
            min_support=int(cfg.get("min_support", 2)),
        )
        subprocess.run(shlex.split(command), cwd=self.project_root, check=True)
        if not output_path.exists():
            return None
        return load_prediction_jsonl(output_path)


def export_gspan_input(dataset: Any, path: Path, max_edges: Any = 100000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = dataset.load_edges()
    if max_edges not in (None, "all"):
        edges = edges.head(int(max_edges))
    nodes = sorted(set(edges["src"].astype(str)) | set(edges["dst"].astype(str)))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    with open(path, "w", encoding="utf-8") as f:
        f.write("t # 0\n")
        for node, idx in node_to_idx.items():
            f.write(f"v {idx} node\n")
        for row in edges[["src", "dst"]].itertuples(index=False):
            f.write(f"e {node_to_idx[str(row.src)]} {node_to_idx[str(row.dst)]} edge\n")
        f.write("t # -1\n")
    return path


def simplified_gspan_predictions(dataset: Any, max_predictions: int, min_support: int, max_edges: int) -> List[Dict[str, Any]]:
    start = now()
    rows = []
    edges = dataset.load_edges().head(max_edges)
    pair_counts: Dict[str, int] = {}
    for row in edges[["edge_id", "src", "dst"]].itertuples(index=False):
        key = "edge"
        pair_counts[key] = pair_counts.get(key, 0) + 1
        if pair_counts[key] >= min_support:
            rows.append((float(pair_counts[key]), str(row.edge_id), str(row.src), str(row.dst), key))
    rows.sort(key=lambda x: (-x[0], x[1]))
    runtime = elapsed(start)
    return [
        prediction(
            query_id=f"gspan_{pattern}",
            nodes=[src, dst],
            edges=[edge_id],
            score=score,
            runtime=runtime,
            metadata={
                "method": "gspan",
                "task": "frequent_subgraph_mining",
                "pattern": pattern,
                "implementation_status": "wrapper_with_simplified_fallback",
            },
        )
        for score, edge_id, src, dst, pattern in rows[:max_predictions]
    ]


def load_prediction_jsonl(path: Path) -> List[Dict[str, Any]]:
    predictions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions
