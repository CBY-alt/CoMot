import json
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

from methods.base_method import BaseMethod
from methods.baselines.common import elapsed, now, prediction, save_predictions
from methods.baselines.vf2 import build_query_graph, edge_ids_for_nodes


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("TurboISO fallback timeout")


class TurboISOBaseline(BaseMethod):
    """TurboISO wrapper with a NetworkX subgraph matching fallback."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.graph = None
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        cfg = self.config.get("baseline", {})
        graph = nx.DiGraph() if bool(cfg.get("directed", False)) else nx.Graph()
        for row in dataset.load_edges()[["edge_id", "src", "dst"]].itertuples(index=False):
            graph.add_edge(str(row.src), str(row.dst), edge_id=str(row.edge_id))
        self.graph = graph
        return graph

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        if bool(cfg.get("use_external", False)):
            external = self._try_external(dataset)
            if external is not None:
                self.predictions = external
                save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
                return self.predictions
        self.predictions = self._networkx_fallback(dataset)
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)

    def _try_external(self, dataset: Any) -> Any:
        cfg = self.config.get("baseline", {})
        tool_path = cfg.get("tool_path")
        command_template = cfg.get("command_template")
        if not tool_path or not command_template:
            return None
        input_dir = self.output_dir / "turboiso_input"
        input_dir.mkdir(parents=True, exist_ok=True)
        graph_path = input_dir / "graph.edgelist"
        query_path = input_dir / "query_motifs.json"
        output_path = self.output_dir / "turboiso_external_predictions.jsonl"
        export_edgelist(dataset, graph_path)
        query_path.write_text(json.dumps(dataset.load_query_motifs(), indent=2), encoding="utf-8")
        command = command_template.format(
            tool_path=tool_path,
            graph_path=graph_path,
            query_path=query_path,
            output_path=output_path,
            timeout=int(cfg.get("timeout", 30)),
        )
        subprocess.run(shlex.split(command), cwd=self.project_root, check=True)
        if not output_path.exists():
            return None
        predictions = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))
        return predictions

    def _networkx_fallback(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        if self.graph is None:
            self.fit(dataset)
        max_predictions = int(cfg.get("max_predictions", 100))
        timeout = int(cfg.get("timeout", 30))
        predictions = []
        start = now()
        for query in dataset.load_query_motifs():
            if len(predictions) >= max_predictions:
                break
            query_graph = build_query_graph(query, directed=isinstance(self.graph, nx.DiGraph))
            if query_graph.number_of_nodes() == 0 or query_graph.number_of_edges() == 0:
                continue
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)
                matcher = (
                    nx.algorithms.isomorphism.DiGraphMatcher(self.graph, query_graph)
                    if isinstance(self.graph, nx.DiGraph)
                    else nx.algorithms.isomorphism.GraphMatcher(self.graph, query_graph)
                )
                for mapping in matcher.subgraph_isomorphisms_iter():
                    nodes = list(mapping.keys())
                    predictions.append(
                        prediction(
                            query_id=query.get("query_id", "turboiso_query"),
                            nodes=nodes,
                            edges=edge_ids_for_nodes(self.graph, nodes),
                            score=1.0,
                            runtime=elapsed(start),
                            metadata={
                                "method": "turboiso",
                                "implementation_status": "wrapper_networkx_fallback",
                                "timeout": timeout,
                            },
                        )
                    )
                    if len(predictions) >= max_predictions:
                        break
            except TimeoutError:
                predictions.append(
                    prediction(
                        query_id=query.get("query_id", "turboiso_query"),
                        nodes=[],
                        edges=[],
                        score=0.0,
                        runtime=elapsed(start),
                        metadata={"method": "turboiso", "status": "timeout", "timeout": timeout},
                    )
                )
            finally:
                signal.alarm(0)
        return predictions


def export_edgelist(dataset: Any, path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset.load_edges()[["src", "dst"]].itertuples(index=False):
            f.write(f"{row.src} {row.dst}\n")
    return path
