import signal
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

from methods.base_method import BaseMethod
from methods.baselines.common import elapsed, now, prediction, save_predictions


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("VF2 timeout")


class VF2Baseline(BaseMethod):
    """NetworkX VF2 exact matching baseline for small query motifs."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        self.graph = None
        self.predictions: List[Dict[str, Any]] = []

    def fit(self, dataset: Any) -> Any:
        cfg = self.config.get("baseline", {})
        directed = bool(cfg.get("directed", False))
        graph = nx.DiGraph() if directed else nx.Graph()
        for row in dataset.load_edges()[["edge_id", "src", "dst"]].itertuples(index=False):
            graph.add_edge(str(row.src), str(row.dst), edge_id=str(row.edge_id))
        self.graph = graph
        return graph

    def predict(self, dataset: Any) -> List[Dict[str, Any]]:
        cfg = self.config.get("baseline", {})
        if self.graph is None:
            self.fit(dataset)
        max_predictions = int(cfg.get("max_predictions", 100))
        timeout = int(cfg.get("timeout", 30))
        self.predictions = []
        start = now()
        for query in dataset.load_query_motifs():
            if len(self.predictions) >= max_predictions:
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
                    edges = edge_ids_for_nodes(self.graph, nodes)
                    self.predictions.append(
                        prediction(
                            query_id=query.get("query_id", "vf2_query"),
                            nodes=nodes,
                            edges=edges,
                            score=1.0,
                            runtime=elapsed(start),
                            metadata={"method": "vf2", "implementation_status": "full_networkx", "timeout": timeout},
                        )
                    )
                    if len(self.predictions) >= max_predictions:
                        break
            except TimeoutError:
                self.predictions.append(
                    prediction(
                        query_id=query.get("query_id", "vf2_query"),
                        nodes=[],
                        edges=[],
                        score=0.0,
                        runtime=elapsed(start),
                        metadata={"method": "vf2", "status": "timeout", "timeout": timeout},
                    )
                )
            finally:
                signal.alarm(0)
        save_predictions(self.predictions, self.output_dir / "predictions.jsonl")
        return self.predictions

    def evaluate(self, dataset: Any) -> Any:
        from evaluation.evaluate import evaluate_predictions
        return evaluate_predictions(self.predictions, dataset.load_ground_truth(), self.output_dir)


def build_query_graph(query: Dict[str, Any], directed: bool):
    graph = nx.DiGraph() if directed else nx.Graph()
    for node in query.get("nodes", []):
        if isinstance(node, str) and node != "..." and not node.endswith("_i"):
            graph.add_node(node)
    for edge in query.get("edges", []):
        if isinstance(edge, dict):
            src = edge.get("src")
            dst = edge.get("dst")
        elif isinstance(edge, list) and len(edge) == 2:
            src, dst = edge
        else:
            continue
        if "..." in (src, dst) or str(src).endswith("_i") or str(dst).endswith("_i"):
            continue
        graph.add_edge(src, dst)
    return graph


def edge_ids_for_nodes(graph, nodes: List[str]) -> List[str]:
    node_set = set(nodes)
    edge_ids = []
    for src, dst, data in graph.edges(data=True):
        if src in node_set and dst in node_set and "edge_id" in data:
            edge_ids.append(str(data["edge_id"]))
    return sorted(edge_ids)
