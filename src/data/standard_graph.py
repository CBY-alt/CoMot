import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from data.query_view import load_method_queries


Prediction = Dict[str, Any]


class StandardGraphDataset:
    """Lightweight reader for CoMot standard processed datasets."""

    def __init__(self, dataset_name: str, project_root: Path, config: Dict[str, Any]):
        self.dataset_name = dataset_name
        self.project_root = Path(project_root)
        self.config = config
        data_cfg = config.get("data", {})
        dataset_dir = data_cfg.get("dataset_dir", f"data/processed/{dataset_name}")
        self.dataset_dir = self._resolve(dataset_dir)
        self.method_queries_path = self.dataset_dir / "method_queries.json"
        self.full_query_motifs_path = self.dataset_dir / "query_motifs.json"
        self.max_edges = data_cfg.get("max_edges")
        self.max_nodes = data_cfg.get("max_nodes")

        self.nodes_df: Optional[pd.DataFrame] = None
        self.edges_df: Optional[pd.DataFrame] = None
        self.ground_truth: Optional[List[Dict[str, Any]]] = None
        self.query_motifs: Optional[List[Dict[str, Any]]] = None
        self.partitions: Optional[List[Dict[str, Any]]] = None

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    def load_nodes(self) -> pd.DataFrame:
        if self.nodes_df is None:
            df = pd.read_csv(self.dataset_dir / "nodes.csv", dtype=str)
            if self.max_nodes not in (None, "all"):
                df = df.head(int(self.max_nodes)).copy()
            self.nodes_df = df
        return self.nodes_df

    def load_edges(self) -> pd.DataFrame:
        if self.edges_df is None:
            if self.max_edges in (None, "all"):
                df = pd.read_csv(self.dataset_dir / "edges.csv", dtype=str)
            else:
                df = pd.read_csv(self.dataset_dir / "edges.csv", dtype=str, nrows=int(self.max_edges))
            node_ids = set(self.load_nodes()["node_id"].astype(str))
            df = df[df["src"].astype(str).isin(node_ids) & df["dst"].astype(str).isin(node_ids)].copy()
            self.edges_df = df
        return self.edges_df

    def load_ground_truth(self) -> List[Dict[str, Any]]:
        if self.ground_truth is None:
            with open(self.dataset_dir / "ground_truth.json", "r", encoding="utf-8") as f:
                self.ground_truth = json.load(f)
        return self.ground_truth

    def load_query_motifs(self) -> List[Dict[str, Any]]:
        if self.query_motifs is None:
            self.query_motifs = load_method_queries(self.dataset_dir)
        return self.query_motifs

    def load_partitions(self) -> List[Dict[str, Any]]:
        if self.partitions is None:
            with open(self.dataset_dir / "partitions.json", "r", encoding="utf-8") as f:
                self.partitions = json.load(f)
        return self.partitions

    def graph(self, directed: bool = False):
        edges = self.load_edges()
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        edge_lookup: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for row in edges[["edge_id", "src", "dst"]].itertuples(index=False):
            src = str(row.src)
            dst = str(row.dst)
            edge_id = str(row.edge_id)
            adjacency[src].add(dst)
            adjacency.setdefault(dst, set())
            edge_lookup[(src, dst)].append(edge_id)
            if not directed:
                adjacency[dst].add(src)
                edge_lookup[(dst, src)].append(edge_id)
        return adjacency, edge_lookup


def now() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return float(time.perf_counter() - start)


def prediction(query_id: str, nodes: Iterable[str], edges: Iterable[str], score: float, runtime: float, metadata: Dict[str, Any]) -> Prediction:
    return {
        "query_id": str(query_id),
        "predicted_nodes": [str(x) for x in nodes],
        "predicted_edges": [str(x) for x in edges],
        "score": float(score),
        "runtime": float(runtime),
        "metadata": metadata,
    }


def save_predictions(predictions: List[Prediction], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item) + "\n")
    return path


def load_json_or_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    return json.loads(text)
