import argparse
import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.base_dataset import BaseDataset
from data.standard_format import ensure_dir, write_json


CLASS_TO_LABEL = {
    "1": "illicit",
    "2": "licit",
    "unknown": "unknown",
}


class EllipticDataset(BaseDataset):
    """Elliptic Bitcoin transaction graph adapter."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        dataset_cfg = config.get("dataset", {})
        motif_cfg = config.get("motif_ground_truth", {})

        self.raw_dir = self._resolve_path(dataset_cfg.get("raw_dir", "data/raw/elliptic"))
        self.processed_dir = self._resolve_path(dataset_cfg.get("processed_dir", "data/processed/elliptic"))
        self.dataset_name = dataset_cfg.get("dataset_name", "elliptic")
        self.k_hop = int(motif_cfg.get("k_hop", 1))
        self.max_instances = motif_cfg.get("max_instances", 1000)
        self.max_edges_per_instance = int(motif_cfg.get("max_edges_per_instance", 200))

        self.dataset_dir = self.raw_dir / "elliptic_bitcoin_dataset"
        self.classes_path = self.dataset_dir / "elliptic_txs_classes.csv"
        self.edgelist_path = self.dataset_dir / "elliptic_txs_edgelist.csv"
        self.features_path = self.dataset_dir / "elliptic_txs_features.csv"

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    def load_raw(self) -> Dict[str, Path]:
        missing = [
            path for path in [self.classes_path, self.edgelist_path, self.features_path]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing Elliptic raw files: {[str(p) for p in missing]}")
        return {
            "classes": self.classes_path,
            "edgelist": self.edgelist_path,
            "features": self.features_path,
        }

    def preprocess(self) -> Dict[str, pd.DataFrame]:
        self.load_raw()
        classes = pd.read_csv(self.classes_path, dtype={"txId": "string", "class": "string"})
        edges = pd.read_csv(self.edgelist_path, dtype={"txId1": "string", "txId2": "string"})

        feature_cols = ["txId", "time_step"] + [f"f{i:03d}" for i in range(1, 166)]
        features = pd.read_csv(self.features_path, header=None, names=feature_cols, dtype={"txId": "string"})
        features["time_step"] = features["time_step"].astype(int)

        nodes = features[["txId", "time_step"]].merge(classes, on="txId", how="left")
        nodes["class"] = nodes["class"].fillna("unknown").astype(str)
        nodes["label_name"] = nodes["class"].map(CLASS_TO_LABEL).fillna("unknown")
        return {"nodes": nodes, "edges": edges, "features": features}

    def build_graph(self) -> Dict[str, List[Tuple[str, str, str]]]:
        raw = self.preprocess()
        edge_rows = []
        for idx, row in raw["edges"].iterrows():
            edge_rows.append((f"e{idx}", str(row["txId1"]), str(row["txId2"])))
        return {"edges": edge_rows}

    def generate_partitions(self) -> Path:
        # Temporal partitions are exported in export_standard_format().
        return self.processed_dir / "partitions.json"

    def get_query_motifs(self) -> Path:
        return self.processed_dir / "query_motifs.json"

    def get_ground_truth(self) -> Path:
        return self.processed_dir / "ground_truth.json"

    def export_standard_format(self) -> Dict[str, Path]:
        raw = self.preprocess()
        ensure_dir(self.processed_dir)

        nodes = raw["nodes"]
        edges = raw["edges"].copy()
        features = raw["features"]

        edge_table = self._build_edge_table(edges, nodes)
        self._write_nodes(nodes)
        self._write_edges(edge_table)
        self._write_node_features(features)
        self._write_edge_features(edge_table)
        self._write_labels(nodes, edge_table)
        self._write_partitions(nodes, edge_table)
        self._write_query_motifs()
        self._write_ground_truth(nodes, edge_table)
        self._write_metadata(nodes, edge_table)

        return {name: self.processed_dir / name for name in [
            "nodes.csv",
            "edges.csv",
            "node_features.csv",
            "edge_features.csv",
            "labels.csv",
            "partitions.json",
            "query_motifs.json",
            "ground_truth.json",
            "metadata.json",
        ]}

    def _build_edge_table(self, edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
        time_lookup = nodes.set_index("txId")["time_step"].to_dict()
        label_lookup = nodes.set_index("txId")["label_name"].to_dict()
        out = pd.DataFrame({
            "edge_id": [f"e{i}" for i in range(len(edges))],
            "src": edges["txId1"].astype(str),
            "dst": edges["txId2"].astype(str),
        })
        out["timestamp"] = out["src"].map(time_lookup)
        out["weight"] = ""
        out["edge_type"] = "bitcoin_flow"
        out["label"] = out.apply(
            lambda row: "illicit_related"
            if label_lookup.get(row["src"]) == "illicit" or label_lookup.get(row["dst"]) == "illicit"
            else "unlabeled",
            axis=1,
        )
        return out

    def _write_nodes(self, nodes: pd.DataFrame) -> None:
        out = pd.DataFrame({
            "node_id": nodes["txId"].astype(str),
            "original_id": nodes["txId"].astype(str),
            "node_type": "transaction",
            "label": nodes["label_name"].astype(str),
        })
        out.to_csv(self.processed_dir / "nodes.csv", index=False)

    def _write_edges(self, edge_table: pd.DataFrame) -> None:
        edge_table[["edge_id", "src", "dst", "timestamp", "weight", "edge_type", "label"]].to_csv(
            self.processed_dir / "edges.csv", index=False
        )

    def _write_node_features(self, features: pd.DataFrame) -> None:
        out = features.rename(columns={"txId": "node_id"})
        out.to_csv(self.processed_dir / "node_features.csv", index=False)

    def _write_edge_features(self, edge_table: pd.DataFrame) -> None:
        pd.DataFrame({"edge_id": edge_table["edge_id"]}).to_csv(
            self.processed_dir / "edge_features.csv", index=False
        )

    def _write_labels(self, nodes: pd.DataFrame, edge_table: pd.DataFrame) -> None:
        node_labels = pd.DataFrame({
            "object_id": nodes["txId"].astype(str),
            "object_type": "node",
            "label": nodes["label_name"].astype(str),
            "label_name": nodes["label_name"].astype(str),
        })
        edge_labels = pd.DataFrame({
            "object_id": edge_table["edge_id"].astype(str),
            "object_type": "edge",
            "label": edge_table["label"].astype(str),
            "label_name": edge_table["label"].astype(str),
        })
        selected = self._selected_illicit_nodes(nodes)
        motif_labels = pd.DataFrame({
            "object_id": [f"ELLIPTIC_ILLICIT_EGO_{idx:06d}" for idx, _ in enumerate(selected)],
            "object_type": "motif",
            "label": "illicit_khop_ego",
            "label_name": "illicit_khop_ego",
        })
        pd.concat([node_labels, edge_labels, motif_labels], ignore_index=True).to_csv(
            self.processed_dir / "labels.csv", index=False
        )

    def _write_partitions(self, nodes: pd.DataFrame, edge_table: pd.DataFrame) -> None:
        visible_nodes = defaultdict(list)
        visible_edges = defaultdict(list)

        for row in nodes[["txId", "time_step"]].itertuples(index=False):
            visible_nodes[f"time_{int(row.time_step):02d}"].append(str(row.txId))

        for row in edge_table[["edge_id", "timestamp"]].itertuples(index=False):
            if pd.isna(row.timestamp):
                continue
            visible_edges[f"time_{int(row.timestamp):02d}"].append(str(row.edge_id))

        payload = []
        for partition_id in sorted(visible_nodes):
            payload.append({
                "partition_id": partition_id,
                "visible_nodes": visible_nodes[partition_id],
                "visible_edges": visible_edges.get(partition_id, []),
            })
        write_json(self.processed_dir / "partitions.json", payload)

    def _write_query_motifs(self) -> None:
        payload = [{
            "query_id": "illicit_khop_ego",
            "motif_type": "illicit_khop_ego",
            "nodes": [],
            "edges": [],
            "anchor_nodes": ["illicit_transaction"],
            "description": (
                f"Directed {self.k_hop}-hop ego transaction subgraph centered on a labeled illicit "
                "Elliptic transaction node. Instances are sampled deterministically from sorted illicit nodes."
            ),
        }]
        write_json(self.processed_dir / "query_motifs.json", payload)

    def _write_ground_truth(self, nodes: pd.DataFrame, edge_table: pd.DataFrame) -> None:
        adjacency_out, adjacency_in, edge_lookup = self._adjacency(edge_table)
        time_lookup = nodes.set_index("txId")["time_step"].to_dict()
        payload = []

        for idx, center in enumerate(self._selected_illicit_nodes(nodes)):
            true_nodes, true_edges = self._khop_subgraph(center, adjacency_out, adjacency_in, edge_lookup)
            if len(true_edges) > self.max_edges_per_instance:
                true_edges = true_edges[:self.max_edges_per_instance]
                true_nodes = sorted({n for edge_id in true_edges for n in edge_lookup[edge_id]})
                if center not in true_nodes:
                    true_nodes = [center] + true_nodes

            partitions = sorted({
                f"time_{int(time_lookup[n]):02d}"
                for n in true_nodes
                if n in time_lookup and not pd.isna(time_lookup[n])
            })
            payload.append({
                "query_id": "illicit_khop_ego",
                "true_nodes": true_nodes,
                "true_edges": true_edges,
                "true_partitions": partitions,
                "motif_instance_id": f"ELLIPTIC_ILLICIT_EGO_{idx:06d}",
            })
        write_json(self.processed_dir / "ground_truth.json", payload)

    def _write_metadata(self, nodes: pd.DataFrame, edge_table: pd.DataFrame) -> None:
        payload = {
            "dataset_name": self.dataset_name,
            "graph_type": "bitcoin_transaction",
            "directed": True,
            "weighted": False,
            "has_timestamps": True,
            "has_node_features": True,
            "has_edge_features": False,
            "has_ground_truth": True,
            "source": "Elliptic Data Set",
            "citation": "weber2019anti",
            "paper": "Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics",
            "scenario": "Bitcoin illicit transaction detection and financial forensics",
            "num_nodes": int(len(nodes)),
            "num_edges": int(len(edge_table)),
            "num_motif_instances": int(len(self._selected_illicit_nodes(nodes))),
            "format_version": "1.0",
            "ground_truth_construction": {
                "query_id": "illicit_khop_ego",
                "center_label": "illicit",
                "k_hop": self.k_hop,
                "max_instances": self.max_instances,
                "max_edges_per_instance": self.max_edges_per_instance,
                "seed": self.seed,
                "sampling": "deterministic sample from sorted illicit transaction ids",
            },
        }
        write_json(self.processed_dir / "metadata.json", payload)

    def _selected_illicit_nodes(self, nodes: pd.DataFrame) -> List[str]:
        illicit = sorted(nodes.loc[nodes["label_name"] == "illicit", "txId"].astype(str).tolist())
        if self.max_instances in (None, "all"):
            return illicit

        max_instances = int(self.max_instances)
        if len(illicit) <= max_instances:
            return illicit
        rng = random.Random(self.seed)
        return sorted(rng.sample(illicit, max_instances))

    def _adjacency(self, edge_table: pd.DataFrame):
        adjacency_out = defaultdict(list)
        adjacency_in = defaultdict(list)
        edge_lookup = {}
        for row in edge_table[["edge_id", "src", "dst"]].itertuples(index=False):
            edge_id = str(row.edge_id)
            src = str(row.src)
            dst = str(row.dst)
            adjacency_out[src].append((dst, edge_id))
            adjacency_in[dst].append((src, edge_id))
            edge_lookup[edge_id] = (src, dst)
        for adj in (adjacency_out, adjacency_in):
            for node in adj:
                adj[node] = sorted(adj[node], key=lambda x: (x[0], x[1]))
        return adjacency_out, adjacency_in, edge_lookup

    def _khop_subgraph(self, center: str, adjacency_out, adjacency_in, edge_lookup) -> Tuple[List[str], List[str]]:
        visited_nodes: Set[str] = {center}
        visited_edges: Set[str] = set()
        queue = deque([(center, 0)])

        while queue:
            node, depth = queue.popleft()
            if depth >= self.k_hop:
                continue

            neighbors = adjacency_out.get(node, []) + adjacency_in.get(node, [])
            for neighbor, edge_id in neighbors:
                visited_edges.add(edge_id)
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return sorted(visited_nodes), sorted(visited_edges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Elliptic Data Set to CoMot standard format.")
    parser.add_argument("--raw_dir", default="data/raw/elliptic")
    parser.add_argument("--processed_dir", default="data/processed/elliptic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_hop", type=int, default=1)
    parser.add_argument("--max_instances", default="1000")
    parser.add_argument("--max_edges_per_instance", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_instances: Any = args.max_instances
    if max_instances != "all":
        max_instances = int(max_instances)
    config = {
        "dataset": {
            "raw_dir": args.raw_dir,
            "processed_dir": args.processed_dir,
            "dataset_name": "elliptic",
        },
        "motif_ground_truth": {
            "k_hop": args.k_hop,
            "max_instances": max_instances,
            "max_edges_per_instance": args.max_edges_per_instance,
        },
    }
    project_root = Path(__file__).resolve().parents[2]
    dataset = EllipticDataset(config=config, project_root=project_root, output_dir=project_root / "outputs", seed=args.seed)
    outputs = dataset.export_standard_format()
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
