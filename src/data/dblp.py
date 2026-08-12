import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.base_dataset import BaseDataset
from data.standard_format import ensure_dir, write_json


Edge = Tuple[str, str]


QUERY_DEFS = [
    {
        "query_id": "triangle",
        "motif_type": "triangle",
        "nodes": ["a", "b", "c"],
        "edges": [["a", "b"], ["b", "c"], ["c", "a"]],
        "anchor_nodes": [],
        "description": "Three-author closed collaboration triangle.",
    },
    {
        "query_id": "cycle4",
        "motif_type": "cycle4",
        "nodes": ["a", "b", "c", "d"],
        "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"]],
        "anchor_nodes": [],
        "description": "Four-author collaboration cycle.",
    },
    {
        "query_id": "star",
        "motif_type": "star",
        "nodes": ["center", "leaf_1", "..."],
        "edges": [["center", "leaf_i"]],
        "anchor_nodes": ["center"],
        "description": "High-degree author with sampled collaborators as leaves.",
    },
    {
        "query_id": "dense_subgraph",
        "motif_type": "dense_subgraph",
        "nodes": ["community_member_i"],
        "edges": [["community_member_i", "community_member_j"]],
        "anchor_nodes": [],
        "description": "Dense coauthor group sampled from SNAP DBLP community annotations.",
    },
]


class DBLPDataset(BaseDataset):
    """DBLP Collaboration Network adapter."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        dataset_cfg = config.get("dataset", {})
        motif_cfg = config.get("motifs", {})
        partition_cfg = config.get("partition", {})

        self.raw_dir = self._resolve_path(dataset_cfg.get("raw_dir", "data/raw/dblp"))
        self.processed_dir = self._resolve_path(dataset_cfg.get("processed_dir", "data/processed/dblp"))
        self.dataset_name = dataset_cfg.get("dataset_name", "dblp")

        self.edge_path = self.raw_dir / dataset_cfg.get("edge_file", "com-dblp.ungraph.txt")
        self.community_path = self.raw_dir / dataset_cfg.get("community_file", "com-dblp.top5000.cmty.txt")

        self.motif_size = int(motif_cfg.get("motif_size", 4))
        self.num_queries = int(motif_cfg.get("num_queries", 1000))
        self.partition_num = int(partition_cfg.get("partition_num", 5))

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    def load_raw(self) -> Dict[str, Path]:
        missing = [path for path in [self.edge_path, self.community_path] if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing DBLP raw files: {[str(p) for p in missing]}")
        return {"edges": self.edge_path, "communities": self.community_path}

    def preprocess(self) -> Dict[str, Any]:
        self.load_raw()
        nodes, edges = read_snap_undirected_edges(self.edge_path)
        adjacency = build_adjacency(edges)
        communities = read_communities(self.community_path)
        edge_ids = {edge: f"e{idx}" for idx, edge in enumerate(edges)}
        return {
            "nodes": nodes,
            "edges": edges,
            "adjacency": adjacency,
            "communities": communities,
            "edge_ids": edge_ids,
        }

    def build_graph(self) -> Dict[str, Any]:
        return self.preprocess()

    def generate_partitions(self) -> Path:
        return self.processed_dir / "partitions.json"

    def get_query_motifs(self) -> Path:
        return self.processed_dir / "query_motifs.json"

    def get_ground_truth(self) -> Path:
        return self.processed_dir / "ground_truth.json"

    def export_standard_format(self) -> Dict[str, Path]:
        graph = self.preprocess()
        ensure_dir(self.processed_dir)

        partitions = assign_partitions(graph["nodes"], self.partition_num, self.seed)
        motif_instances = sample_motif_instances(
            adjacency=graph["adjacency"],
            communities=graph["communities"],
            edge_ids=graph["edge_ids"],
            partitions=partitions,
            motif_size=self.motif_size,
            num_queries=self.num_queries,
            seed=self.seed,
        )

        write_nodes(self.processed_dir / "nodes.csv", graph["nodes"])
        write_edges(self.processed_dir / "edges.csv", graph["edges"])
        write_node_features(self.processed_dir / "node_features.csv", graph["nodes"], graph["adjacency"])
        write_edge_features(self.processed_dir / "edge_features.csv", len(graph["edges"]))
        write_labels(self.processed_dir / "labels.csv", graph["nodes"], len(graph["edges"]), motif_instances)
        write_partitions(self.processed_dir / "partitions.json", graph["nodes"], graph["edges"], partitions)
        write_json(self.processed_dir / "query_motifs.json", QUERY_DEFS)
        write_json(self.processed_dir / "ground_truth.json", motif_instances)
        write_metadata(
            self.processed_dir / "metadata.json",
            dataset_name=self.dataset_name,
            num_nodes=len(graph["nodes"]),
            num_edges=len(graph["edges"]),
            num_motifs=len(motif_instances),
            motif_size=self.motif_size,
            num_queries=self.num_queries,
            partition_num=self.partition_num,
            seed=self.seed,
        )

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


def normalize_edge(a: str, b: str) -> Edge:
    return (a, b) if a <= b else (b, a)


def read_snap_undirected_edges(path: Path) -> Tuple[List[str], List[Edge]]:
    nodes: Set[str] = set()
    edges: List[Edge] = []
    seen: Set[Edge] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split()[:2]
            if a == b:
                continue
            edge = normalize_edge(a, b)
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)
            nodes.add(a)
            nodes.add(b)
    return sorted(nodes, key=lambda x: int(x) if x.isdigit() else x), edges


def build_adjacency(edges: Iterable[Edge]) -> Dict[str, Set[str]]:
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def read_communities(path: Path) -> List[List[str]]:
    communities = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            members = [x for x in line.strip().split() if x]
            if members:
                communities.append(sorted(members, key=lambda x: int(x) if x.isdigit() else x))
    return communities


def assign_partitions(nodes: List[str], partition_num: int, seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    shuffled = list(nodes)
    rng.shuffle(shuffled)
    assignments = {}
    for idx, node in enumerate(shuffled):
        assignments[node] = f"P{idx % partition_num:03d}"
    return assignments


def sample_motif_instances(
    adjacency: Dict[str, Set[str]],
    communities: List[List[str]],
    edge_ids: Dict[Edge, str],
    partitions: Dict[str, str],
    motif_size: int,
    num_queries: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    per_type = max(1, num_queries // len(QUERY_DEFS))
    remainder = num_queries % len(QUERY_DEFS)
    quotas = {
        "triangle": per_type + (1 if remainder > 0 else 0),
        "cycle4": per_type + (1 if remainder > 1 else 0),
        "star": per_type + (1 if remainder > 2 else 0),
        "dense_subgraph": per_type,
    }

    instances = []
    instances.extend(sample_triangles(adjacency, edge_ids, partitions, quotas["triangle"], rng))
    instances.extend(sample_cycle4(adjacency, edge_ids, partitions, quotas["cycle4"], rng))
    instances.extend(sample_stars(adjacency, edge_ids, partitions, quotas["star"], motif_size, rng))
    instances.extend(sample_dense_subgraphs(adjacency, communities, edge_ids, partitions, quotas["dense_subgraph"], motif_size, rng))

    instances = prefer_cross_partition(instances, num_queries)
    for idx, item in enumerate(instances):
        item["motif_instance_id"] = f"DBLP_MOTIF_{idx:06d}"
    return instances


def motif_record(query_id: str, nodes: List[str], edge_ids: List[str], partitions: Dict[str, str]) -> Dict[str, Any]:
    return {
        "query_id": query_id,
        "true_nodes": sorted(set(nodes), key=lambda x: int(x) if x.isdigit() else x),
        "true_edges": sorted(set(edge_ids)),
        "true_partitions": sorted({partitions[n] for n in nodes if n in partitions}),
        "motif_instance_id": "",
    }


def edge_id(edge_ids: Dict[Edge, str], a: str, b: str) -> Optional[str]:
    return edge_ids.get(normalize_edge(a, b))


def sample_triangles(adjacency, edge_ids, partitions, quota: int, rng: random.Random) -> List[Dict[str, Any]]:
    nodes = list(adjacency.keys())
    rng.shuffle(nodes)
    results = []
    seen = set()
    for u in nodes:
        neigh = list(adjacency[u])
        if len(neigh) < 2:
            continue
        for _ in range(min(200, len(neigh) * 2)):
            v, w = rng.sample(neigh, 2)
            if w not in adjacency[v]:
                continue
            tri = tuple(sorted([u, v, w], key=lambda x: int(x) if x.isdigit() else x))
            if tri in seen:
                continue
            seen.add(tri)
            ids = [edge_id(edge_ids, tri[0], tri[1]), edge_id(edge_ids, tri[1], tri[2]), edge_id(edge_ids, tri[0], tri[2])]
            if all(ids):
                results.append(motif_record("triangle", list(tri), ids, partitions))
            if len(results) >= quota:
                return results
    return results


def sample_cycle4(adjacency, edge_ids, partitions, quota: int, rng: random.Random) -> List[Dict[str, Any]]:
    nodes = list(adjacency.keys())
    rng.shuffle(nodes)
    results = []
    seen = set()
    for u in nodes:
        neigh = list(adjacency[u])
        if len(neigh) < 2:
            continue
        rng.shuffle(neigh)
        for i in range(min(len(neigh), 100)):
            v = neigh[i]
            for x in neigh[i + 1:min(len(neigh), i + 50)]:
                common = list((adjacency[v] & adjacency[x]) - {u, v, x})
                if not common:
                    continue
                y = rng.choice(common)
                cyc_nodes = [u, v, y, x]
                key = tuple(sorted(cyc_nodes, key=lambda z: int(z) if z.isdigit() else z))
                if key in seen:
                    continue
                seen.add(key)
                ids = [
                    edge_id(edge_ids, u, v),
                    edge_id(edge_ids, v, y),
                    edge_id(edge_ids, y, x),
                    edge_id(edge_ids, x, u),
                ]
                if all(ids):
                    results.append(motif_record("cycle4", cyc_nodes, ids, partitions))
                if len(results) >= quota:
                    return results
    return results


def sample_stars(adjacency, edge_ids, partitions, quota: int, motif_size: int, rng: random.Random) -> List[Dict[str, Any]]:
    leaf_count = max(2, motif_size - 1)
    centers = [n for n, neigh in adjacency.items() if len(neigh) >= leaf_count]
    centers.sort(key=lambda n: (-len(adjacency[n]), int(n) if n.isdigit() else n))
    rng.shuffle(centers)
    results = []
    for center in centers:
        leaves = rng.sample(sorted(adjacency[center]), leaf_count)
        ids = [edge_id(edge_ids, center, leaf) for leaf in leaves]
        if all(ids):
            results.append(motif_record("star", [center] + leaves, ids, partitions))
        if len(results) >= quota:
            return results
    return results


def sample_dense_subgraphs(adjacency, communities, edge_ids, partitions, quota: int, motif_size: int, rng: random.Random) -> List[Dict[str, Any]]:
    candidates = [c for c in communities if len(c) >= motif_size]
    rng.shuffle(candidates)
    results = []
    seen = set()
    for community in candidates:
        members = rng.sample(community, motif_size)
        members = sorted([m for m in members if m in adjacency], key=lambda x: int(x) if x.isdigit() else x)
        if len(members) < motif_size:
            continue
        ids = []
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                eid = edge_id(edge_ids, a, b)
                if eid:
                    ids.append(eid)
        min_edges = max(motif_size, motif_size * (motif_size - 1) // 4)
        key = tuple(members)
        if len(ids) < min_edges or key in seen:
            continue
        seen.add(key)
        results.append(motif_record("dense_subgraph", members, ids, partitions))
        if len(results) >= quota:
            return results
    return results


def prefer_cross_partition(instances: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    cross = [item for item in instances if len(item["true_partitions"]) >= 2]
    same = [item for item in instances if len(item["true_partitions"]) < 2]
    return (cross + same)[:limit]


def write_nodes(path: Path, nodes: List[str]) -> None:
    pd.DataFrame({
        "node_id": nodes,
        "original_id": nodes,
        "node_type": "author",
        "label": "",
    }).to_csv(path, index=False)


def write_edges(path: Path, edges: List[Edge]) -> None:
    pd.DataFrame({
        "edge_id": [f"e{i}" for i in range(len(edges))],
        "src": [a for a, _ in edges],
        "dst": [b for _, b in edges],
        "timestamp": "",
        "weight": "",
        "edge_type": "coauthor",
        "label": "",
    }).to_csv(path, index=False)


def write_node_features(path: Path, nodes: List[str], adjacency: Dict[str, Set[str]]) -> None:
    pd.DataFrame({
        "node_id": nodes,
        "degree": [len(adjacency[n]) for n in nodes],
    }).to_csv(path, index=False)


def write_edge_features(path: Path, num_edges: int) -> None:
    pd.DataFrame({"edge_id": [f"e{i}" for i in range(num_edges)]}).to_csv(path, index=False)


def write_labels(path: Path, nodes: List[str], num_edges: int, motif_instances: List[Dict[str, Any]]) -> None:
    motif_labels = pd.DataFrame({
        "object_id": [item["motif_instance_id"] for item in motif_instances],
        "object_type": "motif",
        "label": [item["query_id"] for item in motif_instances],
        "label_name": [item["query_id"] for item in motif_instances],
    })
    # Keep node/edge rows schema-compatible even though DBLP has no node/edge class labels.
    node_labels = pd.DataFrame({
        "object_id": nodes,
        "object_type": "node",
        "label": "",
        "label_name": "",
    })
    edge_labels = pd.DataFrame({
        "object_id": [f"e{i}" for i in range(num_edges)],
        "object_type": "edge",
        "label": "",
        "label_name": "",
    })
    pd.concat([node_labels, edge_labels, motif_labels], ignore_index=True).to_csv(path, index=False)


def write_partitions(path: Path, nodes: List[str], edges: List[Edge], partitions: Dict[str, str]) -> None:
    visible_nodes = defaultdict(list)
    visible_edges = defaultdict(list)
    for node in nodes:
        visible_nodes[partitions[node]].append(node)
    for idx, (a, b) in enumerate(edges):
        eid = f"e{idx}"
        pa = partitions[a]
        pb = partitions[b]
        visible_edges[pa].append(eid)
        if pb != pa:
            visible_edges[pb].append(eid)
    payload = []
    for partition_id in sorted(visible_nodes):
        payload.append({
            "partition_id": partition_id,
            "visible_nodes": sorted(visible_nodes[partition_id], key=lambda x: int(x) if x.isdigit() else x),
            "visible_edges": visible_edges[partition_id],
        })
    write_json(path, payload)


def write_metadata(path: Path, dataset_name: str, num_nodes: int, num_edges: int, num_motifs: int, motif_size: int, num_queries: int, partition_num: int, seed: int) -> None:
    payload = {
        "dataset_name": dataset_name,
        "graph_type": "collaboration",
        "directed": False,
        "weighted": False,
        "has_timestamps": False,
        "has_node_features": True,
        "has_edge_features": False,
        "has_ground_truth": True,
        "source": "SNAP DBLP Collaboration Network",
        "citation": "leskovec2007graph",
        "paper": "Graph evolution: Densification and shrinking diameters",
        "scenario": "Collaboration network and graph structure evolution",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_motif_instances": num_motifs,
        "format_version": "1.0",
        "motif_construction": {
            "motif_types": ["triangle", "cycle4", "star", "dense_subgraph"],
            "motif_size": motif_size,
            "num_queries": num_queries,
            "seed": seed,
            "sampling": "deterministic random sampling from graph structure and SNAP community annotations",
        },
        "partition_construction": {
            "partition_num": partition_num,
            "seed": seed,
            "strategy": "seeded random assignment of authors to partitions; cross-partition motif instances are preferred",
        },
    }
    write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DBLP Collaboration Network to CoMot standard format.")
    parser.add_argument("--raw_dir", default="data/raw/dblp")
    parser.add_argument("--processed_dir", default="data/processed/dblp")
    parser.add_argument("--motif_size", type=int, default=4)
    parser.add_argument("--num_queries", type=int, default=1000)
    parser.add_argument("--partition_num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "dataset": {
            "raw_dir": args.raw_dir,
            "processed_dir": args.processed_dir,
            "dataset_name": "dblp",
        },
        "motifs": {
            "motif_size": args.motif_size,
            "num_queries": args.num_queries,
        },
        "partition": {
            "partition_num": args.partition_num,
        },
    }
    project_root = Path(__file__).resolve().parents[2]
    dataset = DBLPDataset(config=config, project_root=project_root, output_dir=project_root / "outputs", seed=args.seed)
    outputs = dataset.export_standard_format()
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
