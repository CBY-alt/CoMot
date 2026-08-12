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


DirectedEdge = Tuple[str, str]


QUERY_DEFS = [
    {
        "query_id": "directed_star",
        "motif_type": "directed_star",
        "nodes": ["center", "leaf_1", "..."],
        "edges": [["center", "leaf_i"]],
        "anchor_nodes": ["center"],
        "description": "Directed out-star centered on a webpage with multiple outgoing hyperlinks.",
    },
    {
        "query_id": "directed_chain",
        "motif_type": "directed_chain",
        "nodes": ["v_1", "v_2", "..."],
        "edges": [["v_i", "v_{i+1}"]],
        "anchor_nodes": ["v_1"],
        "description": "Directed hyperlink path.",
    },
    {
        "query_id": "directed_cycle",
        "motif_type": "directed_cycle",
        "nodes": ["v_1", "v_2", "..."],
        "edges": [["v_i", "v_{i+1}"], ["v_k", "v_1"]],
        "anchor_nodes": [],
        "description": "Directed hyperlink cycle.",
    },
    {
        "query_id": "feed_forward",
        "motif_type": "feed_forward",
        "nodes": ["a", "b", "c"],
        "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
        "anchor_nodes": ["a"],
        "description": "Feed-forward-like hyperlink pattern.",
    },
    {
        "query_id": "dense_hyperlink_block",
        "motif_type": "dense_hyperlink_block",
        "nodes": ["page_i"],
        "edges": [["page_i", "page_j"]],
        "anchor_nodes": [],
        "description": "Small dense directed hyperlink block sampled from local neighborhoods.",
    },
]


class WebGoogleDataset(BaseDataset):
    """Web-Google hyperlink graph adapter."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        dataset_cfg = config.get("dataset", {})
        sample_cfg = config.get("sampling", {})
        motif_cfg = config.get("motifs", {})
        partition_cfg = config.get("partition", {})

        self.raw_dir = self._resolve_path(dataset_cfg.get("raw_dir", "data/raw/web_google"))
        self.processed_dir = self._resolve_path(dataset_cfg.get("processed_dir", "data/processed/web_google"))
        self.dataset_name = dataset_cfg.get("dataset_name", "web_google")
        self.edge_path = self.raw_dir / dataset_cfg.get("edge_file", "web-Google.txt")

        self.max_nodes = parse_optional_int(sample_cfg.get("max_nodes", 200_000))
        self.max_edges = parse_optional_int(sample_cfg.get("max_edges", 1_000_000))
        self.motif_size = int(motif_cfg.get("motif_size", 4))
        self.num_queries = int(motif_cfg.get("num_queries", 1000))
        self.partition_num = int(partition_cfg.get("partition_num", 5))

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    def load_raw(self) -> Dict[str, Path]:
        if not self.edge_path.exists():
            raise FileNotFoundError(self.edge_path)
        return {"edges": self.edge_path}

    def preprocess(self) -> Dict[str, Any]:
        self.load_raw()
        nodes, edges = read_sampled_directed_edges(
            self.edge_path,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
        )
        out_adj, in_adj = build_directed_adjacency(edges)
        edge_ids = {edge: f"e{idx}" for idx, edge in enumerate(edges)}
        return {"nodes": nodes, "edges": edges, "out_adj": out_adj, "in_adj": in_adj, "edge_ids": edge_ids}

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
            nodes=graph["nodes"],
            out_adj=graph["out_adj"],
            in_adj=graph["in_adj"],
            edge_ids=graph["edge_ids"],
            partitions=partitions,
            motif_size=self.motif_size,
            num_queries=self.num_queries,
            seed=self.seed,
        )

        write_nodes(self.processed_dir / "nodes.csv", graph["nodes"])
        write_edges(self.processed_dir / "edges.csv", graph["edges"])
        write_node_features(self.processed_dir / "node_features.csv", graph["nodes"], graph["out_adj"], graph["in_adj"])
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
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
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


def parse_optional_int(value) -> Optional[int]:
    if value in (None, "", "none", "None", "all"):
        return None
    return int(value)


def read_sampled_directed_edges(path: Path, max_nodes: Optional[int], max_edges: Optional[int]) -> Tuple[List[str], List[DirectedEdge]]:
    nodes: Set[str] = set()
    edges: List[DirectedEdge] = []
    seen: Set[DirectedEdge] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            src, dst = line.split()[:2]
            if src == dst:
                continue
            edge = (src, dst)
            if edge in seen:
                continue

            new_nodes = [node for node in edge if node not in nodes]
            if max_nodes is not None and len(nodes) + len(new_nodes) > max_nodes:
                if src not in nodes or dst not in nodes:
                    continue

            nodes.update(edge)
            edges.append(edge)
            seen.add(edge)
            if max_edges is not None and len(edges) >= max_edges:
                break

    return sorted(nodes, key=node_sort_key), edges


def node_sort_key(value: str):
    return int(value) if value.isdigit() else value


def build_directed_adjacency(edges: Iterable[DirectedEdge]):
    out_adj: Dict[str, Set[str]] = defaultdict(set)
    in_adj: Dict[str, Set[str]] = defaultdict(set)
    for src, dst in edges:
        out_adj[src].add(dst)
        in_adj[dst].add(src)
        out_adj.setdefault(dst, set())
        in_adj.setdefault(src, set())
    return out_adj, in_adj


def assign_partitions(nodes: List[str], partition_num: int, seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    shuffled = list(nodes)
    rng.shuffle(shuffled)
    return {node: f"P{idx % partition_num:03d}" for idx, node in enumerate(shuffled)}


def sample_motif_instances(
    nodes: List[str],
    out_adj: Dict[str, Set[str]],
    in_adj: Dict[str, Set[str]],
    edge_ids: Dict[DirectedEdge, str],
    partitions: Dict[str, str],
    motif_size: int,
    num_queries: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    per_type = max(1, num_queries // len(QUERY_DEFS))
    remainder = num_queries % len(QUERY_DEFS)
    quotas = {
        "directed_star": per_type + (1 if remainder > 0 else 0),
        "directed_chain": per_type + (1 if remainder > 1 else 0),
        "directed_cycle": per_type + (1 if remainder > 2 else 0),
        "feed_forward": per_type + (1 if remainder > 3 else 0),
        "dense_hyperlink_block": per_type,
    }

    instances = []
    instances.extend(sample_directed_stars(nodes, out_adj, edge_ids, partitions, quotas["directed_star"], motif_size, rng))
    instances.extend(sample_directed_chains(nodes, out_adj, edge_ids, partitions, quotas["directed_chain"], motif_size, rng))
    instances.extend(sample_directed_cycles(nodes, out_adj, edge_ids, partitions, quotas["directed_cycle"], motif_size, rng))
    instances.extend(sample_feed_forward(nodes, out_adj, edge_ids, partitions, quotas["feed_forward"], rng))
    instances.extend(sample_dense_blocks(nodes, out_adj, edge_ids, partitions, quotas["dense_hyperlink_block"], motif_size, rng))

    instances = prefer_cross_partition(instances, num_queries)
    for idx, item in enumerate(instances):
        item["motif_instance_id"] = f"WEBGOOGLE_MOTIF_{idx:06d}"
    return instances


def motif_record(query_id: str, nodes: List[str], edge_ids_list: List[str], partitions: Dict[str, str]) -> Dict[str, Any]:
    node_set = sorted(set(nodes), key=node_sort_key)
    return {
        "query_id": query_id,
        "true_nodes": node_set,
        "true_edges": sorted(set(edge_ids_list)),
        "true_partitions": sorted({partitions[n] for n in node_set if n in partitions}),
        "motif_instance_id": "",
    }


def get_edge_id(edge_ids: Dict[DirectedEdge, str], src: str, dst: str) -> Optional[str]:
    return edge_ids.get((src, dst))


def sample_directed_stars(nodes, out_adj, edge_ids, partitions, quota, motif_size, rng):
    leaf_count = max(2, motif_size - 1)
    centers = [n for n in nodes if len(out_adj.get(n, [])) >= leaf_count]
    rng.shuffle(centers)
    results = []
    for center in centers:
        leaves = rng.sample(sorted(out_adj[center], key=node_sort_key), leaf_count)
        ids = [get_edge_id(edge_ids, center, leaf) for leaf in leaves]
        if all(ids):
            results.append(motif_record("directed_star", [center] + leaves, ids, partitions))
        if len(results) >= quota:
            return results
    return results


def sample_directed_chains(nodes, out_adj, edge_ids, partitions, quota, motif_size, rng):
    results = []
    seen = set()
    attempts = 0
    max_attempts = max(10_000, quota * 200)
    while len(results) < quota and attempts < max_attempts:
        attempts += 1
        current = rng.choice(nodes)
        path = [current]
        ids = []
        while len(path) < motif_size:
            candidates = [n for n in out_adj.get(current, []) if n not in path]
            if not candidates:
                break
            nxt = rng.choice(sorted(candidates, key=node_sort_key))
            eid = get_edge_id(edge_ids, current, nxt)
            if not eid:
                break
            path.append(nxt)
            ids.append(eid)
            current = nxt
        key = tuple(path)
        if len(path) == motif_size and key not in seen:
            seen.add(key)
            results.append(motif_record("directed_chain", path, ids, partitions))
    return results


def sample_directed_cycles(nodes, out_adj, edge_ids, partitions, quota, motif_size, rng):
    results = []
    seen = set()
    attempts = 0
    max_attempts = max(20_000, quota * 500)
    while len(results) < quota and attempts < max_attempts:
        attempts += 1
        start = rng.choice(nodes)
        current = start
        path = [start]
        ids = []
        while len(path) < motif_size:
            candidates = [n for n in out_adj.get(current, []) if n not in path and n != start]
            if not candidates:
                break
            nxt = rng.choice(sorted(candidates, key=node_sort_key))
            eid = get_edge_id(edge_ids, current, nxt)
            if not eid:
                break
            path.append(nxt)
            ids.append(eid)
            current = nxt
        closing = get_edge_id(edge_ids, current, start)
        key = tuple(path)
        if len(path) == motif_size and closing and key not in seen:
            seen.add(key)
            results.append(motif_record("directed_cycle", path, ids + [closing], partitions))
    return results


def sample_feed_forward(nodes, out_adj, edge_ids, partitions, quota, rng):
    candidates = [n for n in nodes if len(out_adj.get(n, [])) >= 2]
    rng.shuffle(candidates)
    results = []
    seen = set()
    for a in candidates:
        outs = sorted(out_adj[a], key=node_sort_key)
        for _ in range(min(200, len(outs) * 2)):
            b, c = rng.sample(outs, 2)
            if c not in out_adj.get(b, set()):
                b, c = c, b
            if c not in out_adj.get(b, set()):
                continue
            ids = [get_edge_id(edge_ids, a, b), get_edge_id(edge_ids, a, c), get_edge_id(edge_ids, b, c)]
            key = (a, b, c)
            if all(ids) and key not in seen:
                seen.add(key)
                results.append(motif_record("feed_forward", [a, b, c], ids, partitions))
            if len(results) >= quota:
                return results
    return results


def sample_dense_blocks(nodes, out_adj, edge_ids, partitions, quota, motif_size, rng):
    centers = [n for n in nodes if len(out_adj.get(n, [])) + 1 >= motif_size]
    centers.sort(key=lambda n: (-len(out_adj.get(n, [])), node_sort_key(n)))
    rng.shuffle(centers)
    results = []
    seen = set()
    min_edges = max(motif_size, motif_size * (motif_size - 1) // 3)
    for center in centers:
        neighbors = sorted(out_adj.get(center, []), key=node_sort_key)
        if len(neighbors) < motif_size - 1:
            continue
        for _ in range(5):
            block = [center] + rng.sample(neighbors, motif_size - 1)
            block = sorted(set(block), key=node_sort_key)
            if len(block) < motif_size:
                continue
            ids = []
            for src in block:
                for dst in block:
                    if src == dst:
                        continue
                    eid = get_edge_id(edge_ids, src, dst)
                    if eid:
                        ids.append(eid)
            key = tuple(block)
            if len(ids) >= min_edges and key not in seen:
                seen.add(key)
                results.append(motif_record("dense_hyperlink_block", block, ids, partitions))
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
        "node_type": "webpage",
        "label": "",
    }).to_csv(path, index=False)


def write_edges(path: Path, edges: List[DirectedEdge]) -> None:
    pd.DataFrame({
        "edge_id": [f"e{i}" for i in range(len(edges))],
        "src": [src for src, _ in edges],
        "dst": [dst for _, dst in edges],
        "timestamp": "",
        "weight": "",
        "edge_type": "hyperlink",
        "label": "",
    }).to_csv(path, index=False)


def write_node_features(path: Path, nodes: List[str], out_adj: Dict[str, Set[str]], in_adj: Dict[str, Set[str]]) -> None:
    pd.DataFrame({
        "node_id": nodes,
        "out_degree": [len(out_adj.get(n, [])) for n in nodes],
        "in_degree": [len(in_adj.get(n, [])) for n in nodes],
    }).to_csv(path, index=False)


def write_edge_features(path: Path, num_edges: int) -> None:
    pd.DataFrame({"edge_id": [f"e{i}" for i in range(num_edges)]}).to_csv(path, index=False)


def write_labels(path: Path, nodes: List[str], num_edges: int, motif_instances: List[Dict[str, Any]]) -> None:
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
    motif_labels = pd.DataFrame({
        "object_id": [item["motif_instance_id"] for item in motif_instances],
        "object_type": "motif",
        "label": [item["query_id"] for item in motif_instances],
        "label_name": [item["query_id"] for item in motif_instances],
    })
    pd.concat([node_labels, edge_labels, motif_labels], ignore_index=True).to_csv(path, index=False)


def write_partitions(path: Path, nodes: List[str], edges: List[DirectedEdge], partitions: Dict[str, str]) -> None:
    visible_nodes = defaultdict(list)
    visible_edges = defaultdict(list)
    for node in nodes:
        visible_nodes[partitions[node]].append(node)
    for idx, (src, dst) in enumerate(edges):
        edge_id = f"e{idx}"
        ps = partitions[src]
        pdst = partitions[dst]
        visible_edges[ps].append(edge_id)
        if pdst != ps:
            visible_edges[pdst].append(edge_id)
    payload = []
    for partition_id in sorted(visible_nodes):
        payload.append({
            "partition_id": partition_id,
            "visible_nodes": sorted(visible_nodes[partition_id], key=node_sort_key),
            "visible_edges": visible_edges[partition_id],
        })
    write_json(path, payload)


def write_metadata(
    path: Path,
    dataset_name: str,
    num_nodes: int,
    num_edges: int,
    num_motifs: int,
    max_nodes: Optional[int],
    max_edges: Optional[int],
    motif_size: int,
    num_queries: int,
    partition_num: int,
    seed: int,
) -> None:
    payload = {
        "dataset_name": dataset_name,
        "graph_type": "web_hyperlink",
        "directed": True,
        "weighted": False,
        "has_timestamps": False,
        "has_node_features": True,
        "has_edge_features": False,
        "has_ground_truth": True,
        "source": "SNAP Web-Google",
        "citation": "leskovec2009community",
        "paper": "Community structure in large networks: Natural cluster sizes and the absence of large well-defined clusters",
        "scenario": "Web hyperlink graph and community structure",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_motif_instances": num_motifs,
        "format_version": "1.0",
        "sampling": {
            "max_nodes": max_nodes,
            "max_edges": max_edges,
            "strategy": "streaming edge-list prefix with node and edge caps",
        },
        "motif_construction": {
            "motif_types": ["directed_star", "directed_chain", "directed_cycle", "feed_forward", "dense_hyperlink_block"],
            "motif_size": motif_size,
            "num_queries": num_queries,
            "seed": seed,
            "sampling": "deterministic random sampling from the sampled directed hyperlink graph",
        },
        "partition_construction": {
            "partition_num": partition_num,
            "seed": seed,
            "strategy": "seeded random assignment of webpages to partitions; cross-partition motif instances are preferred",
        },
    }
    write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Web-Google to CoMot standard format.")
    parser.add_argument("--raw_dir", default="data/raw/web_google")
    parser.add_argument("--processed_dir", default="data/processed/web_google")
    parser.add_argument("--max_nodes", default="200000")
    parser.add_argument("--max_edges", default="1000000")
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
            "dataset_name": "web_google",
        },
        "sampling": {
            "max_nodes": parse_optional_int(args.max_nodes),
            "max_edges": parse_optional_int(args.max_edges),
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
    dataset = WebGoogleDataset(config=config, project_root=project_root, output_dir=project_root / "outputs", seed=args.seed)
    outputs = dataset.export_standard_format()
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
