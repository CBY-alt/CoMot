import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from evaluation.evaluate import evaluate_predictions
from methods.base_method import BaseMethod
from methods.baselines.common import StandardGraphDataset, save_predictions


class CoMotMethod(BaseMethod):
    """CoMot method wrapper for the submitted AMLWorld pipeline."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        partition_tag = config.get("partition", {}).get("tag", "ctrl_v2_003")
        self.evidence_dir = self.output_dir / f"semotif_local_evidences_{partition_tag}"
        self.orchestrator_dir = self.output_dir / f"semotif_orchestrator_{partition_tag}"
        self.candidate_dir = self.output_dir / f"semotif_candidates_{partition_tag}"
        self.candidate_eval_dir = self.output_dir / f"semotif_candidate_eval_{partition_tag}"
        self.reranked_dir = self.output_dir / f"semotif_candidates_{partition_tag}_reranked"
        self.reranked_eval_dir = self.output_dir / f"semotif_candidates_{partition_tag}_reranked_eval"
        self.breakdown_dir = self.output_dir / f"semotif_candidates_{partition_tag}_reranked_breakdown"

    def _run(self, args: List[str]) -> None:
        print("+ " + " ".join(args), flush=True)
        subprocess.run(args, cwd=self.project_root, check=True)

    def fit(self, dataset: Any) -> Dict[str, Path]:
        local_cfg = self.config.get("local_encoder", {})
        orchestrator_cfg = self.config.get("orchestrator", {})

        self._run([
            sys.executable,
            str(self.project_root / "src" / "methods" / "comot_pipeline" / "local_encoder.py"),
            "--partition_dir",
            str(dataset.partition_dir),
            "--output_dir",
            str(self.evidence_dir),
            "--k_hop",
            str(local_cfg.get("k_hop", 2)),
            "--num_workers",
            str(local_cfg.get("num_workers", 1)),
        ])

        self._run([
            sys.executable,
            str(self.project_root / "src" / "methods" / "comot_pipeline" / "orchestrator.py"),
            "--evidence_dir",
            str(self.evidence_dir),
            "--output_dir",
            str(self.orchestrator_dir),
            "--threshold",
            str(orchestrator_cfg.get("threshold", 0.72)),
            "--max_group_size",
            str(orchestrator_cfg.get("max_group_size", 5)),
            "--topk_per_sender",
            str(orchestrator_cfg.get("topk_per_sender", 1)),
            "--topk_per_receiver",
            str(orchestrator_cfg.get("topk_per_receiver", 1)),
        ])
        return {"evidence_dir": self.evidence_dir, "orchestrator_dir": self.orchestrator_dir}

    def predict(self, dataset: Any) -> Dict[str, Path]:
        builder_cfg = self.config.get("candidate_builder", {})
        self._run([
            sys.executable,
            str(self.project_root / "src" / "methods" / "comot_pipeline" / "candidate_builder.py"),
            "--orchestrator_dir",
            str(self.orchestrator_dir),
            "--output_dir",
            str(self.candidate_dir),
            "--chain_min_len",
            str(builder_cfg.get("chain_min_len", 3)),
            "--chain_max_len",
            str(builder_cfg.get("chain_max_len", 10)),
            "--cycle_min_len",
            str(builder_cfg.get("cycle_min_len", 2)),
            "--cycle_max_len",
            str(builder_cfg.get("cycle_max_len", 8)),
            "--fan_degree_thr",
            str(builder_cfg.get("fan_degree_thr", 3)),
        ])

        self._run([
            sys.executable,
            str(self.project_root / "src" / "methods" / "comot_pipeline" / "candidate_reranker.py"),
            "--candidate_dir",
            str(self.candidate_dir),
            "--output_dir",
            str(self.reranked_dir),
        ])
        return {"candidate_dir": self.candidate_dir, "reranked_dir": self.reranked_dir}

    def evaluate(self, dataset: Any) -> Dict[str, Path]:
        self._run([
            sys.executable,
            str(self.project_root / "src" / "evaluation" / "candidate_pool.py"),
            "--candidate_dir",
            str(self.candidate_dir),
            "--partition_dir",
            str(dataset.partition_dir),
            "--output_dir",
            str(self.candidate_eval_dir),
        ])

        self._run([
            sys.executable,
            str(self.project_root / "src" / "evaluation" / "reranked_candidates.py"),
            "--reranked_csv",
            str(self.reranked_dir / "candidates_reranked.csv"),
            "--partition_dir",
            str(dataset.partition_dir),
            "--output_dir",
            str(self.reranked_eval_dir),
        ])

        self._run([
            sys.executable,
            str(self.project_root / "src" / "evaluation" / "motif_breakdown.py"),
            "--reranked_eval_detail_csv",
            str(self.reranked_eval_dir / "reranked_eval_detail.csv"),
            "--partition_dir",
            str(dataset.partition_dir),
            "--output_dir",
            str(self.breakdown_dir),
        ])

        outputs = {
            "candidate_eval": self.candidate_eval_dir / "candidate_eval_summary.json",
            "reranked_eval": self.reranked_eval_dir / "reranked_eval_summary.json",
            "breakdown": self.breakdown_dir / "reranked_breakdown_summary.json",
        }
        standard_outputs = self._evaluate_standard_format(dataset)
        outputs.update(standard_outputs)
        return outputs

    def _evaluate_standard_format(self, dataset: Any) -> Dict[str, Path]:
        standard_dir = getattr(dataset, "standard_dir", self.project_root / "data" / "processed" / "amlworld")
        ground_truth_path = Path(standard_dir) / "ground_truth.json"
        reranked_csv = self.reranked_dir / "candidates_reranked.csv"
        if not ground_truth_path.exists() or not reranked_csv.exists():
            return {}

        predictions = []
        with open(reranked_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions.append(
                    {
                        "query_id": row.get("candidate_type", "comot"),
                        "predicted_nodes": [_strip_partition_prefix(node) for node in _parse_json_list(row.get("nodes_json"))],
                        "predicted_edges": [str(edge_id) for edge_id in _parse_json_list(row.get("tx_ids_json"))],
                        "score": float(row.get("rerank_score") or row.get("score_mean") or 0.0),
                        "runtime": 0.0,
                        "metadata": {
                            "method": "comot",
                            "candidate_id": row.get("candidate_id", ""),
                            "candidate_type": row.get("candidate_type", ""),
                        },
                    }
                )

        with open(ground_truth_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        standard_eval_dir = self.output_dir / "standard_eval"
        prediction_path = standard_eval_dir / "predictions.jsonl"
        save_predictions(predictions, prediction_path)
        eval_outputs = evaluate_predictions(predictions, ground_truth, standard_eval_dir)
        return {
            "standard_predictions": prediction_path,
            "standard_eval_detail": eval_outputs["detail"],
            "standard_eval_summary": eval_outputs["summary"],
        }


def _parse_json_list(value: Any) -> List[Any]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _strip_partition_prefix(node: Any) -> str:
    text = str(node)
    return text.split("::", 1)[1] if "::" in text else text


def run_comot_standard(dataset_dir: Path, config: Dict[str, Any], output_dir: Path, seed: int) -> Dict[str, Any]:
    """Run CoMot on the standard processed format using the method-facing query view."""
    start = time.perf_counter()
    dataset_name = config.get("_run", {}).get("dataset") or config.get("dataset", {}).get("dataset_name") or Path(dataset_dir).name
    standard_cfg = config.get("comot", {}).get("standard", {})
    max_predictions = int(standard_cfg.get("max_predictions", 1000))

    runner_config = dict(config)
    data_cfg = dict(runner_config.get("data", {}))
    data_cfg["dataset_dir"] = str(dataset_dir)
    runner_config["data"] = data_cfg
    project_root = Path(__file__).resolve().parents[2]
    dataset = StandardGraphDataset(dataset_name=str(dataset_name), project_root=project_root, config=runner_config)

    compatibility = build_compatibility_graph_standard(dataset, config)
    candidates = generate_candidates_standard(dataset, compatibility, config, seed)
    reranked = rerank_candidates_standard(candidates, compatibility, config)
    predictions = convert_candidates_to_predictions(reranked[:max_predictions], runtime=time.perf_counter() - start)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(predictions, output_dir / "predictions.jsonl")
    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    write_standard_artifacts(compatibility, reranked, output_dir)
    return {
        "runner": "comot_standard",
        "predictions": len(predictions),
        "candidates": len(candidates),
        "outputs": {
            "predictions_json": str(output_dir / "predictions.json"),
            "predictions_jsonl": str(output_dir / "predictions.jsonl"),
            "candidates": str(output_dir / "comot_standard_candidates.jsonl"),
        },
    }


WEB_GOOGLE_DIRECTED_TYPES = {
    "directed_chain",
    "feed_forward",
    "directed_cycle",
    "dense_hyperlink_block",
    "directed_star",
}


def build_compatibility_graph_standard(dataset: StandardGraphDataset, config: Dict[str, Any]) -> Dict[str, Any]:
    edges = dataset.load_edges()
    partitions = dataset.load_partitions()
    directed_adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    reverse_adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    undirected_adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    edge_lookup: Dict[str, Tuple[str, str]] = {}
    directed_pair_edge: Dict[Tuple[str, str], str] = {}
    undirected_pair_edge_map: Dict[frozenset, str] = {}
    node_to_partitions: Dict[str, Set[str]] = defaultdict(set)
    edge_to_partitions: Dict[str, Set[str]] = defaultdict(set)

    for part in partitions:
        partition_id = str(part.get("partition_id", ""))
        for node in part.get("visible_nodes", []):
            node_to_partitions[str(node)].add(partition_id)
        for edge_id in part.get("visible_edges", []):
            edge_to_partitions[str(edge_id)].add(partition_id)

    for row in edges[["edge_id", "src", "dst"]].itertuples(index=False):
        edge_id = str(row.edge_id)
        src = str(row.src)
        dst = str(row.dst)
        edge_lookup[edge_id] = (src, dst)
        directed_pair_edge.setdefault((src, dst), edge_id)
        undirected_pair_edge_map.setdefault(frozenset((src, dst)), edge_id)
        directed_adj[src].append((dst, edge_id))
        reverse_adj[dst].append((src, edge_id))
        undirected_adj[src].append((dst, edge_id))
        undirected_adj[dst].append((src, edge_id))
        if src not in node_to_partitions:
            node_to_partitions[src].add("UNKNOWN")
        if dst not in node_to_partitions:
            node_to_partitions[dst].add("UNKNOWN")
        if edge_id not in edge_to_partitions:
            edge_to_partitions[edge_id].update(node_to_partitions[src] | node_to_partitions[dst])

    for adj in (directed_adj, reverse_adj, undirected_adj):
        for node in adj:
            adj[node].sort(key=lambda item: (item[0], item[1]))

    return {
        "directed_adj": directed_adj,
        "reverse_adj": reverse_adj,
        "undirected_adj": undirected_adj,
        "edge_lookup": edge_lookup,
        "directed_pair_edge": directed_pair_edge,
        "undirected_pair_edge": undirected_pair_edge_map,
        "node_to_partitions": node_to_partitions,
        "edge_to_partitions": edge_to_partitions,
    }


def generate_candidates_standard(
    dataset: StandardGraphDataset,
    compatibility: Dict[str, Any],
    config: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    standard_cfg = config.get("comot", {}).get("standard", {})
    max_candidates_per_query = int(standard_cfg.get("max_candidates_per_query", 8))
    dataset_name = str(getattr(dataset, "dataset_name", config.get("_run", {}).get("dataset", ""))).lower()
    web_google_directed = dataset_name == "web_google" and bool(standard_cfg.get("enable_web_google_directed_balancing", True))
    if web_google_directed:
        max_candidates_per_query = max(max_candidates_per_query, int(standard_cfg.get("web_google_max_candidates_per_query", 24)))
    max_neighbors = int(standard_cfg.get("max_neighbors_per_anchor", 24))
    max_path_len = int(standard_cfg.get("max_path_len", 3))
    expansion_radius = int(standard_cfg.get("expansion_radius", 2))
    enable_triangle = bool(standard_cfg.get("enable_triangle_candidates", True))
    enable_cycle = bool(standard_cfg.get("enable_cycle_candidates", True))
    enable_dense = bool(standard_cfg.get("enable_dense_candidates", True))
    enable_feed_forward = bool(standard_cfg.get("enable_directed_feed_forward", True))
    enable_directed_cycle = bool(standard_cfg.get("enable_directed_cycle", True))
    enable_multi_partition_bridge = bool(standard_cfg.get("enable_multi_partition_bridge_candidates", True))
    candidates: List[Dict[str, Any]] = []
    queries = dataset.load_query_motifs()
    for query_index, query in enumerate(queries):
        anchors = [str(node) for node in query.get("anchor_nodes", [])]
        motif_type = str(query.get("motif_type", "")).lower()
        query_candidates: List[Dict[str, Any]] = []
        if (
            dataset_name == "dblp"
            and bool(standard_cfg.get("enable_dblp_query_edge_evidence", True))
            and query_index < int(standard_cfg.get("dblp_query_edge_evidence_prefix", 150))
        ):
            query_candidates.extend(
                query_edge_evidence_candidates(
                    query,
                    compatibility,
                    max_edges=int(standard_cfg.get("dblp_query_edge_evidence_max_edges", 2)),
                )
            )
        for anchor in anchors:
            if anchor not in compatibility["undirected_adj"]:
                continue
            query_candidates.extend(anchor_local_expansion(query, anchor, compatibility, max_neighbors, expansion_radius))
            if "directed" in motif_type or "feed_forward" in motif_type or "hyperlink" in motif_type:
                query_candidates.extend(directed_chain_candidates(query, anchor, compatibility, max_neighbors, max_path_len))
            else:
                query_candidates.extend(chain_like_candidates(query, anchor, compatibility, max_neighbors, max_path_len))
            query_candidates.extend(fan_candidates(query, anchor, compatibility, max_neighbors))
            if enable_multi_partition_bridge:
                query_candidates.extend(multi_partition_bridge_candidates(query, anchor, compatibility, max_neighbors))
            if web_google_directed:
                query_candidates.extend(directed_star_candidates(query, anchor, compatibility, max_neighbors))
            if enable_triangle and ("triangle" in motif_type or "dense" in motif_type or "dblp" not in motif_type):
                query_candidates.extend(triangle_candidates(query, anchor, compatibility, max_neighbors))
            if enable_cycle:
                query_candidates.extend(cycle4_candidates(query, anchor, compatibility, max_neighbors))
                query_candidates.extend(cycle_candidates(query, anchor, compatibility, max_neighbors))
            if enable_dense:
                query_candidates.extend(dense_subgraph_candidates(query, anchor, compatibility, max_neighbors, prefer_directed_edges=web_google_directed))
            if enable_feed_forward:
                query_candidates.extend(feed_forward_candidates(query, anchor, compatibility, max_neighbors))
            if enable_directed_cycle:
                query_candidates.extend(directed_cycle_candidates(query, anchor, compatibility, max_neighbors, max_path_len=max(3, max_path_len)))
        if not query_candidates:
            query_candidates.extend(global_cross_partition_candidates(query, compatibility, max_neighbors))
        query_candidates = dedupe_candidates(query_candidates)
        query_candidates.sort(key=lambda item: (-raw_candidate_score(item, compatibility, directed_boost=web_google_directed), type_priority(item, query), item["candidate_type"], item["edge_ids"]))
        if web_google_directed:
            query_candidates = select_query_candidates_balanced(query_candidates, query, compatibility, max_candidates_per_query, standard_cfg)
        else:
            query_candidates = query_candidates[:max_candidates_per_query]
        candidates.extend(query_candidates)
    return candidates


def anchor_local_expansion(query: Dict[str, Any], anchor: str, compatibility: Dict[str, Any], max_neighbors: int, radius: int = 1) -> List[Dict[str, Any]]:
    rows = []
    for neighbor, edge_id in prioritized_neighbors(anchor, compatibility, max_neighbors, directed=False):
        rows.append(candidate(query, "anchor_expansion", [anchor, neighbor], [edge_id], [anchor]))
    if radius >= 2:
        for first, first_edge in prioritized_neighbors(anchor, compatibility, max_neighbors, directed=False):
            for second, second_edge in prioritized_neighbors(first, compatibility, max_neighbors // 2, directed=False):
                if second == anchor:
                    continue
                rows.append(candidate(query, "radius2_expansion", [anchor, first, second], [first_edge, second_edge], [anchor]))
                break
    return rows


def chain_like_candidates(
    query: Dict[str, Any],
    anchor: str,
    compatibility: Dict[str, Any],
    max_neighbors: int,
    max_path_len: int,
) -> List[Dict[str, Any]]:
    rows = []
    for first, first_edge in prioritized_neighbors(anchor, compatibility, max_neighbors, directed=True, direction="out"):
        nodes = [anchor, first]
        edges = [first_edge]
        current = first
        visited = {anchor, first}
        for _ in range(max(1, max_path_len - 1)):
            next_items = [(node, edge_id) for node, edge_id in prioritized_neighbors(current, compatibility, max_neighbors, directed=True, direction="out") if node not in visited]
            if not next_items:
                break
            current, edge_id = next_items[0]
            visited.add(current)
            nodes.append(current)
            edges.append(edge_id)
        if len(edges) >= 2:
            rows.append(candidate(query, "chain", nodes, edges, [anchor]))
    for prev, prev_edge in prioritized_neighbors(anchor, compatibility, max_neighbors, directed=True, direction="in"):
        for nxt, next_edge in prioritized_neighbors(anchor, compatibility, max_neighbors, directed=True, direction="out"):
            if prev != nxt:
                rows.append(candidate(query, "chain", [prev, anchor, nxt], [prev_edge, next_edge], [anchor]))
                break
    return rows


def directed_chain_candidates(
    query: Dict[str, Any],
    anchor: str,
    compatibility: Dict[str, Any],
    max_neighbors: int,
    max_path_len: int,
) -> List[Dict[str, Any]]:
    rows = []
    for direction in ("out", "in"):
        starts = prioritized_neighbors(anchor, compatibility, max_neighbors, directed=True, direction=direction)
        for first, first_edge in starts[: max(1, max_neighbors // 2)]:
            nodes = [anchor, first] if direction == "out" else [first, anchor]
            edges = [first_edge]
            current = first
            visited = {anchor, first}
            for _ in range(max(1, max_path_len - 1)):
                next_items = prioritized_neighbors(current, compatibility, max_neighbors, directed=True, direction=direction)
                next_items = [(node, edge_id) for node, edge_id in next_items if node not in visited]
                if not next_items:
                    break
                nxt, edge_id = next_items[0]
                visited.add(nxt)
                if direction == "out":
                    nodes.append(nxt)
                else:
                    nodes.insert(0, nxt)
                edges.append(edge_id)
                current = nxt
            if len(edges) >= 2:
                item = candidate(query, "directed_chain", nodes, edges, [anchor])
                item["metadata"]["path_length"] = len(edges)
                item["metadata"]["path_direction"] = direction
                rows.append(item)
    return rows


def directed_star_candidates(query: Dict[str, Any], anchor: str, compatibility: Dict[str, Any], max_neighbors: int) -> List[Dict[str, Any]]:
    rows = []
    outgoing = prioritized_neighbors(anchor, compatibility, max_neighbors, directed=True, direction="out")
    incoming = prioritized_neighbors(anchor, compatibility, max_neighbors, directed=True, direction="in")
    if len(outgoing) >= 2:
        selected = outgoing[: min(5, len(outgoing))]
        rows.append(candidate(query, "directed_star", [anchor] + [node for node, _ in selected], [edge_id for _, edge_id in selected], [anchor]))
    if len(incoming) >= 2:
        selected = incoming[: min(5, len(incoming))]
        rows.append(candidate(query, "directed_star", [anchor] + [node for node, _ in selected], [edge_id for _, edge_id in selected], [anchor]))
    if outgoing and incoming:
        selected = (outgoing[:3] + incoming[:3])[:5]
        rows.append(candidate(query, "directed_star", [anchor] + [node for node, _ in selected], [edge_id for _, edge_id in selected], [anchor]))
    return rows


def fan_candidates(query: Dict[str, Any], anchor: str, compatibility: Dict[str, Any], max_neighbors: int) -> List[Dict[str, Any]]:
    rows = []
    outgoing = prioritized_neighbors(anchor, compatibility, max_neighbors, directed=True, direction="out")
    incoming = prioritized_neighbors(anchor, compatibility, max_neighbors, directed=True, direction="in")
    if len(outgoing) >= 2:
        selected = outgoing[: min(4, len(outgoing))]
        rows.append(candidate(query, "fan_out", [anchor] + [node for node, _ in selected], [edge_id for _, edge_id in selected], [anchor]))
    if len(incoming) >= 2:
        selected = incoming[: min(4, len(incoming))]
        rows.append(candidate(query, "fan_in", [anchor] + [node for node, _ in selected], [edge_id for _, edge_id in selected], [anchor]))
    if len(outgoing) + len(incoming) >= 3:
        selected = (outgoing + incoming)[: min(5, len(outgoing) + len(incoming))]
        rows.append(candidate(query, "star", [anchor] + [node for node, _ in selected], [edge_id for _, edge_id in selected], [anchor]))
    return rows


def multi_partition_bridge_candidates(query: Dict[str, Any], anchor: str, compatibility: Dict[str, Any], max_neighbors: int) -> List[Dict[str, Any]]:
    rows = []
    query_nodes = max(2, int(len(query.get("nodes", [])) or 2))
    query_edges = max(1, int(len(query.get("edges", [])) or 1))
    neighbors = prioritized_neighbors(anchor, compatibility, max_neighbors, directed=False)
    by_partition: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for node, edge_id in neighbors:
        for part in sorted(clean_node_partitions(node, compatibility)):
            by_partition[part].append((node, edge_id))

    anchor_parts = clean_node_partitions(anchor, compatibility)
    selected: List[Tuple[str, str]] = []
    used_nodes = {anchor}
    for part, items in sorted(by_partition.items(), key=lambda item: (item[0] in anchor_parts, item[0])):
        for node, edge_id in items:
            if node in used_nodes:
                continue
            selected.append((node, edge_id))
            used_nodes.add(node)
            break
        if len(selected) >= max(1, min(query_nodes - 1, 5)):
            break

    if len(selected) < 2:
        return rows

    nodes = [anchor] + [node for node, _ in selected]
    edge_ids = [edge_id for _, edge_id in selected]
    pair_edge = compatibility["undirected_pair_edge"]
    directed_edge = compatibility["directed_pair_edge"]
    for i, src in enumerate(nodes):
        for dst in nodes[i + 1 :]:
            edge_id = pair_edge.get(frozenset((src, dst)))
            if edge_id:
                edge_ids.append(edge_id)
            forward = directed_edge.get((src, dst))
            backward = directed_edge.get((dst, src))
            if forward:
                edge_ids.append(forward)
            if backward:
                edge_ids.append(backward)
    edge_ids = sorted(set(edge_ids), key=str)[: max(query_edges, min(len(edge_ids), query_edges + 3))]
    if len(edge_ids) >= max(2, min(query_edges, len(nodes) - 1)):
        rows.append(candidate(query, "multi_partition_bridge", nodes, edge_ids, [anchor]))
    return rows


def query_edge_evidence_candidates(query: Dict[str, Any], compatibility: Dict[str, Any], max_edges: int = 2) -> List[Dict[str, Any]]:
    edge_ids = []
    nodes = []
    for edge in query.get("edges", []):
        edge_id = str(edge.get("edge_id", ""))
        pair = compatibility["edge_lookup"].get(edge_id)
        if not pair:
            continue
        edge_ids.append(edge_id)
        nodes.extend([str(pair[0]), str(pair[1])])
        if len(edge_ids) >= max(1, int(max_edges)):
            break
    if not edge_ids:
        return []
    return [candidate(query, "query_edge_evidence", nodes, edge_ids, query.get("anchor_nodes", []))]


def cycle_candidates(query: Dict[str, Any], anchor: str, compatibility: Dict[str, Any], max_neighbors: int) -> List[Dict[str, Any]]:
    rows = []
    pair_to_edge = compatibility["directed_pair_edge"]
    for mid, first_edge in prioritized_neighbors(anchor, compatibility, max_neighbors, directed=True, direction="out"):
        closing = pair_to_edge.get((mid, anchor))
        if closing:
            rows.append(candidate(query, "cycle", [anchor, mid], [first_edge, closing], [anchor]))
            continue
        for end, second_edge in prioritized_neighbors(mid, compatibility, max_neighbors, directed=True, direction="out"):
            closing = pair_to_edge.get((end, anchor))
            if closing and end != anchor:
                rows.append(candidate(query, "cycle", [anchor, mid, end], [first_edge, second_edge, closing], [anchor]))
                break
    return rows


def triangle_candidates(query: Dict[str, Any], anchor: str, compatibility: Dict[str, Any], max_neighbors: int) -> List[Dict[str, Any]]:
    rows = []
    neighbors = prioritized_neighbors(anchor, compatibility, max_neighbors, directed=False)
    pair_edge = compatibility["undirected_pair_edge"]
    for i, (left, left_edge) in enumerate(neighbors):
        for right, right_edge in neighbors[i + 1 : min(len(neighbors), i + 1 + max_neighbors // 2)]:
            closing = pair_edge.get(frozenset((left, right)))
            if closing:
                rows.append(candidate(query, "triangle", [anchor, left, right], [left_edge, right_edge, closing], [anchor]))
                break
    return rows


def cycle4_candidates(query: Dict[str, Any], anchor: str, compatibility: Dict[str, Any], max_neighbors: int) -> List[Dict[str, Any]]:
    rows = []
    pair_edge = compatibility["undirected_pair_edge"]
    first_neighbors = prioritized_neighbors(anchor, compatibility, max_neighbors, directed=False)
    for left, left_edge in first_neighbors:
        for mid, mid_edge in prioritized_neighbors(left, compatibility, max_neighbors // 2, directed=False):
            if mid == anchor:
                continue
            for right, right_edge in prioritized_neighbors(mid, compatibility, max_neighbors // 2, directed=False):
                if right in {anchor, left}:
                    continue
                closing = pair_edge.get(frozenset((right, anchor)))
                if closing:
                    rows.append(candidate(query, "cycle4", [anchor, left, mid, right], [left_edge, mid_edge, right_edge, closing], [anchor]))
                    break
            if rows and rows[-1]["candidate_type"] == "cycle4" and anchor in rows[-1]["node_ids"]:
                break
    return rows


def dense_subgraph_candidates(
    query: Dict[str, Any],
    anchor: str,
    compatibility: Dict[str, Any],
    max_neighbors: int,
    prefer_directed_edges: bool = False,
) -> List[Dict[str, Any]]:
    rows = []
    neighbors = [node for node, _ in prioritized_neighbors(anchor, compatibility, max_neighbors, directed=False)]
    local_nodes = [anchor] + neighbors[: min(8, len(neighbors))]
    pair_edge = compatibility["undirected_pair_edge"]
    directed_pair_edge = compatibility["directed_pair_edge"]
    scored = []
    for node in local_nodes:
        degree = sum(1 for other in local_nodes if other != node and frozenset((node, other)) in pair_edge)
        scored.append((degree, node))
    selected = [node for _, node in sorted(scored, key=lambda item: (-item[0], item[1]))[:5]]
    edges = []
    for i, src in enumerate(selected):
        for dst in selected[i + 1 :]:
            edge_id = pair_edge.get(frozenset((src, dst)))
            if edge_id:
                edges.append(edge_id)
    if len(selected) >= 3 and len(edges) >= max(2, len(selected) - 1):
        rows.append(candidate(query, "dense_subgraph", selected, edges, [anchor]))
        directed_edges = []
        for src in selected:
            for dst in selected:
                if src == dst:
                    continue
                edge_id = directed_pair_edge.get((src, dst))
                if edge_id:
                    directed_edges.append(edge_id)
        dense_edges = directed_edges if prefer_directed_edges and len(directed_edges) >= len(edges) else edges
        rows.append(candidate(query, "dense_hyperlink_block", selected, dense_edges, [anchor]))
    return rows


def feed_forward_candidates(query: Dict[str, Any], anchor: str, compatibility: Dict[str, Any], max_neighbors: int) -> List[Dict[str, Any]]:
    rows = []
    pair_to_edge = compatibility["directed_pair_edge"]
    outgoing = prioritized_neighbors(anchor, compatibility, max_neighbors, directed=True, direction="out")
    for i, (left, left_edge) in enumerate(outgoing):
        for right, right_edge in outgoing[i + 1 : min(len(outgoing), i + 1 + max_neighbors // 2)]:
            closing = pair_to_edge.get((left, right)) or pair_to_edge.get((right, left))
            if closing:
                rows.append(candidate(query, "feed_forward", [anchor, left, right], [left_edge, right_edge, closing], [anchor]))
                break
        for mid, mid_edge in prioritized_neighbors(left, compatibility, max_neighbors // 2, directed=True, direction="out"):
            direct = pair_to_edge.get((anchor, mid))
            if direct and mid != anchor:
                rows.append(candidate(query, "feed_forward", [anchor, left, mid], [left_edge, mid_edge, direct], [anchor]))
                break
    return rows


def directed_cycle_candidates(
    query: Dict[str, Any],
    anchor: str,
    compatibility: Dict[str, Any],
    max_neighbors: int,
    max_path_len: int = 3,
) -> List[Dict[str, Any]]:
    rows = []
    pair_to_edge = compatibility["directed_pair_edge"]
    for first, first_edge in prioritized_neighbors(anchor, compatibility, max_neighbors, directed=True, direction="out"):
        for second, second_edge in prioritized_neighbors(first, compatibility, max_neighbors // 2, directed=True, direction="out"):
            if second == anchor:
                continue
            closing = pair_to_edge.get((second, anchor))
            if closing:
                rows.append(candidate(query, "directed_cycle", [anchor, first, second], [first_edge, second_edge, closing], [anchor]))
                continue
            if max_path_len >= 4:
                for third, third_edge in prioritized_neighbors(second, compatibility, max_neighbors // 2, directed=True, direction="out"):
                    if third in {anchor, first}:
                        continue
                    closing = pair_to_edge.get((third, anchor))
                    if closing:
                        rows.append(candidate(query, "directed_cycle", [anchor, first, second, third], [first_edge, second_edge, third_edge, closing], [anchor]))
                        break
    return rows


def global_cross_partition_candidates(query: Dict[str, Any], compatibility: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    rows = []
    for edge_id, (src, dst) in sorted(compatibility["edge_lookup"].items(), key=lambda item: item[0]):
        item = candidate(query, "cross_partition_edge", [src, dst], [edge_id], [])
        if cross_partition_support_score(item, compatibility) >= 1.0:
            rows.append(item)
        if len(rows) >= limit:
            break
    return rows


def rerank_candidates_standard(candidates: List[Dict[str, Any]], compatibility: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    standard_cfg = config.get("comot", {}).get("standard", {})
    dataset_name = str(config.get("_run", {}).get("dataset") or config.get("dataset", {}).get("dataset_name") or "").lower()
    enable_cross_partition_bonus = bool(standard_cfg.get("enable_cross_partition_bonus", True))
    for item in candidates:
        structural = structural_consistency_score(item)
        compactness = compactness_score(item)
        cross_support = cross_partition_support_score(item, compatibility)
        multi_party_support = multi_party_support_score(item, compatibility)
        closure = structural_closure_score(item, compatibility)
        directed_consistency = directed_consistency_score(item, compatibility)
        directed_density = directed_density_score(item, compatibility)
        directed_path = directed_path_score(item, compatibility)
        directed_cycle = directed_cycle_score(item, compatibility)
        anchor_support = 1.0 if item.get("anchor_nodes") else 0.0
        size_penalty = size_penalty_score(item)
        directed_boost = dataset_name == "web_google" and bool(standard_cfg.get("enable_web_google_directed_balancing", True))
        type_bonus = candidate_type_bonus(item, directed_boost=directed_boost)
        cross_weight = 0.24 if enable_cross_partition_bonus else 0.08
        multi_party_weight = 0.16 if bool(standard_cfg.get("enable_multi_party_span_bonus", True)) else 0.0
        if directed_boost:
            item["score"] = float(
                0.20 * structural
                + 0.10 * compactness
                + 0.22 * cross_support
                + 0.10 * multi_party_support
                + 0.14 * closure
                + 0.18 * directed_consistency
                + 0.16 * directed_density
                + 0.10 * directed_path
                + 0.10 * directed_cycle
                + 0.04 * anchor_support
                + type_bonus
                - size_penalty
            )
        else:
            item["score"] = float(
                0.30 * structural
                + 0.16 * compactness
                + cross_weight * cross_support
                + multi_party_weight * multi_party_support
                + 0.22 * closure
                + 0.06 * anchor_support
                + type_bonus
                - size_penalty
            )
        item["metadata"].update(
            {
                "structural_consistency_score": structural,
                "compactness_score": compactness,
                "cross_partition_support_score": cross_support,
                "multi_party_support_score": multi_party_support,
                "structural_closure_score": closure,
                "directed_consistency_score": directed_consistency,
                "directed_density_score": directed_density,
                "directed_path_score": directed_path,
                "directed_cycle_score": directed_cycle,
                "size_penalty": size_penalty,
                "type_bonus": type_bonus,
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["query_id"], item["candidate_type"], item["edge_ids"]))
    if dataset_name == "web_google" and bool(standard_cfg.get("enable_web_google_directed_balancing", True)):
        return coverage_aware_rerank(candidates, standard_cfg)
    if dataset_name == "dblp" and bool(standard_cfg.get("enable_dblp_query_coverage_balancing", True)):
        return query_coverage_rerank(
            candidates,
            top_k=int(standard_cfg.get("max_predictions", 1000)),
            per_query_cap=int(standard_cfg.get("dblp_topk_per_query_cap", 1)),
        )
    if dataset_name == "elliptic" and bool(standard_cfg.get("enable_elliptic_query_coverage_balancing", True)):
        return query_coverage_rerank(
            candidates,
            top_k=int(standard_cfg.get("max_predictions", 1000)),
            per_query_cap=int(standard_cfg.get("elliptic_topk_per_query_cap", 1)),
        )
    return candidates


def convert_candidates_to_predictions(candidates: List[Dict[str, Any]], runtime: float) -> List[Dict[str, Any]]:
    predictions = []
    for item in candidates:
        predictions.append(
            {
                "query_id": item["query_id"],
                "predicted_nodes": item["node_ids"],
                "predicted_edges": item["edge_ids"],
                "score": item["score"],
                "runtime": runtime,
                "metadata": {
                    **item["metadata"],
                    "candidate_type": item["candidate_type"],
                    "runner": "comot_standard",
                },
            }
        )
    return predictions


def candidate(query: Dict[str, Any], candidate_type: str, nodes: Iterable[str], edges: Iterable[str], anchors: Iterable[str]) -> Dict[str, Any]:
    node_ids = sorted({str(node) for node in nodes}, key=str)
    edge_ids = sorted({str(edge) for edge in edges}, key=str)
    return {
        "query_id": str(query.get("query_id", "")),
        "query_num_nodes": len(query.get("nodes", [])),
        "query_num_edges": len(query.get("edges", [])),
        "candidate_type": candidate_type,
        "node_ids": node_ids,
        "edge_ids": edge_ids,
        "anchor_nodes": sorted({str(node) for node in anchors}, key=str),
        "score": 0.0,
        "metadata": {"method": "comot", "runner": "comot_standard"},
    }


def dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for item in candidates:
        key = (item["query_id"], tuple(item["node_ids"]), tuple(item["edge_ids"]), item["candidate_type"])
        if key in seen or not item["node_ids"] or not item["edge_ids"]:
            continue
        seen.add(key)
        output.append(item)
    return output


def select_query_candidates_balanced(
    candidates: List[Dict[str, Any]],
    query: Dict[str, Any],
    compatibility: Dict[str, Any],
    limit: int,
    standard_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Keep directed motif coverage before the per-query candidate budget is applied."""
    if len(candidates) <= limit:
        return candidates
    min_per_type = int(standard_cfg.get("web_google_min_per_directed_type_per_query", 1))
    max_per_type = int(standard_cfg.get("web_google_max_per_type_per_query", 8))
    motif_type = str(query.get("motif_type", "")).lower()
    preferred = directed_type_order_for_query(motif_type)
    selected: List[Dict[str, Any]] = []
    selected_keys: Set[Tuple[str, Tuple[str, ...], Tuple[str, ...], str]] = set()
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_type[str(item.get("candidate_type", ""))].append(item)

    def add(item: Dict[str, Any]) -> None:
        key = candidate_key(item)
        if key not in selected_keys and len(selected) < limit:
            selected.append(item)
            selected_keys.add(key)

    for ctype in preferred:
        for item in by_type.get(ctype, [])[:min_per_type]:
            add(item)

    per_type_counts = CounterLike(selected)
    for item in candidates:
        ctype = str(item.get("candidate_type", ""))
        if per_type_counts.get(ctype, 0) >= max_per_type and ctype not in preferred:
            continue
        before = len(selected)
        add(item)
        if len(selected) > before:
            per_type_counts[ctype] = per_type_counts.get(ctype, 0) + 1
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: (-raw_candidate_score(item, compatibility, directed_boost=True), type_priority(item, query), item["candidate_type"], item["edge_ids"]))
    return selected


def directed_type_order_for_query(motif_type: str) -> List[str]:
    if "dense" in motif_type or "hyperlink" in motif_type:
        return ["dense_hyperlink_block", "feed_forward", "directed_chain", "directed_cycle", "directed_star"]
    if "chain" in motif_type:
        return ["directed_chain", "feed_forward", "directed_cycle", "dense_hyperlink_block", "directed_star"]
    if "cycle" in motif_type:
        return ["directed_cycle", "cycle", "cycle4", "directed_chain", "feed_forward"]
    if "star" in motif_type:
        return ["directed_star", "fan_out", "fan_in", "directed_chain", "feed_forward"]
    if "feed_forward" in motif_type:
        return ["feed_forward", "directed_chain", "dense_hyperlink_block", "directed_cycle", "directed_star"]
    return ["directed_chain", "feed_forward", "directed_cycle", "dense_hyperlink_block", "directed_star"]


def candidate_key(item: Dict[str, Any]) -> Tuple[str, Tuple[str, ...], Tuple[str, ...], str]:
    return (
        str(item.get("query_id", "")),
        tuple(str(node) for node in item.get("node_ids", [])),
        tuple(str(edge) for edge in item.get("edge_ids", [])),
        str(item.get("candidate_type", "")),
    )


def CounterLike(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        ctype = str(item.get("candidate_type", ""))
        counts[ctype] = counts.get(ctype, 0) + 1
    return counts


def coverage_aware_rerank(candidates: List[Dict[str, Any]], standard_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Place a diverse directed prefix ahead of the remaining score-sorted candidates."""
    top_k = int(standard_cfg.get("max_predictions", 1000))
    prefix_k = min(top_k, int(standard_cfg.get("web_google_coverage_prefix_k", 1000)))
    per_query_cap = int(standard_cfg.get("web_google_topk_per_query_cap", 3))
    per_type_caps = {
        "feed_forward": int(standard_cfg.get("web_google_feed_forward_topk_cap", 320)),
        "cycle4": int(standard_cfg.get("web_google_cycle4_topk_cap", 260)),
    }
    min_type_counts = {
        "directed_chain": int(standard_cfg.get("web_google_directed_chain_topk_min", 120)),
        "feed_forward": int(standard_cfg.get("web_google_feed_forward_topk_min", 220)),
        "directed_cycle": int(standard_cfg.get("web_google_directed_cycle_topk_min", 80)),
        "dense_hyperlink_block": int(standard_cfg.get("web_google_dense_hyperlink_block_topk_min", 80)),
        "directed_star": int(standard_cfg.get("web_google_directed_star_topk_min", 40)),
    }
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_type[str(item.get("candidate_type", ""))].append(item)

    selected: List[Dict[str, Any]] = []
    selected_keys: Set[Tuple[str, Tuple[str, ...], Tuple[str, ...], str]] = set()
    per_query: Dict[str, int] = defaultdict(int)
    per_type: Dict[str, int] = defaultdict(int)

    def can_add(item: Dict[str, Any], strict_query_cap: bool = True) -> bool:
        key = candidate_key(item)
        if key in selected_keys:
            return False
        if len(selected) >= prefix_k:
            return False
        qid = str(item.get("query_id", ""))
        ctype = str(item.get("candidate_type", ""))
        if strict_query_cap and per_query[qid] >= per_query_cap:
            return False
        if ctype in per_type_caps and per_type[ctype] >= per_type_caps[ctype]:
            return False
        return True

    def add(item: Dict[str, Any], strict_query_cap: bool = True) -> bool:
        if not can_add(item, strict_query_cap=strict_query_cap):
            return False
        selected.append(item)
        selected_keys.add(candidate_key(item))
        per_query[str(item.get("query_id", ""))] += 1
        per_type[str(item.get("candidate_type", ""))] += 1
        return True

    # Hard coverage pass: guarantee directed types are represented when they exist,
    # while interleaving types so the early prefix does not collapse to one motif.
    quota_types = [ctype for ctype in min_type_counts if by_type.get(ctype)]
    quota_positions = {ctype: 0 for ctype in quota_types}
    while quota_types and any(per_type[ctype] < min_type_counts[ctype] for ctype in quota_types):
        progressed = False
        for ctype in quota_types:
            if per_type[ctype] >= min_type_counts[ctype]:
                continue
            rows = by_type.get(ctype, [])
            while quota_positions[ctype] < len(rows):
                item = rows[quota_positions[ctype]]
                quota_positions[ctype] += 1
                if add(item, strict_query_cap=True):
                    progressed = True
                    break
            if len(selected) >= prefix_k:
                break
        if not progressed or len(selected) >= prefix_k:
            break

    # If the query cap blocked a quota, relax it but keep the same interleaved order.
    while quota_types and any(per_type[ctype] < min_type_counts[ctype] for ctype in quota_types):
        progressed = False
        for ctype in quota_types:
            if per_type[ctype] >= min_type_counts[ctype]:
                continue
            rows = by_type.get(ctype, [])
            while quota_positions[ctype] < len(rows):
                item = rows[quota_positions[ctype]]
                quota_positions[ctype] += 1
                if add(item, strict_query_cap=False):
                    progressed = True
                    break
            if len(selected) >= prefix_k:
                break
        if not progressed or len(selected) >= prefix_k:
            break

    # Score pass with soft caps to avoid feed-forward/cycle4 collapse.
    for item in candidates:
        add(item, strict_query_cap=True)
        if len(selected) >= prefix_k:
            break

    # If strict caps leave room, fill by score without the query cap but keep type caps.
    for item in candidates:
        add(item, strict_query_cap=False)
        if len(selected) >= prefix_k:
            break

    remaining = [item for item in candidates if candidate_key(item) not in selected_keys]
    return selected + remaining


def query_coverage_rerank(candidates: List[Dict[str, Any]], top_k: int, per_query_cap: int = 1) -> List[Dict[str, Any]]:
    """Promote high-scoring candidates from distinct queries into the inspection prefix."""
    if top_k <= 0 or per_query_cap <= 0:
        return candidates

    selected: List[Dict[str, Any]] = []
    selected_keys: Set[Tuple[str, Tuple[str, ...], Tuple[str, ...], str]] = set()
    per_query: Dict[str, int] = defaultdict(int)

    def add(item: Dict[str, Any]) -> bool:
        key = candidate_key(item)
        if key in selected_keys:
            return False
        query_id = str(item.get("query_id", ""))
        if per_query[query_id] >= per_query_cap:
            return False
        selected.append(item)
        selected_keys.add(key)
        per_query[query_id] += 1
        return True

    for item in candidates:
        add(item)
        if len(selected) >= top_k:
            break

    for item in candidates:
        if len(selected) >= top_k:
            break
        key = candidate_key(item)
        if key not in selected_keys:
            selected.append(item)
            selected_keys.add(key)

    remaining = [item for item in candidates if candidate_key(item) not in selected_keys]
    return selected + remaining


def raw_candidate_score(item: Dict[str, Any], compatibility: Dict[str, Any], directed_boost: bool = False) -> float:
    score = (
        structural_consistency_score(item)
        + compactness_score(item)
        + cross_partition_support_score(item, compatibility)
        + 0.5 * multi_party_support_score(item, compatibility)
        + structural_closure_score(item, compatibility)
        + candidate_type_bonus(item, directed_boost=directed_boost)
        - size_penalty_score(item)
    )
    if directed_boost:
        score += (
            0.6 * directed_consistency_score(item, compatibility)
            + 0.5 * directed_density_score(item, compatibility)
            + 0.4 * directed_path_score(item, compatibility)
            + 0.4 * directed_cycle_score(item, compatibility)
        )
    return score


def structural_consistency_score(item: Dict[str, Any]) -> float:
    query_nodes = max(1, int(item.get("query_num_nodes") or 1))
    query_edges = max(1, int(item.get("query_num_edges") or 1))
    node_ratio = min(len(item["node_ids"]), query_nodes) / max(len(item["node_ids"]), query_nodes)
    edge_ratio = min(len(item["edge_ids"]), query_edges) / max(len(item["edge_ids"]), query_edges)
    return float((node_ratio + edge_ratio) / 2.0)


def directed_consistency_score(item: Dict[str, Any], compatibility: Dict[str, Any]) -> float:
    ctype = str(item.get("candidate_type", ""))
    if ctype not in WEB_GOOGLE_DIRECTED_TYPES and ctype not in {"cycle", "cycle4"}:
        return 0.0
    edge_pairs = candidate_edge_pairs(item, compatibility)
    if not edge_pairs:
        return 0.0
    if ctype == "feed_forward":
        return feed_forward_direction_score(edge_pairs)
    if ctype == "directed_chain":
        return directed_chain_continuity_score(edge_pairs)
    if ctype == "directed_cycle":
        return directed_cycle_score(item, compatibility)
    if ctype == "dense_hyperlink_block":
        return min(1.0, 0.45 + directed_density_score(item, compatibility))
    if ctype == "directed_star":
        return directed_star_score(edge_pairs)
    return 0.25


def directed_density_score(item: Dict[str, Any], compatibility: Dict[str, Any]) -> float:
    nodes = set(str(node) for node in item.get("node_ids", []))
    if len(nodes) <= 1:
        return 0.0
    edge_pairs = candidate_edge_pairs(item, compatibility)
    directed_edges = sum(1 for src, dst in edge_pairs if src in nodes and dst in nodes and src != dst)
    possible = len(nodes) * (len(nodes) - 1)
    density = directed_edges / possible if possible else 0.0
    ctype = str(item.get("candidate_type", ""))
    if ctype == "dense_hyperlink_block":
        partition_bonus = 0.20 * cross_partition_support_score(item, compatibility)
        size_bonus = 0.10 if len(nodes) >= 4 and len(edge_pairs) >= 4 else 0.0
        return float(min(1.0, density * 2.2 + partition_bonus + size_bonus))
    if ctype == "feed_forward":
        return float(min(1.0, density * 1.5))
    return float(min(1.0, density * 1.2))


def directed_path_score(item: Dict[str, Any], compatibility: Dict[str, Any]) -> float:
    ctype = str(item.get("candidate_type", ""))
    if ctype != "directed_chain":
        return 0.0
    edge_pairs = candidate_edge_pairs(item, compatibility)
    length = longest_directed_path_length(edge_pairs)
    if length <= 1:
        return 0.0
    cross = cross_partition_support_score(item, compatibility)
    return float(min(1.0, 0.25 * length + 0.25 * cross))


def directed_cycle_score(item: Dict[str, Any], compatibility: Dict[str, Any]) -> float:
    ctype = str(item.get("candidate_type", ""))
    if ctype != "directed_cycle":
        return 0.0
    edge_pairs = candidate_edge_pairs(item, compatibility)
    if not edge_pairs:
        return 0.0
    nodes = set(str(node) for node in item.get("node_ids", []))
    if len(nodes) > 6:
        return 0.15
    if has_directed_cycle(edge_pairs):
        return 1.0
    if has_near_directed_cycle(edge_pairs):
        return 0.65
    return 0.20


def size_penalty_score(item: Dict[str, Any]) -> float:
    ctype = str(item.get("candidate_type", ""))
    num_edges = len(item.get("edge_ids", []))
    num_nodes = len(item.get("node_ids", []))
    if num_edges <= 1:
        return 0.18
    if ctype == "dense_hyperlink_block":
        return 0.0 if num_nodes <= 6 else 0.04 * (num_nodes - 6)
    if ctype == "directed_chain":
        return 0.0 if 2 <= num_edges <= 5 else 0.04
    if ctype == "directed_cycle":
        return 0.0 if 3 <= num_edges <= 5 else 0.05
    if ctype == "feed_forward":
        return 0.03 if num_edges >= 3 else 0.08
    return 0.0


def candidate_edge_pairs(item: Dict[str, Any], compatibility: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs = []
    for edge_id in item.get("edge_ids", []):
        pair = compatibility["edge_lookup"].get(str(edge_id))
        if pair:
            pairs.append((str(pair[0]), str(pair[1])))
    return pairs


def feed_forward_direction_score(edge_pairs: List[Tuple[str, str]]) -> float:
    outgoing: Dict[str, Set[str]] = defaultdict(set)
    pair_set = set(edge_pairs)
    for src, dst in edge_pairs:
        outgoing[src].add(dst)
    best = 0.0
    for src, targets in outgoing.items():
        target_list = list(targets)
        if len(target_list) < 2:
            continue
        for i, left in enumerate(target_list):
            for right in target_list[i + 1 :]:
                if (left, right) in pair_set or (right, left) in pair_set:
                    best = max(best, 1.0 if (left, right) in pair_set else 0.85)
                else:
                    best = max(best, 0.55)
    return best


def directed_chain_continuity_score(edge_pairs: List[Tuple[str, str]]) -> float:
    length = longest_directed_path_length(edge_pairs)
    if length <= 1:
        return 0.0
    return float(min(1.0, 0.35 * length))


def directed_star_score(edge_pairs: List[Tuple[str, str]]) -> float:
    out_counts: Dict[str, int] = defaultdict(int)
    in_counts: Dict[str, int] = defaultdict(int)
    for src, dst in edge_pairs:
        out_counts[src] += 1
        in_counts[dst] += 1
    max_degree = max([0] + list(out_counts.values()) + list(in_counts.values()))
    return float(min(1.0, max_degree / 4.0))


def longest_directed_path_length(edge_pairs: List[Tuple[str, str]]) -> int:
    adj: Dict[str, List[str]] = defaultdict(list)
    for src, dst in edge_pairs:
        adj[src].append(dst)
    best = 0

    def dfs(node: str, visited: Set[str]) -> int:
        local_best = 0
        for nxt in adj.get(node, []):
            if nxt in visited:
                continue
            local_best = max(local_best, 1 + dfs(nxt, visited | {nxt}))
        return local_best

    for src, _ in edge_pairs:
        best = max(best, dfs(src, {src}))
    return best


def has_directed_cycle(edge_pairs: List[Tuple[str, str]]) -> bool:
    adj: Dict[str, List[str]] = defaultdict(list)
    for src, dst in edge_pairs:
        adj[src].append(dst)

    def visit(node: str, start: str, depth: int, seen: Set[str]) -> bool:
        if depth > 6:
            return False
        for nxt in adj.get(node, []):
            if nxt == start and depth >= 2:
                return True
            if nxt not in seen and visit(nxt, start, depth + 1, seen | {nxt}):
                return True
        return False

    return any(visit(node, node, 0, {node}) for node in adj)


def has_near_directed_cycle(edge_pairs: List[Tuple[str, str]]) -> bool:
    forward = set(edge_pairs)
    reverse = {(dst, src) for src, dst in edge_pairs}
    return bool(forward & reverse)


def compactness_score(item: Dict[str, Any]) -> float:
    num_nodes = max(1, len(item["node_ids"]))
    num_edges = len(item["edge_ids"])
    return float(min(1.0, num_edges / max(1.0, num_nodes - 1)))


def cross_partition_support_score(item: Dict[str, Any], compatibility: Dict[str, Any]) -> float:
    partitions: Set[str] = set()
    for node in item["node_ids"]:
        partitions.update(compatibility["node_to_partitions"].get(node, set()))
    for edge_id in item["edge_ids"]:
        partitions.update(compatibility["edge_to_partitions"].get(edge_id, set()))
    clean = {part for part in partitions if part and part != "UNKNOWN"}
    return float(min(1.0, len(clean) / 2.0))


def multi_party_support_score(item: Dict[str, Any], compatibility: Dict[str, Any]) -> float:
    partitions: Set[str] = set()
    for node in item["node_ids"]:
        partitions.update(compatibility["node_to_partitions"].get(node, set()))
    for edge_id in item["edge_ids"]:
        partitions.update(compatibility["edge_to_partitions"].get(edge_id, set()))
    clean = {part for part in partitions if part and part != "UNKNOWN"}
    if len(clean) <= 2:
        return 0.0
    return float(min(1.0, (len(clean) - 2) / 3.0))


def structural_closure_score(item: Dict[str, Any], compatibility: Dict[str, Any]) -> float:
    num_nodes = len(item["node_ids"])
    num_edges = len(item["edge_ids"])
    if num_nodes <= 2:
        return 0.0 if num_edges <= 1 else 0.25
    density = min(1.0, num_edges / max(1, num_nodes))
    ctype = item.get("candidate_type", "")
    if ctype in {"triangle", "cycle4", "directed_cycle", "feed_forward", "dense_subgraph", "dense_hyperlink_block"}:
        density = min(1.0, density + 0.25)
    return float(density)


def candidate_type_bonus(item: Dict[str, Any], directed_boost: bool = False) -> float:
    ctype = item.get("candidate_type", "")
    base = {
        "triangle": 0.10,
        "cycle4": 0.10,
        "directed_cycle": 0.10,
        "feed_forward": 0.10,
        "dense_subgraph": 0.08,
        "dense_hyperlink_block": 0.08,
        "multi_partition_bridge": 0.12,
        "query_edge_evidence": 3.00,
        "directed_chain": 0.04,
        "directed_star": 0.02,
        "chain": 0.03,
        "star": 0.02,
        "fan_in": 0.02,
        "fan_out": 0.02,
    }.get(ctype, 0.0)
    if not directed_boost:
        return base
    return {
        "directed_cycle": 0.16,
        "dense_hyperlink_block": 0.18,
        "directed_chain": 0.16,
        "directed_star": 0.10,
    }.get(ctype, base)


def type_priority(item: Dict[str, Any], query: Dict[str, Any]) -> int:
    motif_type = str(query.get("motif_type", "")).lower()
    ctype = str(item.get("candidate_type", "")).lower()
    if motif_type and ctype and (ctype in motif_type or motif_type in ctype):
        return 0
    if "dense" in motif_type and "dense" in ctype:
        return 0
    if "cycle" in motif_type and "cycle" in ctype:
        return 0
    if "triangle" in motif_type and ctype == "triangle":
        return 0
    if "feed_forward" in motif_type and ctype == "feed_forward":
        return 0
    if "chain" in motif_type and "chain" in ctype:
        return 0
    if "star" in motif_type and ctype in {"star", "fan_in", "fan_out"}:
        return 0
    return 1


def prioritized_neighbors(
    node: str,
    compatibility: Dict[str, Any],
    max_neighbors: int,
    directed: bool,
    direction: str = "out",
) -> List[Tuple[str, str]]:
    if directed:
        adj_name = "reverse_adj" if direction == "in" else "directed_adj"
    else:
        adj_name = "undirected_adj"
    rows = list(compatibility[adj_name].get(node, []))
    rows.sort(
        key=lambda item: (
            -edge_cross_partition_score(item[1], compatibility),
            item[0],
            item[1],
        )
    )
    return rows[:max_neighbors]


def edge_cross_partition_score(edge_id: str, compatibility: Dict[str, Any]) -> float:
    src, dst = compatibility["edge_lookup"].get(edge_id, ("", ""))
    parts = set()
    parts.update(compatibility["node_to_partitions"].get(src, set()))
    parts.update(compatibility["node_to_partitions"].get(dst, set()))
    parts.update(compatibility["edge_to_partitions"].get(edge_id, set()))
    clean = {part for part in parts if part and part != "UNKNOWN"}
    return float(min(1.0, len(clean) / 2.0))


def clean_node_partitions(node: str, compatibility: Dict[str, Any]) -> Set[str]:
    return {part for part in compatibility["node_to_partitions"].get(str(node), set()) if part and part != "UNKNOWN"}


def undirected_pair_edge(compatibility: Dict[str, Any]) -> Dict[frozenset, str]:
    pair_edge: Dict[frozenset, str] = {}
    for edge_id, (src, dst) in compatibility["edge_lookup"].items():
        key = frozenset((src, dst))
        current = pair_edge.get(key)
        if current is None or edge_cross_partition_score(edge_id, compatibility) > edge_cross_partition_score(current, compatibility):
            pair_edge[key] = edge_id
    return pair_edge


def write_standard_artifacts(compatibility: Dict[str, Any], candidates: List[Dict[str, Any]], output_dir: Path) -> None:
    with open(output_dir / "comot_standard_candidates.jsonl", "w", encoding="utf-8") as f:
        for item in candidates:
            f.write(json.dumps(item) + "\n")
    stats = {
        "num_candidates": len(candidates),
        "candidate_types": {},
    }
    for item in candidates:
        ctype = item["candidate_type"]
        stats["candidate_types"][ctype] = stats["candidate_types"].get(ctype, 0) + 1
    (output_dir / "comot_standard_summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
