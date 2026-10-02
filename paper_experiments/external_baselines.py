#!/usr/bin/env python3
"""Run gMatch and MixMatch as task-adapted baselines.

The adapter follows the VF2/TurboISO protocol: the full processed
topology is collapsed to an undirected, unlabelled simple graph; anonymized query
topology is matched; at most 100 predictions are emitted globally; and the
CoMot evaluator is used unchanged. The script never reads ground truth until
the evaluator call.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.evaluate import evaluate_predictions  # noqa: E402
from methods.baselines.common import prediction, save_predictions  # noqa: E402


DATASETS = ("amlworld", "elliptic", "dblp", "web_google")
METHODS = ("gMatch-adapted", "MixMatch-adapted")
MAX_EDGES = {"amlworld": 500_000, "elliptic": None, "dblp": None, "web_google": None}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def load_processed_graph(dataset: str) -> tuple[list[str], OrderedDict[tuple[str, str], str]]:
    path = PROJECT_ROOT / "data" / "processed" / dataset / "edges.csv"
    nrows = MAX_EDGES[dataset]
    df = pd.read_csv(path, dtype=str, usecols=["edge_id", "src", "dst"], nrows=nrows)
    nodes: OrderedDict[str, None] = OrderedDict()
    edges: OrderedDict[tuple[str, str], str] = OrderedDict()
    for row in df.itertuples(index=False):
        src, dst = str(row.src), str(row.dst)
        nodes.setdefault(src, None)
        nodes.setdefault(dst, None)
        # nx.Graph.add_edge used by the VF2 baseline overwrites the
        # edge_id of an existing undirected pair while preserving insertion order.
        edges[pair_key(src, dst)] = str(row.edge_id)
    return list(nodes), edges


def write_graph(path: Path, nodes: list[str], edges: OrderedDict[tuple[str, str], str]) -> dict[str, int]:
    node_id = {node: i for i, node in enumerate(nodes)}
    degree = [0] * len(nodes)
    int_edges: list[tuple[int, int]] = []
    for src, dst in edges:
        u, v = node_id[src], node_id[dst]
        int_edges.append((u, v))
        degree[u] += 1
        degree[v] += 1
    with path.open("w", encoding="utf-8") as f:
        f.write(f"t {len(nodes)} {len(int_edges)}\n")
        for i, deg in enumerate(degree):
            f.write(f"v {i} 0 {deg}\n")
        for u, v in int_edges:
            f.write(f"e {u} {v}\n")
    return node_id


def query_topology(query: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
    nodes = [str(n) for n in query.get("nodes", []) if isinstance(n, str) and n != "..." and not n.endswith("_i")]
    node_set = set(nodes)
    edges: OrderedDict[tuple[str, str], None] = OrderedDict()
    for edge in query.get("edges", []):
        if isinstance(edge, dict):
            src, dst = str(edge.get("src")), str(edge.get("dst"))
        elif isinstance(edge, list) and len(edge) == 2:
            src, dst = str(edge[0]), str(edge[1])
        else:
            continue
        if src in node_set and dst in node_set:
            edges.setdefault(pair_key(src, dst), None)
    return nodes, list(edges)


def write_query(path: Path, query: dict[str, Any]) -> int:
    nodes, edges = query_topology(query)
    node_id = {node: i for i, node in enumerate(nodes)}
    degree = [0] * len(nodes)
    int_edges = []
    for src, dst in edges:
        u, v = node_id[src], node_id[dst]
        int_edges.append((u, v))
        degree[u] += 1
        degree[v] += 1
    with path.open("w", encoding="utf-8") as f:
        f.write(f"t {len(nodes)} {len(int_edges)}\n")
        for i, deg in enumerate(degree):
            f.write(f"v {i} 0 {deg}\n")
        for u, v in int_edges:
            f.write(f"e {u} {v}\n")
    return len(nodes)


def parse_gmatch(mapping: Path, order: Path) -> list[dict[int, int]]:
    if not mapping.exists() or not order.exists():
        return []
    qorder = [int(x) for x in order.read_text().strip().split(",") if x]
    rows = []
    for line in mapping.read_text().splitlines():
        values = [int(x) for x in line.split(",") if x]
        if len(values) == len(qorder):
            rows.append({qorder[pos]: value for pos, value in enumerate(values)})
    return rows


def parse_mixmatch(mapping: Path) -> tuple[list[dict[int, int]], list[float]]:
    rows, scores = [], []
    if not mapping.exists():
        return rows, scores
    for line in mapping.read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 3:
            continue
        scores.append(float(fields[1]))
        rows.append({int(q): int(d) for q, d in (field.split(":") for field in fields[2:])})
    return rows, scores


def induced_edge_ids(mapped_nodes: list[str], edges: OrderedDict[tuple[str, str], str]) -> list[str]:
    selected = sorted(set(mapped_nodes))
    found = []
    for i, src in enumerate(selected):
        for dst in selected[i:]:
            edge_id = edges.get(pair_key(src, dst))
            if edge_id is not None:
                found.append(edge_id)
    return sorted(found)


def run_query(method: str, binary: Path, graph_path: Path, query_path: Path, work: Path,
              remaining: int, timeout: int) -> tuple[list[dict[int, int]], list[float], dict[str, Any]]:
    mapping = work / "mappings.csv"
    log = work / "engine.log"
    env = os.environ.copy()
    internal_mapping_cap = remaining
    if method == "gMatch-adapted":
        order = work / "order.csv"
        # gMatch does not enforce query self-loops. Over-fetch raw embeddings so
        # post-validation can still fill the requested budget.
        internal_mapping_cap = min(10_000, max(1_000, remaining * 16))
        env.update(GMATCH_MAPPING_OUTPUT=str(mapping.resolve()),
                   GMATCH_MAPPING_LIMIT=str(internal_mapping_cap),
                   GMATCH_ORDER_OUTPUT=str(order.resolve()))
        command = [str(binary.resolve()), "-d", str(graph_path.resolve()), "-q", str(query_path.resolve())]
    else:
        env["MIXMATCH_MAPPING_OUTPUT"] = str(mapping.resolve())
        command = [str(binary.resolve()), "-d", str(graph_path.resolve()), "-q", str(query_path.resolve()),
                   "-filter", "VEQ", "-order", "CFL", "-engine", "MMK", "-num", str(remaining),
                   "-symmetry", "1", "-FairT", "2", "-time", str(timeout),
                   "-SF", str((work / "coverage").resolve())]
    start = time.perf_counter()
    try:
        proc = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=timeout + 15)
        status = "complete" if proc.returncode == 0 else f"error:{proc.returncode}"
        log.write_text(proc.stdout, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        output = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        log.write_text(output, encoding="utf-8")
    wall = time.perf_counter() - start
    if method == "gMatch-adapted":
        rows = parse_gmatch(mapping, work / "order.csv")
        scores = [1.0] * len(rows)
    else:
        rows, scores = parse_mixmatch(mapping)
    return rows, scores, {"status": status, "wall_seconds": wall, "command": command,
                          "raw_mappings": len(rows), "internal_mapping_cap": internal_mapping_cap}


def run(dataset: str, method: str, seed: int, binary: Path, out_root: Path,
        budget: int, timeout: int) -> None:
    dataset_dir = PROJECT_ROOT / "data" / "processed" / dataset
    out = out_root / dataset / method / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    cache = out_root / "graph_cache" / dataset
    cache.mkdir(parents=True, exist_ok=True)
    graph_path = cache / "processed_undirected_unlabelled.graph"
    map_path = cache / "node_ids.json"
    edge_path = cache / "simple_edges.json"

    if not graph_path.exists():
        nodes, edges = load_processed_graph(dataset)
        write_graph(graph_path, nodes, edges)
        map_path.write_text(json.dumps(nodes), encoding="utf-8")
        edge_path.write_text(json.dumps([[a, b, eid] for (a, b), eid in edges.items()]), encoding="utf-8")
    else:
        nodes = json.loads(map_path.read_text(encoding="utf-8"))
        edges = OrderedDict(((a, b), eid) for a, b, eid in json.loads(edge_path.read_text(encoding="utf-8")))

    queries = json.loads((dataset_dir / "method_queries.json").read_text(encoding="utf-8"))
    predictions: list[dict[str, Any]] = []
    runs = []
    overall_start = time.perf_counter()
    for qi, query in enumerate(queries):
        if len(predictions) >= budget:
            break
        work = out / "engine_runs" / f"query_{qi:04d}"
        work.mkdir(parents=True, exist_ok=True)
        query_path = work / "query.graph"
        qn = write_query(query_path, query)
        _, qedges = query_topology(query)
        if qn == 0 or not qedges:
            runs.append({"query_index": qi, "query_id": query.get("query_id"), "status": "empty_query"})
            continue
        if method == "gMatch-adapted" and qn > 32:
            runs.append({"query_index": qi, "query_id": query.get("query_id"), "status": "unsupported_gt32"})
            continue
        remaining = budget - len(predictions)
        mappings, scores, record = run_query(method, binary, graph_path, query_path, work, remaining, timeout)
        accepted = 0
        rejected_invalid = 0
        qnodes, topology_edges = query_topology(query)
        qindex = {node: i for i, node in enumerate(qnodes)}
        for mapping, score in zip(mappings, scores):
            if set(mapping) != set(range(qn)) or len(set(mapping.values())) != qn:
                rejected_invalid += 1
                continue
            try:
                mapped_nodes = [nodes[mapping[i]] for i in range(qn)]
            except (IndexError, KeyError):
                rejected_invalid += 1
                continue
            if any(pair_key(mapped_nodes[qindex[src]], mapped_nodes[qindex[dst]]) not in edges
                   for src, dst in topology_edges):
                # gMatch's native simple-graph code does not enforce query
                # self-loops. The task adapter rejects any exported row that
                # does not satisfy every collapsed query edge.
                rejected_invalid += 1
                continue
            pred_edges = induced_edge_ids(mapped_nodes, edges)
            predictions.append(prediction(
                query_id=str(query.get("query_id", f"query_{qi}")),
                nodes=mapped_nodes,
                edges=pred_edges,
                score=float(score),
                runtime=time.perf_counter() - overall_start,
                metadata={
                    "method": method,
                    "implementation_status": "official_search_task_adapted",
                    "input_protocol": "undirected_unlabelled_full_processed_topology_global_top100",
                    "query_index": qi,
                },
            ))
            accepted += 1
            if len(predictions) >= budget:
                break
        record.update(query_index=qi, query_id=query.get("query_id"), accepted=accepted,
                      rejected_invalid=rejected_invalid,
                      cumulative_predictions=len(predictions), query_nodes=qn, query_edges=len(qedges))
        runs.append(record)

    save_predictions(predictions, out / "predictions.jsonl")
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    evaluate_predictions(predictions, json.loads((dataset_dir / "ground_truth.json").read_text(encoding="utf-8")), out)
    runtime = {
        "dataset": dataset, "method": method, "seed": seed, "budget": budget,
        "timeout_per_query": timeout, "total_seconds": time.perf_counter() - overall_start,
        "num_predictions": len(predictions), "engine_runs": runs,
    }
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    provenance = {
        "binary": str(binary.resolve()), "binary_sha256": sha256(binary),
        "graph": str(graph_path.resolve()), "graph_sha256": sha256(graph_path),
        "method_queries": str((dataset_dir / "method_queries.json").resolve()),
        "ground_truth_used_only_by_evaluator": str((dataset_dir / "ground_truth.json").resolve()),
        "reference_implementation": str((PROJECT_ROOT / "src/methods/baselines/matching.py").resolve()),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": dataset, "method": method, "seed": seed,
                      "predictions": len(predictions), "seconds": runtime["total_seconds"]}), flush=True)


def aggregate(out_root: Path) -> None:
    rows = []
    for method in METHODS:
        row: dict[str, Any] = {"method": method}
        for dataset in DATASETS:
            values = {"motif_f1": [], "edge_f1": [], "node_f1": []}
            for seed in range(3):
                metrics = json.loads((out_root / dataset / method / f"seed_{seed}" / "metrics.json").read_text())
                precision = float(metrics["motif_level"]["partial_match"])
                recall = float(metrics["motif_level"]["motif_recall"])
                values["motif_f1"].append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
                values["edge_f1"].append(float(metrics["edge_level"]["f1"]))
                values["node_f1"].append(float(metrics["node_level"]["f1"]))
            for metric, vals in values.items():
                row[f"{dataset}_{metric}"] = 100.0 * sum(vals) / len(vals)
        rows.append(row)
    columns = ["method"] + [f"{ds}_{m}" for ds in DATASETS for m in ("motif_f1", "edge_f1", "node_f1")]
    with (out_root / "table_2x12.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    md = ["| Method | " + " | ".join(f"{ds}:{m}" for ds in DATASETS for m in ("M-F1", "E-F1", "N-F1")) + " |",
          "|---|" + "---:|" * 12]
    for row in rows:
        md.append("| " + row["method"] + " | " + " | ".join(
            f"{row[f'{ds}_{metric}']:.2f}" for ds in DATASETS for metric in ("motif_f1", "edge_f1", "node_f1")) + " |")
    (out_root / "table_2x12.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/paper/external_adapters")
    ap.add_argument("--gmatch", type=Path, default=PROJECT_ROOT / "third_party/gmatch/SubgraphMatching_unlabeled")
    ap.add_argument("--mixmatch", type=Path, default=PROJECT_ROOT / "third_party/mixmatch/SubgraphMatching.out")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if not args.aggregate_only:
        binaries = {"gMatch-adapted": args.gmatch, "MixMatch-adapted": args.mixmatch}
        for dataset in args.datasets:
            for method in args.methods:
                for seed in args.seeds:
                    run(dataset, method, seed, binaries[method], args.out, args.budget, args.timeout)
    if set(args.datasets) == set(DATASETS) and set(args.methods) == set(METHODS) and set(args.seeds) == {0, 1, 2}:
        aggregate(args.out)


if __name__ == "__main__":
    main()
