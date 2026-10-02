#!/usr/bin/env python3
"""Evaluate Raw/Base/Diverse/Hybrid ranking strategies on all four datasets.

Strategy definitions:
- Raw: sort the full candidate pool by its original candidate score.
- Base: use the persisted CoMot/base-rerank order when available.
- Diverse: round-robin candidates by candidate type, preserving Base order inside each type.
- Hybrid: keep the first M Base candidates, then append the Diverse order of the tail.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from methods.comot_pipeline.candidate_reranker import rerank_score  # noqa: E402


DATASETS = ["amlworld", "elliptic", "dblp", "web_google"]
SEEDS = [0, 1, 2]
METHOD = "comot"
AML_TAG = "five_party"
TOPKS = [10, 20, 50, 100]
GT_HIT_TOPKS = [100, 200]
STRATEGIES = ["raw", "base", "diverse", "hybrid"]

METRIC_COLUMNS = [
    "P@10",
    "P@20",
    "P@50",
    "P@100",
    "R@10",
    "R@20",
    "R@50",
    "R@100",
    "GT hit@100",
    "GT hit@200",
    "Candidate-type coverage@100",
    "Motif-type coverage@100",
]
BY_SEED_COLUMNS = [
    "dataset",
    "method",
    "seed",
    "ranking_strategy",
    "status",
    "num_candidates",
    "num_ground_truth",
    *METRIC_COLUMNS,
    "runtime",
    "source",
    "notes",
]


def parse_json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.6f}"
    return str(value)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def as_str_set(values: Iterable[Any]) -> Set[str]:
    return {str(value) for value in values if str(value) not in {"", "None", "nan"}}


def load_aml_candidates(run_dir: Path) -> tuple[pd.DataFrame, Path]:
    candidate_dir = run_dir / f"semotif_candidates_{AML_TAG}"
    frames = []
    for name in ["chain_candidates.csv", "fan_candidates.csv", "cycle_candidates.csv"]:
        path = candidate_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No AMLWorld candidate CSVs found in {candidate_dir}")

    df = pd.concat(frames, ignore_index=True).copy()
    for column in ["score_mean", "score_min", "num_edges", "num_nodes"]:
        if column not in df.columns:
            df[column] = 0
    if "candidate_id" not in df.columns:
        df["candidate_id"] = [f"AML_CAND_{idx:08d}" for idx in range(len(df))]

    df["original_rank"] = range(len(df))
    df["raw_score"] = df["score_mean"].astype(float)
    df["raw_tiebreak"] = df["score_min"].astype(float)
    df["edge_ids"] = df["tx_ids_json"].map(lambda value: sorted(as_str_set(parse_json_list(value))))
    df["node_ids"] = df["nodes_json"].map(lambda value: sorted(as_str_set(parse_json_list(value))))

    reranked_path = run_dir / f"semotif_candidates_{AML_TAG}_reranked" / "candidates_reranked.csv"
    if reranked_path.exists():
        base = pd.read_csv(reranked_path)
        base = base.reset_index().rename(columns={"index": "base_rank"})
        rank_cols = ["candidate_id", "base_rank"]
        if "rerank_score" in base.columns:
            rank_cols.append("rerank_score")
        df = df.merge(base[rank_cols], on="candidate_id", how="left")
        df["base_rank"] = df["base_rank"].fillna(len(df) + df["original_rank"]).astype(int)
        if "rerank_score" in df.columns:
            df["base_score"] = df["rerank_score"].fillna(df["raw_score"]).astype(float)
        else:
            df["base_score"] = df["raw_score"].astype(float)
    else:
        df["base_score"] = df.apply(rerank_score, axis=1)
        df = df.sort_values(
            ["base_score", "score_mean", "score_min", "num_edges"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        df["base_rank"] = range(len(df))

    return df, candidate_dir


def load_standard_candidates(run_dir: Path) -> tuple[pd.DataFrame, Path]:
    path = run_dir / "comot_standard_candidates.jsonl"
    if not path.exists():
        raise FileNotFoundError(str(path))

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            rows.append(
                {
                    "candidate_id": f"{item.get('query_id', 'query')}::{item.get('candidate_type', 'type')}::{idx}",
                    "query_id": str(item.get("query_id", "")),
                    "candidate_type": str(item.get("candidate_type", "unknown")),
                    "node_ids": sorted(as_str_set(item.get("node_ids", []))),
                    "edge_ids": sorted(as_str_set(item.get("edge_ids", []))),
                    "num_nodes": len(item.get("node_ids", [])),
                    "num_edges": len(item.get("edge_ids", [])),
                    "raw_score": float(item.get("score") or 0.0),
                    "raw_tiebreak": float(metadata.get("structural_consistency_score") or 0.0),
                    "base_score": float(item.get("score") or 0.0),
                    "base_rank": idx,
                    "original_rank": idx,
                }
            )
    if not rows:
        raise ValueError(f"No candidates found in {path}")
    return pd.DataFrame(rows), path


def load_candidates(dataset: str, run_dir: Path) -> tuple[pd.DataFrame, Path, str]:
    if dataset == "amlworld" and (run_dir / f"semotif_candidates_{AML_TAG}").exists():
        df, source = load_aml_candidates(run_dir)
        return df, source, "AMLWorld candidate CSVs plus persisted base rerank when available."
    df, source = load_standard_candidates(run_dir)
    return df, source, "Standard CoMot JSONL; Base uses persisted JSONL order."


def load_aml_ground_truth(run_dir: Path) -> tuple[List[Dict[str, Any]], Dict[str, Set[str]], Dict[str, str]]:
    path = run_dir / f"semotif_partitioned_{AML_TAG}" / "ground_truth_motifs.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    gt_rows: List[Dict[str, Any]] = []
    edge_to_motifs: Dict[str, Set[str]] = defaultdict(set)
    motif_to_type: Dict[str, str] = {}
    for _, row in df.iterrows():
        motif_id = str(row["motif_instance_id"])
        motif_type = str(row.get("motif_type", "unknown"))
        edges = as_str_set(parse_json_list(row.get("tx_ids_json")))
        gt_rows.append({"motif_instance_id": motif_id, "true_edges": sorted(edges), "motif_type": motif_type})
        motif_to_type[motif_id] = motif_type
        for edge_id in edges:
            edge_to_motifs[edge_id].add(motif_id)
    return gt_rows, edge_to_motifs, motif_to_type


def load_standard_ground_truth(dataset: str) -> tuple[List[Dict[str, Any]], Dict[str, Set[str]], Dict[str, str]]:
    data_dir = PROJECT_ROOT / "data" / "processed" / dataset
    gt_path = data_dir / "ground_truth.json"
    query_path = data_dir / "query_motifs.json"
    if not gt_path.exists():
        raise FileNotFoundError(str(gt_path))
    if not query_path.exists():
        raise FileNotFoundError(str(query_path))

    gt_rows = json.loads(gt_path.read_text(encoding="utf-8"))
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    query_to_type = {str(row["query_id"]): str(row.get("motif_type", "unknown")) for row in queries}
    edge_to_motifs: Dict[str, Set[str]] = defaultdict(set)
    motif_to_type: Dict[str, str] = {}
    for row in gt_rows:
        motif_id = str(row.get("motif_instance_id") or row.get("query_id"))
        motif_to_type[motif_id] = query_to_type.get(str(row.get("query_id")), "unknown")
        for edge_id in row.get("true_edges", []):
            edge_to_motifs[str(edge_id)].add(motif_id)
    return gt_rows, edge_to_motifs, motif_to_type


def load_ground_truth(dataset: str, run_dir: Path) -> tuple[List[Dict[str, Any]], Dict[str, Set[str]], Dict[str, str]]:
    if dataset == "amlworld" and (run_dir / f"semotif_partitioned_{AML_TAG}" / "ground_truth_motifs.csv").exists():
        return load_aml_ground_truth(run_dir)
    return load_standard_ground_truth(dataset)


def attach_hits(df: pd.DataFrame, edge_to_motifs: Dict[str, Set[str]]) -> pd.DataFrame:
    out = df.copy()

    def hit_ids(edge_ids: Sequence[str]) -> List[str]:
        hits: Set[str] = set()
        for edge_id in edge_ids:
            hits.update(edge_to_motifs.get(str(edge_id), set()))
        return sorted(hits)

    out["hit_motif_ids"] = out["edge_ids"].map(hit_ids)
    return out


def sort_raw(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["raw_score", "raw_tiebreak", "num_edges", "num_nodes", "original_rank"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def sort_base(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["base_rank", "base_score", "original_rank"], ascending=[True, False, True]).reset_index(drop=True)


def diverse_order(df: pd.DataFrame) -> pd.DataFrame:
    base = sort_base(df)
    buckets = {
        candidate_type: group.reset_index(drop=True)
        for candidate_type, group in base.groupby("candidate_type", sort=False)
    }
    type_order = sorted(
        buckets,
        key=lambda candidate_type: (
            int(buckets[candidate_type].iloc[0]["base_rank"]),
            str(candidate_type),
        ),
    )
    positions = {candidate_type: 0 for candidate_type in buckets}
    selected = []
    while len(selected) < len(base):
        progressed = False
        for candidate_type in type_order:
            pos = positions[candidate_type]
            bucket = buckets[candidate_type]
            if pos >= len(bucket):
                continue
            selected.append(bucket.iloc[pos])
            positions[candidate_type] = pos + 1
            progressed = True
        if not progressed:
            break
    return pd.DataFrame(selected).reset_index(drop=True)


def hybrid_order(df: pd.DataFrame, prefix: int) -> pd.DataFrame:
    base = sort_base(df)
    prefix_df = base.head(prefix).copy()
    prefix_ids = set(prefix_df["candidate_id"].astype(str))
    tail = base[~base["candidate_id"].astype(str).isin(prefix_ids)].copy()
    diverse_tail = diverse_order(tail) if len(tail) else tail
    return pd.concat([prefix_df, diverse_tail], ignore_index=True).reset_index(drop=True)


def strategy_order(strategy: str, df: pd.DataFrame, hybrid_prefix: int) -> tuple[pd.DataFrame, str]:
    if strategy == "raw":
        return sort_raw(df), "Raw score order; no type-balancing rerank."
    if strategy == "base":
        return sort_base(df), "Persisted CoMot/base-rerank order."
    if strategy == "diverse":
        return diverse_order(df), "Candidate-type round-robin over Base order."
    if strategy == "hybrid":
        return hybrid_order(df, hybrid_prefix), f"Base top-{hybrid_prefix} prefix plus Diverse tail."
    raise ValueError(f"Unknown ranking strategy: {strategy}")


def evaluate_order(
    ordered: pd.DataFrame,
    gt_rows: Sequence[Dict[str, Any]],
    motif_to_type: Dict[str, str],
    all_candidate_types: Set[str],
) -> Dict[str, Any]:
    num_gt = len(gt_rows)
    all_motif_types = set(motif_to_type.values())
    row: Dict[str, Any] = {
        "num_candidates": int(len(ordered)),
        "num_ground_truth": int(num_gt),
    }

    for k in TOPKS:
        top = ordered.head(k)
        hit_candidates = top["hit_motif_ids"].map(bool).sum()
        hit_motifs = {motif_id for hits in top["hit_motif_ids"] for motif_id in hits}
        row[f"P@{k}"] = float(hit_candidates / len(top)) if len(top) else 0.0
        row[f"R@{k}"] = float(len(hit_motifs) / num_gt) if num_gt else 0.0

    for k in GT_HIT_TOPKS:
        top = ordered.head(k)
        hit_motifs = {motif_id for hits in top["hit_motif_ids"] for motif_id in hits}
        row[f"GT hit@{k}"] = int(len(hit_motifs))

    top100 = ordered.head(100)
    top100_candidate_types = set(top100["candidate_type"].astype(str))
    top100_motif_types = {
        motif_to_type[motif_id]
        for hits in top100["hit_motif_ids"]
        for motif_id in hits
        if motif_id in motif_to_type
    }
    row["Candidate-type coverage@100"] = (
        float(len(top100_candidate_types) / len(all_candidate_types)) if all_candidate_types else 0.0
    )
    row["Motif-type coverage@100"] = float(len(top100_motif_types) / len(all_motif_types)) if all_motif_types else 0.0
    return row


def write_csv(rows: Sequence[Dict[str, Any]], path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt(row.get(column)) for column in columns})


def summarize(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    completed = df[df["status"] == "completed"].copy()
    for column in ["num_candidates", "num_ground_truth", *METRIC_COLUMNS, "runtime"]:
        completed[column] = pd.to_numeric(completed[column], errors="coerce")

    grouped = completed.groupby(["dataset", "method", "ranking_strategy"], sort=False)
    summary = grouped.agg(
        num_seeds=("seed", "nunique"),
        num_candidates=("num_candidates", "mean"),
        num_ground_truth=("num_ground_truth", "mean"),
        **{f"{column}_mean": (column, "mean") for column in METRIC_COLUMNS},
        **{f"{column}_std": (column, "std") for column in METRIC_COLUMNS},
        runtime_mean=("runtime", "mean"),
    ).reset_index()
    summary["status"] = "completed"
    return summary


def run_dataset_seed(dataset: str, seed: int, output_root: Path, hybrid_prefix: int) -> List[Dict[str, Any]]:
    run_dir = output_root / dataset / METHOD / f"seed_{seed}"
    rows: List[Dict[str, Any]] = []
    try:
        candidate_df, source_path, source_note = load_candidates(dataset, run_dir)
        gt_rows, edge_to_motifs, motif_to_type = load_ground_truth(dataset, run_dir)
        candidate_df = attach_hits(candidate_df, edge_to_motifs)
        all_candidate_types = set(candidate_df["candidate_type"].astype(str))
        for strategy in STRATEGIES:
            start = time.perf_counter()
            ordered, notes = strategy_order(strategy, candidate_df, hybrid_prefix)
            runtime = time.perf_counter() - start
            metrics = evaluate_order(ordered, gt_rows, motif_to_type, all_candidate_types)
            rows.append(
                {
                    "dataset": dataset,
                    "method": METHOD,
                    "seed": seed,
                    "ranking_strategy": strategy,
                    "status": "completed",
                    "runtime": runtime,
                    "source": rel(source_path),
                    "notes": f"{notes} {source_note}",
                    **metrics,
                }
            )
    except Exception as exc:  # noqa: BLE001 - keep multi-dataset batch alive and report failures in CSV.
        for strategy in STRATEGIES:
            rows.append(
                {
                    "dataset": dataset,
                    "method": METHOD,
                    "seed": seed,
                    "ranking_strategy": strategy,
                    "status": "failed",
                    "runtime": 0.0,
                    "source": str(run_dir),
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--summary-dir", type=Path, default=PROJECT_ROOT / "outputs" / "summary")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--hybrid-prefix", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    summary_dir = args.summary_dir if args.summary_dir.is_absolute() else PROJECT_ROOT / args.summary_dir

    rows: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        for seed in args.seeds:
            rows.extend(run_dataset_seed(dataset, seed, output_root, args.hybrid_prefix))

    by_seed_path = summary_dir / "prioritization_by_seed.csv"
    summary_path = summary_dir / "prioritization.csv"
    write_csv(rows, by_seed_path, BY_SEED_COLUMNS)
    summary = summarize(rows)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    print(
        json.dumps(
            {
                "datasets": args.datasets,
                "seeds": args.seeds,
                "hybrid_prefix": args.hybrid_prefix,
                "num_rows": len(rows),
                "outputs": [rel(by_seed_path), rel(summary_path)],
                "failed_rows": sum(1 for row in rows if row.get("status") != "completed"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
