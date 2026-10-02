#!/usr/bin/env python3
"""Evaluate saved ranked predictions at K=100,250,500,1000.

No method is rerun. The saved baseline configurations normally emit 1,000
predictions; CSGM emits 180, so budgets above 180 use all of its available
predictions. Matching uses the same normalization, best-GT assignment, and
non-empty-overlap semantics as src/evaluation/metrics.py.

This intentionally differs from Figure 6's AMLWorld ranking diagnostic,
which counts transaction-edge overlap only and lets one prediction cover
multiple GT motifs.  Candidate order and GT records themselves are unchanged.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.methods.comot_pipeline.candidate_reranker import candidate_type_bonus, compactness_bonus, size_penalty
from src.evaluation.metrics import best_ground_truth_match, normalize_ground_truth


OUT_DIR = ROOT / "outputs" / "paper" / "retrieval_budget"
DATASETS = ("amlworld", "elliptic", "dblp", "web_google")
SEEDS = (0, 1, 2)
BUDGETS = (100, 250, 500, 1000)
METHODS = ("CoMot", "ISONET", "TPAB", "AA", "CSGM", "NeuGN", "CN", "FANMOD", "SubGNN", "SEAL")
OUTPUT_METHODS = {
    "CoMot": "comot",
    "ISONET": "ISONET",
    "TPAB": "TPAB",
    "AA": "aa",
    "CSGM": "CSGM",
    "NeuGN": "NeuGN",
    "CN": "cn",
    "FANMOD": "fanmod",
    "SubGNN": "subgnn",
    "SEAL": "seal",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    parsed = json.loads(value) if isinstance(value, str) else []
    return parsed if isinstance(parsed, list) else []


def strip_partition(value: Any) -> str:
    return str(value).split("::", 1)[-1]


def amlworld_final_score(row: pd.Series) -> float:
    typ = str(row.candidate_type).upper()
    num_nodes, num_edges = int(row.num_nodes), int(row.num_edges)
    score_mean, score_min = float(row.score_mean), float(row.score_min)
    stability = -abs(score_mean - score_min)
    richness = 0.0
    if typ == "CHAIN":
        richness = 0.03 * min(max(num_edges - 2, 0), 4)
    elif typ == "CYCLE":
        richness = 0.03 * min(max(num_nodes - 2, 0), 3)
    elif typ in ("FAN_IN", "FAN_OUT"):
        richness = 0.01 * min(max(num_edges - 2, 0), 2)
    return (
        0.60 * score_mean
        + 0.40 * score_min
        - 5.00 * stability
        + candidate_type_bonus(typ)
        + 0.10 * compactness_bonus(typ, num_nodes, num_edges)
        + richness
        - size_penalty(typ, num_nodes, num_edges)
    )


def load_amlworld_comot(seed: int) -> tuple[list[dict[str, Any]], str]:
    path = (
        ROOT
        / "outputs"
        / "amlworld"
        / "comot"
        / f"seed_{seed}"
        / "semotif_candidates_five_party_reranked"
        / "candidates_reranked.csv"
    )
    frame = pd.read_csv(path)
    frame["final_score"] = frame.apply(amlworld_final_score, axis=1)
    frame = frame.sort_values(
        ["final_score", "score_mean", "score_min", "num_edges"],
        ascending=False,
    )
    predictions = []
    for _, row in frame.iterrows():
        predictions.append(
            {
                "prediction_id": str(row.candidate_id),
                "query_id": str(row.candidate_type),
                "predicted_nodes": [strip_partition(x) for x in parse_list(row.nodes_json)],
                "predicted_edges": [str(x) for x in parse_list(row.tx_ids_json)],
                "score": float(row.final_score),
            }
        )

    # Verify against the main-run prefix when it is present.
    main_path = ROOT / "outputs" / "amlworld" / "comot" / f"seed_{seed}" / "predictions.jsonl"
    if main_path.exists():
        main_rows = [json.loads(line) for line in main_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        main_ids = [str(row.get("metadata", {}).get("candidate_id", row.get("prediction_id", ""))) for row in main_rows]
        generated_ids = [row["prediction_id"] for row in predictions[: len(main_ids)]]
        if main_ids and generated_ids != main_ids:
            raise RuntimeError(f"AMLWorld seed {seed}: generated final-weight prefix differs from main predictions")
    return predictions, str(path.relative_to(ROOT))


def load_standard_comot(dataset: str, seed: int) -> tuple[list[dict[str, Any]], str]:
    path = ROOT / "outputs" / dataset / "comot" / f"seed_{seed}" / "comot_standard_candidates.jsonl"
    predictions = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        predictions.append(
            {
                "prediction_id": f"{row.get('query_id', '')}::{index:08d}",
                "query_id": str(row.get("query_id", "")),
                "predicted_nodes": [str(x) for x in row.get("node_ids", [])],
                "predicted_edges": [str(x) for x in row.get("edge_ids", [])],
                "score": float(row.get("score", 0.0)),
            }
        )

    # The saved main predictions must be the same prefix of this ranked pool.
    main_path = ROOT / "outputs" / dataset / "comot" / f"seed_{seed}" / "predictions.jsonl"
    main_predictions = [json.loads(line) for line in main_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, (full_row, main_row) in enumerate(zip(predictions[: len(main_predictions)], main_predictions)):
        key_full = (full_row["query_id"], full_row["predicted_nodes"], full_row["predicted_edges"])
        key_main = (
            str(main_row.get("query_id", "")),
            [str(x) for x in main_row.get("predicted_nodes", [])],
            [str(x) for x in main_row.get("predicted_edges", [])],
        )
        if key_full != key_main:
            raise RuntimeError(f"{dataset} seed {seed}: full CoMot prefix mismatch at row {index}")
    return predictions, str(path.relative_to(ROOT))


def load_baseline(dataset: str, method: str, seed: int) -> tuple[list[dict[str, Any]], str]:
    output_method = OUTPUT_METHODS[method]
    path = ROOT / "outputs" / dataset / output_method / f"seed_{seed}" / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, row in enumerate(rows):
        row.setdefault("prediction_id", f"{output_method}::{index:08d}")
    return rows, str(path.relative_to(ROOT))


def load_predictions(dataset: str, method: str, seed: int) -> tuple[list[dict[str, Any]], str]:
    if method != "CoMot":
        return load_baseline(dataset, method, seed)
    if dataset == "amlworld":
        return load_amlworld_comot(seed)
    return load_standard_comot(dataset, seed)


def normalize_prediction_preserve_order(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "prediction_id": str(item.get("prediction_id", index)),
        "query_id": str(item.get("query_id", "")),
        "nodes": {str(x) for x in item.get("predicted_nodes", []) if x is not None},
        "edges": {str(x) for x in item.get("predicted_edges", []) if x is not None},
        "score": float(item.get("score", 0.0)),
        "runtime": float(item.get("runtime", 0.0)),
        "metadata": item.get("metadata", {}) or {},
    }


def evaluate_prefixes(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    gt_rows = normalize_ground_truth(ground_truth)
    node_to_gt: dict[str, set[int]] = defaultdict(set)
    edge_to_gt: dict[str, set[int]] = defaultdict(set)
    for gt_index, gt in enumerate(gt_rows):
        for node in gt["nodes"]:
            node_to_gt[node].add(gt_index)
        for edge in gt["edges"]:
            edge_to_gt[edge].add(gt_index)

    hits: list[bool] = []
    matched_ids: list[str] = []
    for index, item in enumerate(predictions):
        pred = normalize_prediction_preserve_order(item, index)
        candidate_indices: set[int] = set()
        for node in pred["nodes"]:
            candidate_indices.update(node_to_gt.get(node, ()))
        for edge in pred["edges"]:
            candidate_indices.update(edge_to_gt.get(edge, ()))
        candidate_gt = [gt for gt_index, gt in enumerate(gt_rows) if gt_index in candidate_indices]
        match = best_ground_truth_match(pred, candidate_gt)
        gt = match["ground_truth"]
        is_hit = bool(gt and ((pred["nodes"] & gt["nodes"]) or (pred["edges"] & gt["edges"])))
        hits.append(is_hit)
        matched_ids.append(gt["motif_id"] if is_hit else "")

    result = {}
    for requested in BUDGETS:
        n = min(int(requested), len(predictions))
        prefix_hits = sum(hits[:n])
        unique_gt = len({x for x in matched_ids[:n] if x})
        precision = prefix_hits / n if n else 0.0
        recall = unique_gt / len(gt_rows) if gt_rows else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[int(requested)] = {
            "num_predictions": n,
            "num_hit_predictions": prefix_hits,
            "num_unique_gt_hit": unique_gt,
            "num_gt": len(gt_rows),
            "precision": precision,
            "recall": recall,
            "motif_f1": f1,
        }
    return result


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, Any]] = []
    ground_truth = {}
    for dataset in DATASETS:
        raw = read_json(ROOT / "data" / "processed" / dataset / "ground_truth.json")
        ground_truth[dataset] = raw["ground_truth"] if isinstance(raw, dict) else raw

    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                predictions, source = load_predictions(dataset, method, seed)
                metrics = evaluate_prefixes(predictions, ground_truth[dataset])
                full_n = len(predictions)
                for budget in BUDGETS:
                    item = metrics[int(budget)]
                    per_seed.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "budget": str(budget),
                            "budget_order": BUDGETS.index(budget),
                            "full_available_predictions": full_n,
                            "source": source,
                            **item,
                        }
                    )
                print(f"done {dataset:10s} {method:7s} seed={seed} full={full_n}", flush=True)

    per = pd.DataFrame(per_seed)
    per.to_csv(OUT_DIR / "retrieval_budget_by_seed.csv", index=False)
    summary_rows = []
    for (dataset, method, budget, budget_order), group in per.groupby(
        ["dataset", "method", "budget", "budget_order"], sort=False
    ):
        item: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "budget": budget,
            "budget_order": budget_order,
            "num_seeds": len(group),
            "num_predictions_mean": group["num_predictions"].mean(),
            "full_available_predictions_mean": group["full_available_predictions"].mean(),
        }
        for metric in ("precision", "recall", "motif_f1"):
            values = group[metric].astype(float).tolist()
            item[f"{metric}_mean"] = sum(values) / len(values)
            item[f"{metric}_std"] = sample_std(values)
        summary_rows.append(item)
    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "budget_order", "method"])
    summary.to_csv(OUT_DIR / "retrieval_budget_summary.csv", index=False)

    # Main-result identity checks at K=1000.
    expected = {
        "amlworld": 0.706574,
        "elliptic": 0.9035,
        "dblp": 0.9350,
        "web_google": 0.5185,
    }
    for dataset, expected_f1 in expected.items():
        actual = float(
            summary[
                (summary.dataset == dataset)
                & (summary.method == "CoMot")
                & (summary.budget == "1000")
            ].motif_f1_mean.iloc[0]
        )
        if not math.isclose(actual, expected_f1, abs_tol=0.000051):
            raise RuntimeError(f"{dataset} CoMot@1000 mismatch: {actual} vs {expected_f1}")

    print(f"wrote {OUT_DIR / 'retrieval_budget_by_seed.csv'}")
    print(f"wrote {OUT_DIR / 'retrieval_budget_summary.csv'}")


if __name__ == "__main__":
    main()
