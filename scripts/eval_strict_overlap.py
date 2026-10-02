#!/usr/bin/env python3
"""Re-evaluate existing Table-II predictions under stricter motif overlap rules.

This script is intentionally independent of the default evaluator. It never
changes predictions, ground truth, or existing result files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.metrics import best_ground_truth_match, jaccard, normalize_ground_truth, normalize_predictions


DATASETS = ["amlworld", "elliptic", "dblp", "web_google"]
CRITERIA: Sequence[Tuple[str, float, str]] = (
    ("Original", 0.0, "original"),
    ("Strict-0.10", 0.10, "or"),
    ("Strict-0.25", 0.25, "or"),
    ("Strict-0.50", 0.50, "or"),
    ("Joint-0.25", 0.25, "and"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(run_dir: Path) -> List[Dict[str, Any]]:
    jsonl_path = run_dir / "predictions.jsonl"
    if jsonl_path.exists():
        return [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    json_path = run_dir / "predictions.json"
    data = read_json(json_path)
    return data["predictions"] if isinstance(data, dict) else data


def load_ground_truth(dataset: str) -> List[Dict[str, Any]]:
    data = read_json(PROJECT_ROOT / "data" / "processed" / dataset / "ground_truth.json")
    return data["ground_truth"] if isinstance(data, dict) else data


def discover_methods(summary_path: Path) -> List[str]:
    with open(summary_path, encoding="utf-8", newline="") as handle:
        return sorted({row["method"] for row in csv.DictReader(handle)})


def is_hit(mode: str, threshold: float, node_intersection: bool, edge_intersection: bool, node_j: float, edge_j: float) -> bool:
    if mode == "original":
        return node_intersection or edge_intersection
    if mode == "or":
        return node_j >= threshold or edge_j >= threshold
    if mode == "and":
        return node_j >= threshold and edge_j >= threshold
    raise ValueError(mode)


def evaluate_run(dataset: str, method: str, seed: int, predictions: Sequence[Dict[str, Any]], ground_truth: Sequence[Dict[str, Any]]):
    pred_rows = normalize_predictions(predictions)
    gt_rows = normalize_ground_truth(ground_truth)
    node_to_gt = defaultdict(set)
    edge_to_gt = defaultdict(set)
    for gt_index, gt in enumerate(gt_rows):
        for node in gt["nodes"]:
            node_to_gt[node].add(gt_index)
        for edge in gt["edges"]:
            edge_to_gt[edge].add(gt_index)
    matched = []
    for pred in pred_rows:
        # The original matcher never selects a zero-overlap GT: its initial
        # score is all zeros and updates use strict tuple comparison. Hence
        # restricting to nonzero-overlap GT rows is exactly equivalent. Keep
        # original GT order so tie behavior also remains unchanged.
        candidate_indices = set()
        for node in pred["nodes"]:
            candidate_indices.update(node_to_gt.get(node, ()))
        for edge in pred["edges"]:
            candidate_indices.update(edge_to_gt.get(edge, ()))
        candidate_gt = [gt for gt_index, gt in enumerate(gt_rows) if gt_index in candidate_indices]
        match = best_ground_truth_match(pred, candidate_gt)
        gt = match["ground_truth"]
        node_intersection = bool(gt and pred["nodes"] & gt["nodes"])
        edge_intersection = bool(gt and pred["edges"] & gt["edges"])
        node_j = jaccard(pred["nodes"], gt["nodes"]) if gt else 0.0
        edge_j = jaccard(pred["edges"], gt["edges"]) if gt else 0.0
        matched.append((pred, gt, node_intersection, edge_intersection, node_j, edge_j))

    rows = []
    diagnostic_rows = []
    for criterion, threshold, mode in CRITERIA:
        hit_count = 0
        gt_hit = set()
        group_counts = defaultdict(lambda: Counter(total=0, hit=0))
        for pred, gt, node_intersection, edge_intersection, node_j, edge_j in matched:
            hit = bool(gt) and is_hit(mode, threshold, node_intersection, edge_intersection, node_j, edge_j)
            if hit:
                hit_count += 1
                gt_hit.add(gt["motif_id"])
            if dataset == "amlworld" and method == "comot":
                candidate_type = str(pred.get("metadata", {}).get("candidate_type", ""))
                group = "anchor_only" if candidate_type == "amlworld_anchor_node_evidence" else "assembled_template"
                group_counts[group]["total"] += 1
                group_counts[group]["hit"] += int(hit)

        precision = hit_count / len(pred_rows) if pred_rows else 0.0
        recall = len(gt_hit) / len(gt_rows) if gt_rows else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "criterion": criterion,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_predictions": len(pred_rows),
            "num_hit_predictions": hit_count,
            "num_unique_gt_hit": len(gt_hit),
            "num_gt": len(gt_rows),
        })
        for group, counts in sorted(group_counts.items()):
            diagnostic_rows.append({
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "criterion": criterion,
                "prediction_group": group,
                "num_predictions": counts["total"],
                "num_hit_predictions": counts["hit"],
                "hit_rate": counts["hit"] / counts["total"] if counts["total"] else 0.0,
            })

    empty_edges = sum(not pred["edges"] for pred in pred_rows)
    duplicate_keys = Counter((pred["query_id"], tuple(sorted(pred["nodes"])), tuple(sorted(pred["edges"]))) for pred in pred_rows)
    run_sanity = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "num_predictions": len(pred_rows),
        "num_edge_empty_predictions": empty_edges,
        "num_duplicate_prediction_rows": sum(count - 1 for count in duplicate_keys.values() if count > 1),
        "num_duplicate_prediction_groups": sum(count > 1 for count in duplicate_keys.values()),
    }
    return rows, diagnostic_rows, run_sanity


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    values = list(values)
    mean = sum(values) / len(values) if values else 0.0
    if len(values) < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def aggregate(full_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in full_rows:
        groups[(row["dataset"], row["method"], row["criterion"], row["threshold"])].append(row)
    output = []
    for (dataset, method, criterion, threshold), rows in sorted(groups.items()):
        item = {"dataset": dataset, "method": method, "criterion": criterion, "threshold": threshold, "num_seeds": len(rows)}
        for metric in ("precision", "recall", "f1"):
            item[f"{metric}_mean"], item[f"{metric}_std"] = mean_std(float(row[metric]) for row in rows)
        for metric in ("num_predictions", "num_hit_predictions", "num_unique_gt_hit", "num_gt"):
            item[f"{metric}_mean"], item[f"{metric}_std"] = mean_std(float(row[metric]) for row in rows)
        output.append(item)
    return output


def top_compare(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {(row["dataset"], row["method"], row["criterion"]): row for row in summary_rows}
    methods = sorted({row["method"] for row in summary_rows})
    output = []
    for dataset in DATASETS:
        original_baselines = [lookup[(dataset, method, "Original")] for method in methods if method != "comot"]
        original_best = max(original_baselines, key=lambda row: (row["f1_mean"], row["method"]))
        for criterion, threshold, _ in CRITERIA:
            baseline_rows = [lookup[(dataset, method, criterion)] for method in methods if method != "comot"]
            strict_best = max(baseline_rows, key=lambda row: (row["f1_mean"], row["method"]))
            comot = lookup[(dataset, "comot", criterion)]
            output.append({
                "dataset": dataset,
                "criterion": criterion,
                "threshold": threshold,
                "comot_f1_mean": comot["f1_mean"],
                "comot_f1_std": comot["f1_std"],
                "original_strongest_baseline": original_best["method"],
                "original_strongest_baseline_original_f1": original_best["f1_mean"],
                "criterion_strongest_baseline": strict_best["method"],
                "criterion_strongest_baseline_f1": strict_best["f1_mean"],
                "comot_minus_best_baseline": comot["f1_mean"] - strict_best["f1_mean"],
                "comot_rank_first": comot["f1_mean"] >= strict_best["f1_mean"],
            })
    return output


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ground_truth_sanity(dataset: str, ground_truth: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gt_rows = normalize_ground_truth(ground_truth)
    return {
        "dataset": dataset,
        "num_gt": len(gt_rows),
        "num_gt_edge_empty": sum(not row["edges"] for row in gt_rows),
        "num_gt_node_empty": sum(not row["nodes"] for row in gt_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/strict_overlap_eval")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    summary_dir = PROJECT_ROOT / "outputs" / "summary"
    summary_candidates = [
        summary_dir / "main_comparison_4datasets.csv",
        summary_dir / "main_table.csv",
        summary_dir / "main_comparison.csv",
    ]
    summary_path = next((path for path in summary_candidates if path.exists()), None)
    if summary_path is None:
        raise FileNotFoundError(
            "Run bash scripts/make_paper_tables.sh before strict-overlap evaluation."
        )
    methods = discover_methods(summary_path)

    full_rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    run_sanity: List[Dict[str, Any]] = []
    gt_sanity = []
    for dataset in DATASETS:
        ground_truth = load_ground_truth(dataset)
        gt_sanity.append(ground_truth_sanity(dataset, ground_truth))
        for method in methods:
            for seed in seeds:
                run_dir = PROJECT_ROOT / "outputs" / dataset / method / f"seed_{seed}"
                rows, diagnostic_rows, sanity = evaluate_run(dataset, method, seed, load_predictions(run_dir), ground_truth)
                full_rows.extend(rows)
                diagnostics.extend(diagnostic_rows)
                run_sanity.append(sanity)

    summary_rows = aggregate(full_rows)
    compare_rows = top_compare(summary_rows)
    write_csv(output_dir / "strict_overlap_full.csv", full_rows)
    write_csv(output_dir / "strict_overlap_summary.csv", summary_rows)
    write_csv(output_dir / "strict_overlap_top_compare.csv", compare_rows)
    write_csv(output_dir / "strict_overlap_amlworld_diagnostics.csv", diagnostics)
    write_csv(output_dir / "strict_overlap_run_sanity.csv", run_sanity)
    write_csv(output_dir / "strict_overlap_gt_sanity.csv", gt_sanity)
    print(json.dumps({
        "output_dir": str(output_dir),
        "datasets": DATASETS,
        "methods": len(methods),
        "seeds": seeds,
        "run_rows": len(full_rows),
        "summary_rows": len(summary_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
