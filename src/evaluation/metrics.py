import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_TOPKS = [10, 20, 50, 100, 200, 500, 1000]


def parse_json_list(value) -> List:
    if value is None:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def transaction_overlap_hits(candidate_tx_ids: Iterable[int], gt_motif_to_txs: Dict[str, Set[int]]) -> List[str]:
    """Return GT motif ids whose transaction sets overlap a candidate."""
    cand_txs = set(candidate_tx_ids)
    return [motif_id for motif_id, motif_txs in gt_motif_to_txs.items() if cand_txs & motif_txs]


def purity_and_recall(hit_rows: Iterable[Tuple[bool, Iterable[str]]], num_gt_motifs: int) -> Dict[str, float]:
    """Compute legacy candidate purity and motif recall from hit rows."""
    rows = list(hit_rows)
    if not rows:
        return {"candidate_purity": 0.0, "motif_recall": 0.0}

    num_hit = sum(1 for is_hit, _ in rows if is_hit)
    gt_hit = set()
    for _, motif_ids in rows:
        gt_hit.update(motif_ids)

    return {
        "candidate_purity": num_hit / len(rows),
        "motif_recall": len(gt_hit) / num_gt_motifs if num_gt_motifs else 0.0,
    }


def as_str_set(values: Optional[Iterable[Any]]) -> Set[str]:
    if values is None:
        return set()
    return {str(value) for value in values if value is not None}


def compute_recovery_metrics(
    predictions: Sequence[Dict[str, Any]],
    ground_truth: Sequence[Dict[str, Any]],
    topks: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute unified node, edge, motif, ranking, and efficiency metrics."""
    topks = list(topks or DEFAULT_TOPKS)
    gt_rows = normalize_ground_truth(ground_truth)
    pred_rows = normalize_predictions(predictions)

    detail_rows = []
    node_tp = node_fp = node_fn = 0
    edge_tp = edge_fp = edge_fn = 0
    exact_matches = 0
    partial_matches = 0
    jaccards = []
    structure_scores = []
    recovered_gt_nodes: Set[str] = set()
    recovered_gt_edges: Set[str] = set()
    recovered_motifs: Set[str] = set()
    labels = []
    scores = []

    all_gt_nodes = set().union(*(gt["nodes"] for gt in gt_rows)) if gt_rows else set()
    all_gt_edges = set().union(*(gt["edges"] for gt in gt_rows)) if gt_rows else set()

    for rank, pred in enumerate(pred_rows, start=1):
        match = best_ground_truth_match(pred, gt_rows)
        gt = match["ground_truth"]
        node_intersection = pred["nodes"] & gt["nodes"] if gt else set()
        edge_intersection = pred["edges"] & gt["edges"] if gt else set()
        is_partial = bool(node_intersection or edge_intersection)
        is_exact = bool(gt) and exact_match(pred, gt)
        combined_jaccard = match["combined_jaccard"]
        structure_consistency = match["structure_consistency"]

        node_tp += len(node_intersection)
        node_fp += len(pred["nodes"] - (gt["nodes"] if gt else set()))
        node_fn += len((gt["nodes"] if gt else set()) - pred["nodes"])
        edge_tp += len(edge_intersection)
        edge_fp += len(pred["edges"] - (gt["edges"] if gt else set()))
        edge_fn += len((gt["edges"] if gt else set()) - pred["edges"])

        if is_exact:
            exact_matches += 1
        if is_partial:
            partial_matches += 1
        if gt and is_partial:
            recovered_motifs.add(gt["motif_id"])
        recovered_gt_nodes.update(node_intersection)
        recovered_gt_edges.update(edge_intersection)
        jaccards.append(combined_jaccard)
        structure_scores.append(structure_consistency)
        labels.append(1 if is_partial else 0)
        scores.append(pred["score"])

        detail_rows.append(
            {
                "rank": rank,
                "query_id": pred["query_id"],
                "score": pred["score"],
                "is_hit": is_partial,
                "node_hit": bool(node_intersection),
                "edge_hit": bool(edge_intersection),
                "exact_match": is_exact,
                "matched_motif_id": gt["motif_id"] if gt and is_partial else "",
                "node_jaccard": jaccard(pred["nodes"], gt["nodes"]) if gt else 0.0,
                "edge_jaccard": jaccard(pred["edges"], gt["edges"]) if gt else 0.0,
                "motif_jaccard": combined_jaccard,
                "structure_consistency": structure_consistency,
                "num_predicted_nodes": len(pred["nodes"]),
                "num_predicted_edges": len(pred["edges"]),
                "runtime": pred["runtime"],
            }
        )

    node_prf = precision_recall_f1(node_tp, node_fp, node_fn)
    edge_prf = precision_recall_f1(edge_tp, edge_fp, edge_fn)
    runtime_values = [row["runtime"] for row in pred_rows]
    metrics: Dict[str, Any] = {
        "node_level": {
            **node_prf,
            "hit_at_k": hit_at_k(detail_rows, topks, "node_hit"),
            "mrr": reciprocal_rank(detail_rows, "node_hit"),
            "unique_recall": len(recovered_gt_nodes) / len(all_gt_nodes) if all_gt_nodes else 0.0,
        },
        "edge_level": {
            **edge_prf,
            "hit_at_k": hit_at_k(detail_rows, topks, "edge_hit"),
            "unique_recall": len(recovered_gt_edges) / len(all_gt_edges) if all_gt_edges else 0.0,
        },
        "motif_level": {
            "exact_match": exact_matches / len(pred_rows) if pred_rows else 0.0,
            "partial_match": partial_matches / len(pred_rows) if pred_rows else 0.0,
            "jaccard_similarity": mean(jaccards),
            "structure_consistency": mean(structure_scores),
            "motif_recall": len(recovered_motifs) / len(gt_rows) if gt_rows else 0.0,
        },
        "ranking": {
            "auc": auc_from_scores(labels, scores),
            "average_precision": average_precision_from_scores(labels, scores),
        },
        "efficiency": {
            "runtime_total": max(runtime_values, default=0.0),
            "runtime_mean": mean(runtime_values),
            "runtime_max": max(runtime_values, default=0.0),
            "memory_usage": memory_usage(predictions),
            "num_candidates": len(pred_rows),
        },
        "legacy_overlap": {
            "candidate_purity": partial_matches / len(pred_rows) if pred_rows else 0.0,
            "motif_recall": len(recovered_motifs) / len(gt_rows) if gt_rows else 0.0,
            "topk": legacy_topk(detail_rows, topks, gt_rows),
        },
    }
    return metrics, detail_rows


def normalize_ground_truth(ground_truth: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for idx, item in enumerate(ground_truth):
        rows.append(
            {
                "motif_id": str(item.get("motif_instance_id") or item.get("query_id") or idx),
                "query_id": str(item.get("query_id", "")),
                "nodes": as_str_set(item.get("true_nodes")),
                "edges": as_str_set(item.get("true_edges")),
                "partitions": as_str_set(item.get("true_partitions")),
            }
        )
    return rows


def normalize_predictions(predictions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for idx, item in enumerate(predictions):
        rows.append(
            {
                "prediction_id": str(item.get("prediction_id", idx)),
                "query_id": str(item.get("query_id", "")),
                "nodes": as_str_set(item.get("predicted_nodes")),
                "edges": as_str_set(item.get("predicted_edges")),
                "score": safe_float(item.get("score", 0.0)),
                "runtime": safe_float(item.get("runtime", 0.0)),
                "metadata": item.get("metadata", {}) or {},
            }
        )
    rows.sort(key=lambda row: (-row["score"], row["prediction_id"]))
    return rows


def best_ground_truth_match(pred: Dict[str, Any], gt_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    best = {"ground_truth": None, "combined_jaccard": 0.0, "structure_consistency": 0.0}
    for gt in gt_rows:
        query_compatible = not pred["query_id"] or not gt["query_id"] or pred["query_id"] == gt["query_id"]
        node_score = jaccard(pred["nodes"], gt["nodes"])
        edge_score = jaccard(pred["edges"], gt["edges"])
        combined_score = jaccard(prefixed_union(pred["nodes"], pred["edges"]), prefixed_union(gt["nodes"], gt["edges"]))
        if not query_compatible and combined_score == 0.0:
            continue
        structure_score = edge_score if pred["edges"] or gt["edges"] else node_score
        if (combined_score, structure_score, edge_score, node_score) > (
            best["combined_jaccard"],
            best["structure_consistency"],
            0.0,
            0.0,
        ):
            best = {
                "ground_truth": gt,
                "combined_jaccard": combined_score,
                "structure_consistency": structure_score,
            }
    return best


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def hit_at_k(detail_rows: Sequence[Dict[str, Any]], topks: Sequence[int], key: str) -> Dict[str, float]:
    return {str(k): 1.0 if any(row.get(key, False) for row in detail_rows[:k]) else 0.0 for k in topks if k > 0}


def reciprocal_rank(detail_rows: Sequence[Dict[str, Any]], key: str) -> float:
    for row in detail_rows:
        if row.get(key, False):
            return 1.0 / row["rank"]
    return 0.0


def exact_match(pred: Dict[str, Any], gt: Dict[str, Any]) -> bool:
    node_ok = pred["nodes"] == gt["nodes"] if gt["nodes"] else not pred["nodes"]
    edge_ok = pred["edges"] == gt["edges"] if gt["edges"] else not pred["edges"]
    return node_ok and edge_ok


def jaccard(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def prefixed_union(nodes: Set[str], edges: Set[str]) -> Set[str]:
    return {f"n:{node}" for node in nodes} | {f"e:{edge}" for edge in edges}


def auc_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos_score in positives:
        for neg_score in negatives:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / total


def average_precision_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(labels)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def legacy_topk(detail_rows: Sequence[Dict[str, Any]], topks: Sequence[int], gt_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    output = {}
    for k in topks:
        rows_at_k = detail_rows[:k]
        motifs = {row["matched_motif_id"] for row in rows_at_k if row.get("matched_motif_id")}
        output[str(k)] = {
            "candidate_purity": sum(1 for row in rows_at_k if row["is_hit"]) / len(rows_at_k) if rows_at_k else 0.0,
            "motif_recall": len(motifs) / len(gt_rows) if gt_rows else 0.0,
            "num_hits": sum(1 for row in rows_at_k if row["is_hit"]),
            "num_unique_gt_hits": len(motifs),
        }
    return output


def memory_usage(predictions: Sequence[Dict[str, Any]]) -> Optional[float]:
    values = []
    for pred in predictions:
        metadata = pred.get("metadata", {}) or {}
        for key in ("memory_usage", "memory_mb", "peak_memory_mb"):
            if key in metadata:
                values.append(safe_float(metadata[key]))
    return max(values) if values else None


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
