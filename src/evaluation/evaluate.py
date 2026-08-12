import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.metrics import DEFAULT_TOPKS, compute_recovery_metrics


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_predictions_jsonl(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "predictions" in data:
        return data["predictions"]
    raise ValueError(f"Unsupported prediction JSON structure: {path}")


def load_predictions_jsonl(path: Path) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "ground_truth" in data:
        return data["ground_truth"]
    raise ValueError(f"Unsupported ground-truth JSON structure: {path}")


def evaluate_predictions(
    predictions: Sequence[Dict[str, Any]],
    ground_truth: Sequence[Dict[str, Any]],
    output_dir: Path,
    topks: Optional[Sequence[int]] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, detail_rows = compute_recovery_metrics(predictions, ground_truth, topks=topks or DEFAULT_TOPKS)

    detail_path = output_dir / "prediction_eval_detail.csv"
    write_detail_csv(detail_rows, detail_path)

    metrics_json = output_dir / "metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metrics_csv = output_dir / "metrics.csv"
    write_metrics_csv(metrics, metrics_csv)

    summary_txt = output_dir / "summary.txt"
    write_summary_txt(metrics, summary_txt)

    summary = {
        "num_predictions": metrics["efficiency"]["num_candidates"],
        "num_ground_truth": len(ground_truth),
        "candidate_purity": metrics["legacy_overlap"]["candidate_purity"],
        "motif_recall": metrics["legacy_overlap"]["motif_recall"],
        "topk": metrics["legacy_overlap"]["topk"],
    }

    summary_path = output_dir / "prediction_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "detail": detail_path,
        "summary": summary_path,
        "metrics_json": metrics_json,
        "metrics_csv": metrics_csv,
        "summary_txt": summary_txt,
    }


def write_detail_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "rank",
        "query_id",
        "score",
        "is_hit",
        "node_hit",
        "edge_hit",
        "exact_match",
        "matched_motif_id",
        "node_jaccard",
        "edge_jaccard",
        "motif_jaccard",
        "structure_consistency",
        "num_predicted_nodes",
        "num_predicted_edges",
        "runtime",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(metrics: Dict[str, Any], path: Path) -> None:
    flat = flatten_metrics(metrics)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in sorted(flat):
            writer.writerow({"metric": key, "value": flat[key]})


def write_summary_txt(metrics: Dict[str, Any], path: Path) -> None:
    lines = [
        "Unified CoMot Evaluation Summary",
        "",
        f"Candidates: {metrics['efficiency']['num_candidates']}",
        f"Node Precision/Recall/F1: {metrics['node_level']['precision']:.6f} / {metrics['node_level']['recall']:.6f} / {metrics['node_level']['f1']:.6f}",
        f"Edge Precision/Recall/F1: {metrics['edge_level']['precision']:.6f} / {metrics['edge_level']['recall']:.6f} / {metrics['edge_level']['f1']:.6f}",
        f"Motif Exact/Partial/Jaccard: {metrics['motif_level']['exact_match']:.6f} / {metrics['motif_level']['partial_match']:.6f} / {metrics['motif_level']['jaccard_similarity']:.6f}",
        f"Structure Consistency: {metrics['motif_level']['structure_consistency']:.6f}",
        f"AUC: {metrics['ranking']['auc']}",
        f"Average Precision: {metrics['ranking']['average_precision']}",
        f"Runtime Total: {metrics['efficiency']['runtime_total']:.6f}",
        f"Memory Usage: {metrics['efficiency']['memory_usage']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        next_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, next_key))
        else:
            flat[next_key] = value
    return flat


def parse_topks(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standard CoMot predictions against standard ground truth.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl or JSON predictions.")
    parser.add_argument("--ground_truth", required=True, help="Path to ground_truth.json.")
    parser.add_argument("--output_dir", required=True, help="Directory for metrics.json, metrics.csv, and summary.txt.")
    parser.add_argument("--top_k", default="10,20,50,100,200,500,1000", help="Comma-separated Hit@K cutoffs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = evaluate_predictions(
        predictions=load_predictions(Path(args.predictions)),
        ground_truth=load_ground_truth(Path(args.ground_truth)),
        output_dir=Path(args.output_dir),
        topks=parse_topks(args.top_k),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
