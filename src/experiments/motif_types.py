import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.metrics import as_str_set, best_ground_truth_match, normalize_ground_truth, normalize_predictions, precision_recall_f1


DATASET = "amlworld"
METHOD = "comot"
SEED_CANDIDATES = [0, 1, 2]
CSV_COLUMNS = [
    "dataset",
    "method",
    "seeds",
    "num_seeds",
    "motif_type",
    "num_instances",
    "avg_num_nodes",
    "avg_num_edges",
    "avg_num_partitions",
    "motif_f1",
    "edge_f1",
    "node_f1",
    "motif_jaccard",
    "motif_recall",
    "avg_rank_of_first_hit",
    "median_rank_of_first_hit",
    "num_hit_queries",
    "num_missed_queries",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        return data["predictions"]
    raise ValueError(f"Unsupported prediction file format: {path}")


def motif_type_lookup(query_motifs: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    lookup = {}
    for item in query_motifs:
        query_id = str(item.get("query_id", ""))
        if query_id:
            lookup[query_id] = str(item.get("motif_type", "UNKNOWN"))
    return lookup


def enrich_ground_truth(ground_truth: Sequence[Dict[str, Any]], query_motifs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = motif_type_lookup(query_motifs)
    output = []
    for idx, item in enumerate(ground_truth):
        row = dict(item)
        query_id = str(row.get("query_id") or row.get("motif_instance_id") or idx)
        row["query_id"] = query_id
        row["motif_instance_id"] = str(row.get("motif_instance_id") or query_id)
        row["motif_type"] = str(row.get("motif_type") or lookup.get(query_id, "UNKNOWN"))
        output.append(row)
    return output


def discover_existing_seeds(output_root: Path) -> List[int]:
    seeds = []
    for seed in SEED_CANDIDATES:
        seed_dir = output_root / DATASET / METHOD / f"seed_{seed}"
        if (seed_dir / "metrics.json").exists() and prediction_path(seed_dir).exists():
            seeds.append(seed)
    return seeds


def prediction_path(seed_dir: Path) -> Path:
    jsonl = seed_dir / "predictions.jsonl"
    if jsonl.exists():
        return jsonl
    return seed_dir / "predictions.json"


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def harmonic(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def fmt(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.6f}"
    return value


def evaluate_seed(
    seed: int,
    predictions: Sequence[Dict[str, Any]],
    ground_truth: Sequence[Dict[str, Any]],
    motif_types: Sequence[str],
) -> List[Dict[str, Any]]:
    gt_rows = normalize_ground_truth(ground_truth)
    gt_by_id = {row["motif_id"]: row for row in gt_rows}
    type_by_id = {
        str(item.get("motif_instance_id") or item.get("query_id")): str(item.get("motif_type", "UNKNOWN"))
        for item in ground_truth
    }
    pred_rows = normalize_predictions(predictions)

    first_hit: Dict[str, Dict[str, Any]] = {}
    candidate_hits_by_type: Dict[str, int] = defaultdict(int)
    jaccard_hits_by_motif: Dict[str, float] = {}

    for rank, pred in enumerate(pred_rows, start=1):
        match = best_ground_truth_match(pred, gt_rows)
        gt = match["ground_truth"]
        if not gt:
            continue
        node_intersection = pred["nodes"] & gt["nodes"]
        edge_intersection = pred["edges"] & gt["edges"]
        if not (node_intersection or edge_intersection):
            continue

        motif_id = gt["motif_id"]
        motif_type = type_by_id.get(motif_id, "UNKNOWN")
        candidate_hits_by_type[motif_type] += 1
        if motif_id not in first_hit:
            first_hit[motif_id] = {
                "rank": rank,
                "prediction": pred,
                "ground_truth": gt,
                "node_intersection": node_intersection,
                "edge_intersection": edge_intersection,
                "motif_jaccard": match["combined_jaccard"],
            }
            jaccard_hits_by_motif[motif_id] = match["combined_jaccard"]

    rows = []
    for motif_type in motif_types:
        gt_items = [item for item in ground_truth if str(item.get("motif_type", "UNKNOWN")) == motif_type]
        motif_ids = [str(item.get("motif_instance_id") or item.get("query_id")) for item in gt_items]

        node_tp = node_fp = node_fn = 0
        edge_tp = edge_fp = edge_fn = 0
        ranks = []
        jaccards = []
        hit_count = 0

        for motif_id in motif_ids:
            gt = gt_by_id[motif_id]
            hit = first_hit.get(motif_id)
            if hit:
                pred = hit["prediction"]
                node_intersection = hit["node_intersection"]
                edge_intersection = hit["edge_intersection"]
                node_tp += len(node_intersection)
                node_fp += len(pred["nodes"] - gt["nodes"])
                node_fn += len(gt["nodes"] - pred["nodes"])
                edge_tp += len(edge_intersection)
                edge_fp += len(pred["edges"] - gt["edges"])
                edge_fn += len(gt["edges"] - pred["edges"])
                ranks.append(float(hit["rank"]))
                jaccards.append(float(hit["motif_jaccard"]))
                hit_count += 1
            else:
                node_fn += len(gt["nodes"])
                edge_fn += len(gt["edges"])
                jaccards.append(0.0)

        num_instances = len(gt_items)
        motif_recall = hit_count / num_instances if num_instances else 0.0
        duplicate_denominator = candidate_hits_by_type.get(motif_type, 0)
        motif_precision = hit_count / duplicate_denominator if duplicate_denominator else 0.0
        node_prf = precision_recall_f1(node_tp, node_fp, node_fn)
        edge_prf = precision_recall_f1(edge_tp, edge_fp, edge_fn)

        rows.append(
            {
                "seed": seed,
                "motif_type": motif_type,
                "num_instances": num_instances,
                "avg_num_nodes": mean([len(as_str_set(item.get("true_nodes"))) for item in gt_items]),
                "avg_num_edges": mean([len(as_str_set(item.get("true_edges"))) for item in gt_items]),
                "avg_num_partitions": mean([len(as_str_set(item.get("true_partitions"))) for item in gt_items]),
                "motif_f1": harmonic(motif_precision, motif_recall),
                "edge_f1": edge_prf["f1"],
                "node_f1": node_prf["f1"],
                "motif_jaccard": mean(jaccards),
                "motif_recall": motif_recall,
                "avg_rank_of_first_hit": mean(ranks),
                "median_rank_of_first_hit": median(ranks),
                "num_hit_queries": hit_count,
                "num_missed_queries": num_instances - hit_count,
            }
        )
    return rows


def aggregate_seed_rows(seed_rows: Sequence[Dict[str, Any]], seeds: Sequence[int]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["motif_type"])].append(row)

    output = []
    for motif_type in sorted(grouped):
        rows = grouped[motif_type]
        first = rows[0]
        agg = {
            "dataset": DATASET,
            "method": METHOD,
            "seeds": ",".join(str(seed) for seed in seeds),
            "num_seeds": len(seeds),
            "motif_type": motif_type,
            "num_instances": int(first["num_instances"]),
            "avg_num_nodes": first["avg_num_nodes"],
            "avg_num_edges": first["avg_num_edges"],
            "avg_num_partitions": first["avg_num_partitions"],
        }
        for key in [
            "motif_f1",
            "edge_f1",
            "node_f1",
            "motif_jaccard",
            "motif_recall",
            "avg_rank_of_first_hit",
            "median_rank_of_first_hit",
            "num_hit_queries",
            "num_missed_queries",
        ]:
            agg[key] = mean([float(row[key]) for row in rows])
        output.append(agg)
    output.sort(key=lambda row: (-float(row["motif_f1"]), row["motif_type"]))
    return output


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in CSV_COLUMNS})


def latex_escape(value: Any) -> str:
    text = str(value)
    for src, dst in {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}.items():
        text = text.replace(src, dst)
    return text


def write_latex(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    columns = [
        ("motif_type", "Motif type"),
        ("num_instances", "N"),
        ("motif_f1", "Motif F1"),
        ("edge_f1", "Edge F1"),
        ("node_f1", "Node F1"),
        ("motif_jaccard", "Jaccard"),
        ("motif_recall", "Recall"),
        ("avg_rank_of_first_hit", "Avg. rank"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        " & ".join(label for _, label in columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        values = [latex_escape(fmt(row[key])) for key, _ in columns]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{AMLWorld CoMot motif-type breakdown.}",
            r"\label{tab:amlworld_motif_type_breakdown}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AMLWorld CoMot motif-type breakdown from existing runs.")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--summary_dir", default="outputs/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = PROJECT_ROOT / args.output_root
    summary_dir = PROJECT_ROOT / args.summary_dir
    dataset_dir = PROJECT_ROOT / "data" / "processed" / DATASET

    query_motifs = load_json(dataset_dir / "query_motifs.json")
    ground_truth = enrich_ground_truth(load_json(dataset_dir / "ground_truth.json"), query_motifs)
    motif_types = sorted({str(item.get("motif_type", "UNKNOWN")) for item in ground_truth})
    seeds = discover_existing_seeds(output_root)
    if not seeds:
        raise FileNotFoundError(f"No existing {DATASET}/{METHOD}/seed_*/metrics.json and predictions found under {output_root}")

    seed_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = output_root / DATASET / METHOD / f"seed_{seed}"
        predictions = load_predictions(prediction_path(seed_dir))
        seed_rows.extend(evaluate_seed(seed, predictions, ground_truth, motif_types))

    rows = aggregate_seed_rows(seed_rows, seeds)
    csv_path = summary_dir / "motif_types_amlworld.csv"
    latex_path = summary_dir / "motif_types_amlworld_latex.tex"
    write_csv(rows, csv_path)
    write_latex(rows, latex_path)
    print(json.dumps({"seeds": seeds, "num_rows": len(rows), "csv": str(csv_path), "latex": str(latex_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
