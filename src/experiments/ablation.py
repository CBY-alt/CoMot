import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["amlworld", "elliptic", "dblp", "web_google"]
SEEDS = [0, 1, 2]
METHOD = "comot"
AML_TAG = "five_party"
TOP_K_RETURNED = 1000

VARIANTS = [
    "Full CoMot",
    "w/o Structural Evidence",
    "w/o Compatibility Filtering",
    "w/o Template Constraints",
    "w/o Candidate Prioritization",
]

BY_SEED_COLUMNS = [
    "dataset",
    "method",
    "seed",
    "ablation_variant",
    "status",
    "motif_f1",
    "edge_f1",
    "node_f1",
    "motif_jaccard",
    "num_candidates",
    "purity",
    "runtime",
    "source",
    "notes",
]

SUMMARY_COLUMNS = [
    "dataset",
    "method",
    "ablation_variant",
    "num_seeds",
    "completed_seeds",
    "status",
    "motif_f1",
    "edge_f1",
    "node_f1",
    "motif_jaccard",
    "num_candidates",
    "purity",
    "runtime",
    "notes",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def parse_main_cell(value: Any) -> Any:
    if value in (None, "", "N/A"):
        return None
    text = str(value).replace("+/-", "+/-").strip()
    text = text.split("+/-", 1)[0].strip()
    try:
        return float(text)
    except ValueError:
        return None


def load_main_table_comot(summary_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = summary_dir / "main_comparison_4datasets.csv"
    if not path.exists():
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") != METHOD:
                continue
            output[str(row["dataset"])] = {
                "motif_f1": parse_main_cell(row.get("motif_f1")),
                "edge_f1": parse_main_cell(row.get("edge_f1")),
                "node_f1": parse_main_cell(row.get("node_f1")),
                "motif_jaccard": parse_main_cell(row.get("motif_jaccard")),
                "num_candidates": parse_main_cell(row.get("num_candidates")),
                "runtime": parse_main_cell(row.get("runtime")),
            }
    return output


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def as_set(values: Iterable[Any]) -> Set[str]:
    return {str(value) for value in values if value is not None and str(value) not in {"", "nan", "None"}}


def strip_partition_prefix(value: Any) -> str:
    text = str(value)
    return text.split("::", 1)[1] if "::" in text else text


def load_ground_truth(dataset: str) -> List[Dict[str, Any]]:
    rows = []
    for idx, item in enumerate(load_json(PROJECT_ROOT / "data" / "processed" / dataset / "ground_truth.json")):
        motif_id = str(item.get("motif_instance_id") or item.get("query_id") or idx)
        row = {
            "motif_id": motif_id,
            "query_id": str(item.get("query_id", "")),
            "nodes": as_set(item.get("true_nodes") or []),
            "edges": as_set(item.get("true_edges") or []),
        }
        rows.append(row)
    return rows


def candidate_record(
    candidate_id: str,
    query_id: str,
    candidate_type: str,
    nodes: Iterable[Any],
    edges: Iterable[Any],
    score: Any = 0.0,
    metadata: Dict[str, Any] | None = None,
    order: int = 0,
) -> Dict[str, Any]:
    return {
        "candidate_id": str(candidate_id),
        "query_id": str(query_id),
        "candidate_type": str(candidate_type),
        "nodes": {strip_partition_prefix(node) for node in nodes},
        "edges": as_set(edges),
        "score": safe_float(score),
        "metadata": metadata or {},
        "order": order,
    }


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_standard_pool(run_dir: Path) -> Tuple[List[Dict[str, Any]], Path]:
    path = run_dir / "comot_standard_candidates.jsonl"
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(candidate_record(
                candidate_id=f"{item.get('query_id', '')}:{idx}",
                query_id=item.get("query_id", ""),
                candidate_type=item.get("candidate_type", "unknown"),
                nodes=item.get("node_ids", []),
                edges=item.get("edge_ids", []),
                score=item.get("score", 0.0),
                metadata=item.get("metadata", {}) or {},
                order=idx,
            ))
    return records, path


def load_aml_raw_pool(run_dir: Path) -> Tuple[List[Dict[str, Any]], Path]:
    candidate_dir = run_dir / f"semotif_candidates_{AML_TAG}"
    records: List[Dict[str, Any]] = []
    for name in ["chain_candidates.csv", "fan_candidates.csv", "cycle_candidates.csv"]:
        path = candidate_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            records.append(candidate_record(
                candidate_id=row.get("candidate_id", len(records)),
                query_id=str(row.get("candidate_type", "comot")),
                candidate_type=row.get("candidate_type", name.split("_", 1)[0]),
                nodes=parse_json_list(row.get("nodes_json")),
                edges=parse_json_list(row.get("tx_ids_json")),
                score=row.get("score_mean", 0.0),
                metadata={},
                order=len(records),
            ))
    return records, candidate_dir


def load_aml_reranked_pool(run_dir: Path, raw_records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Path]:
    path = run_dir / f"semotif_candidates_{AML_TAG}_reranked" / "candidates_reranked.csv"
    if not path.exists():
        return list(raw_records), path
    records: List[Dict[str, Any]] = []
    df = pd.read_csv(path)
    for idx, row in df.iterrows():
        records.append(candidate_record(
            candidate_id=row.get("candidate_id", idx),
            query_id=str(row.get("candidate_type", "comot")),
            candidate_type=row.get("candidate_type", "unknown"),
            nodes=parse_json_list(row.get("nodes_json")),
            edges=parse_json_list(row.get("tx_ids_json")),
            score=row.get("rerank_score", row.get("score_mean", 0.0)),
            metadata={},
            order=idx,
        ))
    return records, path


def load_full_pool(dataset: str, run_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path]:
    if dataset == "amlworld":
        raw, _ = load_aml_raw_pool(run_dir)
        ranked, source = load_aml_reranked_pool(run_dir, raw)
        return raw, ranked, source
    ranked, source = load_standard_pool(run_dir)
    return ranked, ranked, source


def aml_scored_pairs_pool(run_dir: Path) -> Tuple[List[Dict[str, Any]], Path]:
    path = run_dir / f"semotif_orchestrator_{AML_TAG}" / "alignment_pairs_scored.csv"
    records: List[Dict[str, Any]] = []
    usecols = ["evidence_node_id_s", "evidence_node_id_r", "tx_id_s", "tx_id_r", "compat_score", "best_hypothesis"]
    for idx, row in pd.read_csv(path, usecols=usecols).iterrows():
        records.append(candidate_record(
            candidate_id=f"scored_pair_{idx}",
            query_id=str(row.get("best_hypothesis", "pair")),
            candidate_type=f"NO_COMPAT_{row.get('best_hypothesis', 'PAIR')}",
            nodes=[row["evidence_node_id_s"], row["evidence_node_id_r"]],
            edges=[row["tx_id_s"], row["tx_id_r"]],
            score=row["compat_score"],
            order=idx,
        ))
    return records, path


def aml_compat_edges_pool(run_dir: Path) -> Tuple[List[Dict[str, Any]], Path]:
    path = run_dir / f"semotif_orchestrator_{AML_TAG}" / "compatibility_graph_edges.csv"
    records: List[Dict[str, Any]] = []
    usecols = ["src_node", "dst_node", "weight", "tx_id_sender", "tx_id_receiver", "best_hypothesis"]
    for idx, row in pd.read_csv(path, usecols=usecols).iterrows():
        records.append(candidate_record(
            candidate_id=f"compat_edge_{idx}",
            query_id=str(row.get("best_hypothesis", "pair")),
            candidate_type=f"NO_TEMPLATE_{row.get('best_hypothesis', 'PAIR')}",
            nodes=[row["src_node"], row["dst_node"]],
            edges=[row["tx_id_sender"], row["tx_id_receiver"]],
            score=row["weight"],
            order=idx,
        ))
    return records, path


def load_edges_for_standard(dataset: str) -> Dict[str, List[Tuple[str, str]]]:
    path = PROJECT_ROOT / "data" / "processed" / dataset / "edges.csv"
    adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for row in pd.read_csv(path, usecols=["edge_id", "src", "dst"]).itertuples(index=False):
        edge_id = str(row.edge_id)
        src = str(row.src)
        dst = str(row.dst)
        adjacency[src].append((dst, edge_id))
        adjacency[dst].append((src, edge_id))
    for node in adjacency:
        adjacency[node].sort(key=lambda item: (item[1], item[0]))
    return adjacency


def standard_anchor_edge_pool(dataset: str, cap_per_anchor: int, candidate_type: str) -> Tuple[List[Dict[str, Any]], Path]:
    data_dir = PROJECT_ROOT / "data" / "processed" / dataset
    queries = load_json(data_dir / "method_queries.json")
    adjacency = load_edges_for_standard(dataset)
    records: List[Dict[str, Any]] = []
    seen = set()
    for query in queries:
        query_id = str(query.get("query_id", ""))
        for anchor in [str(node) for node in query.get("anchor_nodes", [])]:
            for neighbor, edge_id in adjacency.get(anchor, [])[:cap_per_anchor]:
                key = (query_id, anchor, neighbor, edge_id, candidate_type)
                if key in seen:
                    continue
                seen.add(key)
                records.append(candidate_record(
                    candidate_id=f"{candidate_type}_{len(records)}",
                    query_id=query_id,
                    candidate_type=candidate_type,
                    nodes=[anchor, neighbor],
                    edges=[edge_id],
                    score=0.0,
                    order=len(records),
                ))
    return records, data_dir / "edges.csv"


def nonstructural_score(record: Dict[str, Any]) -> float:
    meta = record.get("metadata", {}) or {}
    return (
        safe_float(meta.get("compactness_score"))
        + safe_float(meta.get("cross_partition_support_score"))
        + safe_float(meta.get("multi_party_support_score"))
        + safe_float(meta.get("type_bonus"))
        - safe_float(meta.get("size_penalty"))
        - 0.001 * (len(record["nodes"]) + len(record["edges"]))
    )


def order_records(records: Sequence[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "score":
        return sorted(records, key=lambda r: (-r["score"], r["order"], r["candidate_id"]))
    if mode == "nonstructural":
        return sorted(records, key=lambda r: (-nonstructural_score(r), r["order"], r["candidate_id"]))
    if mode == "unprioritized":
        return sorted(records, key=lambda r: (r["query_id"], r["candidate_type"], sorted(r["edges"]), sorted(r["nodes"]), r["candidate_id"]))
    return list(records)


def records_to_eval_predictions(records: Sequence[Dict[str, Any]], top_k: int = TOP_K_RETURNED) -> List[Dict[str, Any]]:
    predictions = []
    for idx, record in enumerate(records[:top_k]):
        predictions.append({
            "prediction_id": record["candidate_id"],
            "query_id": record["query_id"],
            "nodes": set(record["nodes"]),
            "edges": set(record["edges"]),
            "score": record["score"],
        })
    return predictions


def prefixed_union(nodes: Set[str], edges: Set[str]) -> Set[str]:
    return {f"n:{node}" for node in nodes} | {f"e:{edge}" for edge in edges}


def jaccard(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def best_ground_truth_match(pred: Dict[str, Any], gt_rows: Sequence[Dict[str, Any]]) -> Tuple[Any, float]:
    best_gt = None
    best_score = 0.0
    best_key = (0.0, 0.0, 0.0, 0.0)
    for gt in gt_rows:
        query_compatible = not pred.get("query_id") or not gt.get("query_id") or pred.get("query_id") == gt.get("query_id")
        node_score = jaccard(pred["nodes"], gt["nodes"])
        edge_score = jaccard(pred["edges"], gt["edges"])
        combined_score = jaccard(prefixed_union(pred["nodes"], pred["edges"]), prefixed_union(gt["nodes"], gt["edges"]))
        if not query_compatible and combined_score == 0.0:
            continue
        structure_score = edge_score if pred["edges"] or gt["edges"] else node_score
        key = (combined_score, structure_score, edge_score, node_score)
        if key > best_key:
            best_key = key
            best_gt = gt
            best_score = combined_score
    return best_gt, best_score


def evaluate_predictions(predictions: Sequence[Dict[str, Any]], gt_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    node_tp = node_fp = node_fn = 0
    edge_tp = edge_fp = edge_fn = 0
    partial_matches = 0
    recovered_motifs: Set[str] = set()
    jaccards = []
    for pred in predictions:
        best_gt, combined_jaccard = best_ground_truth_match(pred, gt_rows)
        if best_gt is None:
            node_fp += len(pred["nodes"])
            edge_fp += len(pred["edges"])
            jaccards.append(0.0)
            continue
        node_intersection = pred["nodes"] & best_gt["nodes"]
        edge_intersection = pred["edges"] & best_gt["edges"]
        if node_intersection or edge_intersection:
            partial_matches += 1
            recovered_motifs.add(best_gt["motif_id"])
        node_tp += len(node_intersection)
        node_fp += len(pred["nodes"] - best_gt["nodes"])
        node_fn += len(best_gt["nodes"] - pred["nodes"])
        edge_tp += len(edge_intersection)
        edge_fp += len(pred["edges"] - best_gt["edges"])
        edge_fn += len(best_gt["edges"] - pred["edges"])
        jaccards.append(combined_jaccard)
    node = prf(node_tp, node_fp, node_fn)
    edge = prf(edge_tp, edge_fp, edge_fn)
    purity = partial_matches / len(predictions) if predictions else 0.0
    recall = len(recovered_motifs) / len(gt_rows) if gt_rows else 0.0
    motif_f1 = 2.0 * purity * recall / (purity + recall) if purity + recall else 0.0
    return {
        "motif_f1": motif_f1,
        "edge_f1": edge["f1"],
        "node_f1": node["f1"],
        "motif_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
        "purity": purity,
    }


def main_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    partial = safe_float(metrics.get("motif_level", {}).get("partial_match"))
    recall = safe_float(metrics.get("motif_level", {}).get("motif_recall"))
    motif_f1 = 0.0 if partial + recall == 0 else 2.0 * partial * recall / (partial + recall)
    return {
        "motif_f1": motif_f1,
        "edge_f1": safe_float(metrics.get("edge_level", {}).get("f1")),
        "node_f1": safe_float(metrics.get("node_level", {}).get("f1")),
        "motif_jaccard": safe_float(metrics.get("motif_level", {}).get("jaccard_similarity")),
        "purity": safe_float(metrics.get("overlap", {}).get("candidate_purity", partial)),
        "num_candidates": safe_float(metrics.get("efficiency", {}).get("num_candidates")),
        "runtime": safe_float(metrics.get("efficiency", {}).get("runtime_total")),
    }


def load_runtime(run_dir: Path) -> Any:
    runtime_path = run_dir / "runtime.json"
    if not runtime_path.exists():
        return None
    runtime = load_json(runtime_path)
    value = runtime.get("duration_sec")
    if runtime.get("status") == "skipped":
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            value = metrics.get("efficiency", {}).get("runtime_total", value)
    return value


def build_variant_pool(dataset: str, run_dir: Path, variant: str, raw_pool: Sequence[Dict[str, Any]], ranked_pool: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str, str, str]:
    if variant == "Full CoMot":
        return order_records(ranked_pool, "score"), "score", rel(run_dir), "Existing full CoMot output order."
    if variant == "w/o Structural Evidence":
        return order_records(raw_pool, "nonstructural"), "nonstructural", rel(run_dir), "Same assembled pool; removes structural evidence from prioritization."
    if variant == "w/o Compatibility Filtering":
        if dataset == "amlworld":
            records, source = aml_scored_pairs_pool(run_dir)
            return order_records(records, "score"), "score", rel(source), "Uses saved pre-filter alignment pairs directly."
        records, source = standard_anchor_edge_pool(dataset, cap_per_anchor=192, candidate_type="raw_anchor_edge")
        return records, "file_order", rel(source), "Standard public runner has no saved pre-filter pairs; uses bounded raw anchor-neighborhood edge pool."
    if variant == "w/o Template Constraints":
        if dataset == "amlworld":
            records, source = aml_compat_edges_pool(run_dir)
            return order_records(records, "score"), "score", rel(source), "Uses retained compatibility edges directly, bypassing template constraints."
        records, source = standard_anchor_edge_pool(dataset, cap_per_anchor=48, candidate_type="compatibility_edge")
        return records, "file_order", rel(source), "Uses anchor-incident compatibility edges directly, bypassing motif-template assembly."
    if variant == "w/o Candidate Prioritization":
        return order_records(raw_pool, "unprioritized"), "unprioritized", rel(run_dir), "Same assembled pool with deterministic non-score ordering."
    raise ValueError(f"Unknown variant: {variant}")


def mean_std(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[Any, Any]:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "N/A")]
    if not values:
        return None, None
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            subset = [row for row in rows if row["dataset"] == dataset and row["ablation_variant"] == variant and row["status"] == "completed"]
            all_subset = [row for row in rows if row["dataset"] == dataset and row["ablation_variant"] == variant]
            if not subset:
                output.append({
                    "dataset": dataset,
                    "method": METHOD,
                    "ablation_variant": variant,
                    "num_seeds": len(all_subset),
                    "completed_seeds": 0,
                    "status": "needs_run",
                    "notes": "; ".join(sorted({str(row.get("notes", "")) for row in all_subset})),
                })
                continue
            row = {
                "dataset": dataset,
                "method": METHOD,
                "ablation_variant": variant,
                "num_seeds": len(all_subset),
                "completed_seeds": len(subset),
                "status": "completed" if len(subset) == len(all_subset) else "needs_run",
                "notes": subset[0].get("notes", ""),
            }
            for key in ["motif_f1", "edge_f1", "node_f1", "motif_jaccard", "num_candidates", "purity", "runtime"]:
                avg, _ = mean_std(subset, key)
                row[key] = avg
            output.append(row)
    return output


def write_csv(rows: Sequence[Dict[str, Any]], path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in columns})


def latex_escape(value: Any) -> str:
    text = fmt(value)
    for src, dst in {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}.items():
        text = text.replace(src, dst)
    return text


def write_latex(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    columns = [
        ("dataset", "Dataset"),
        ("ablation_variant", "Variant"),
        ("motif_f1", "Motif F1"),
        ("edge_f1", "Edge F1"),
        ("node_f1", "Node F1"),
        ("motif_jaccard", "Jaccard"),
        ("num_candidates", "\\#Cand."),
        ("purity", "Purity"),
        ("runtime", "Runtime"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrrr}",
        r"\hline",
        " & ".join(label for _, label in columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(key)) for key, _ in columns) + r" \\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{CoMot ablation study across four public datasets. Recovery metrics and purity are evaluated on the top returned candidates; candidate count is the assembled pool size.}",
        r"\label{tab:ablation}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build four-dataset CoMot ablation tables from existing artifacts.")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--summary_dir", default="outputs/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = PROJECT_ROOT / args.output_root
    summary_dir = PROJECT_ROOT / args.summary_dir
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        gt_rows = load_ground_truth(dataset)
        for seed in SEEDS:
            run_dir = output_root / dataset / METHOD / f"seed_{seed}"
            try:
                raw_pool, ranked_pool, _ = load_full_pool(dataset, run_dir)
                for variant in VARIANTS:
                    start = time.perf_counter()
                    pool, _, source, notes = build_variant_pool(dataset, run_dir, variant, raw_pool, ranked_pool)
                    if variant == "Full CoMot":
                        metrics = main_metrics(run_dir)
                        runtime = metrics["runtime"]
                        num_candidates = metrics["num_candidates"]
                    else:
                        ordered = pool[:TOP_K_RETURNED]
                        metrics = evaluate_predictions(records_to_eval_predictions(ordered, TOP_K_RETURNED), gt_rows)
                        runtime = time.perf_counter() - start
                        num_candidates = len(pool)
                    rows.append({
                        "dataset": dataset,
                        "method": METHOD,
                        "seed": seed,
                        "ablation_variant": variant,
                        "status": "completed",
                        "motif_f1": metrics["motif_f1"],
                        "edge_f1": metrics["edge_f1"],
                        "node_f1": metrics["node_f1"],
                        "motif_jaccard": metrics["motif_jaccard"],
                        "num_candidates": num_candidates,
                        "purity": metrics["purity"],
                        "runtime": runtime,
                        "source": source,
                        "notes": notes,
                    })
            except FileNotFoundError as exc:
                for variant in VARIANTS:
                    rows.append({
                        "dataset": dataset,
                        "method": METHOD,
                        "seed": seed,
                        "ablation_variant": variant,
                        "status": "needs_run",
                        "notes": str(exc),
                    })
    summary_rows = aggregate_rows(rows)
    by_seed_path = summary_dir / "ablation_by_seed.csv"
    summary_path = summary_dir / "ablation.csv"
    latex_path = summary_dir / "ablation_latex.tex"
    write_csv(rows, by_seed_path, BY_SEED_COLUMNS)
    write_csv(summary_rows, summary_path, SUMMARY_COLUMNS)
    write_latex(summary_rows, latex_path)
    # Compatibility aliases for paper mapping.
    write_csv(summary_rows, summary_dir / "ablation.csv", SUMMARY_COLUMNS)
    write_latex(summary_rows, summary_dir / "ablation_latex.tex")
    print(json.dumps({
        "summary": rel(summary_path),
        "by_seed": rel(by_seed_path),
        "latex": rel(latex_path),
        "rows": len(summary_rows),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
