import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.run_method import ALL_DATASETS, ALL_METHODS


DEFAULT_MAIN_METRICS = ["motif_f1", "edge_f1", "node_f1", "motif_jaccard", "runtime", "num_candidates"]
EFFICIENCY_METRICS = ["runtime", "memory_usage", "num_candidates"]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_list(value: Optional[str], allowed: Optional[Sequence[str]] = None) -> Optional[List[str]]:
    if value in (None, "", "auto"):
        return None
    if value == "all":
        return list(allowed) if allowed else None
    items = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    if allowed:
        unknown = [item for item in items if item not in allowed]
        if unknown:
            raise ValueError(f"Unknown values {unknown}; allowed={allowed}")
    return items


def parse_seeds(value: Optional[str]) -> Optional[List[int]]:
    if value in (None, "", "auto"):
        return None
    return [int(item) for item in value.replace(",", " ").split() if item.strip()]


def flatten(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        next_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, next_key))
        else:
            flat[next_key] = value
    return flat


def safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "N/A"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def metric_value(flat: Dict[str, Any], metric: str) -> Optional[float]:
    aliases = {
        "node_precision": "node_level.precision",
        "node_recall": "node_level.recall",
        "node_f1": "node_level.f1",
        "node_mrr": "node_level.mrr",
        "edge_precision": "edge_level.precision",
        "edge_recall": "edge_level.recall",
        "edge_f1": "edge_level.f1",
        "motif_exact": "motif_level.exact_match",
        "motif_partial": "motif_level.partial_match",
        "motif_jaccard": "motif_level.jaccard_similarity",
        "structure_consistency": "motif_level.structure_consistency",
        "motif_recall": "motif_level.motif_recall",
        "average_precision": "ranking.average_precision",
        "runtime": "efficiency.runtime_total",
        "memory_usage": "efficiency.memory_usage",
        "num_candidates": "efficiency.num_candidates",
    }
    if metric == "motif_f1":
        precision = safe_float(flat.get("motif_level.partial_match"))
        recall = safe_float(flat.get("motif_level.motif_recall"))
        if precision is None or recall is None:
            return None
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    if metric.startswith("hit@"):
        k = metric.split("@", 1)[1]
        return safe_float(flat.get(f"edge_level.hit_at_k.{k}"))
    if metric.startswith("edge_hit@"):
        k = metric.split("@", 1)[1]
        return safe_float(flat.get(f"edge_level.hit_at_k.{k}"))
    if metric.startswith("node_hit@"):
        k = metric.split("@", 1)[1]
        return safe_float(flat.get(f"node_level.hit_at_k.{k}"))
    return safe_float(flat.get(aliases.get(metric, metric)))


def discover_runs(output_root: Path) -> Dict[Tuple[str, str, int], Path]:
    runs: Dict[Tuple[str, str, int], Path] = {}
    for metrics_path in sorted(output_root.rglob("metrics.*")):
        if metrics_path.name not in ("metrics.json", "metrics.csv"):
            continue
        seed_dir = metrics_path.parent
        if not seed_dir.name.startswith("seed_"):
            continue
        try:
            seed = int(seed_dir.name.replace("seed_", "", 1))
        except ValueError:
            continue
        method_dir = seed_dir.parent
        dataset_dir = method_dir.parent
        key = (dataset_dir.name, method_dir.name, seed)
        if key not in runs or metrics_path.suffix == ".json":
            runs[key] = metrics_path
    return runs


def build_run_keys(
    discovered: Dict[Tuple[str, str, int], Path],
    datasets: Optional[List[str]],
    methods: Optional[List[str]],
    seeds: Optional[List[int]],
) -> List[Tuple[str, str, int]]:
    if datasets is None:
        datasets = sorted({key[0] for key in discovered}) or ALL_DATASETS
    if methods is None:
        methods = sorted({key[1] for key in discovered}) or ALL_METHODS
    if seeds is None:
        seeds = sorted({key[2] for key in discovered}) or [0]
    return [(dataset, method, seed) for dataset in datasets for method in methods for seed in seeds]


def load_run_row(key: Tuple[str, str, int], metrics_path: Optional[Path], selected_metrics: Sequence[str]) -> Dict[str, Any]:
    dataset, method, seed = key
    row: Dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "status": "missing",
        "metrics_path": str(metrics_path) if metrics_path else "",
    }
    if metrics_path and metrics_path.exists():
        flat = load_metrics_flat(metrics_path)
        row["status"] = "ok"
        for metric in selected_metrics:
            row[metric] = metric_value(flat, metric)
    else:
        for metric in selected_metrics:
            row[metric] = None
    return row


def load_metrics_flat(path: Path) -> Dict[str, Any]:
    if path.suffix == ".json":
        return flatten(json.loads(path.read_text(encoding="utf-8")))
    flat: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "metric" in row and "value" in row:
                flat[row["metric"]] = row["value"]
    return flat


def mean_std(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    clean = [value for value in values if value is not None]
    if not clean:
        return None, None, 0
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0, 1
    var = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
    return mean, math.sqrt(var), len(clean)


def format_mean_std(mean: Optional[float], std: Optional[float], precision: int = 4) -> str:
    if mean is None:
        return "N/A"
    return f"{mean:.{precision}f} ± {std or 0.0:.{precision}f}"


def aggregate_rows(rows: Sequence[Dict[str, Any]], metrics: Sequence[str], precision: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
    output = []
    for (dataset, method), group in sorted(grouped.items()):
        agg = {"dataset": dataset, "method": method, "num_seeds": len({row["seed"] for row in group}), "num_available": sum(1 for row in group if row["status"] == "ok")}
        for metric in metrics:
            mean, std, _ = mean_std(row.get(metric) for row in group)
            agg[f"{metric}_mean"] = mean
            agg[f"{metric}_std"] = std
            agg[metric] = format_mean_std(mean, std, precision=precision)
        output.append(agg)
    return output


def write_csv(rows: Sequence[Dict[str, Any]], path: Path, fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("N/A" if row.get(key) is None else row.get(key, "N/A")) for key in fieldnames})


def latex_escape(value: Any) -> str:
    text = str(value)
    for src, dst in {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }.items():
        text = text.replace(src, dst)
    return text


def write_latex_table(rows: Sequence[Dict[str, Any]], path: Path, columns: Sequence[str], caption: str, label: str, bold_comot: bool) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\hline",
        " & ".join(latex_escape(col) for col in columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = latex_escape(row.get(col, "N/A") if row.get(col, "") not in (None, "") else "N/A")
            if bold_comot and row.get("method") == "comot":
                value = rf"\textbf{{{value}}}"
            values.append(value)
        lines.append(" & ".join(values) + r" \\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def dataset_statistics(processed_root: Path) -> List[Dict[str, Any]]:
    if not processed_root.exists():
        return []
    rows = []
    for dataset_dir in sorted(path for path in processed_root.iterdir() if path.is_dir()):
        metadata_path = dataset_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        row = {
            "dataset": dataset_dir.name,
            "num_nodes": metadata.get("num_nodes", count_csv_rows(dataset_dir / "nodes.csv")),
            "num_edges": metadata.get("num_edges", count_csv_rows(dataset_dir / "edges.csv")),
            "num_motif_instances": metadata.get("num_motif_instances", count_json_items(dataset_dir / "ground_truth.json")),
            "directed": metadata.get("directed", ""),
            "weighted": metadata.get("weighted", ""),
            "has_timestamps": metadata.get("has_timestamps", ""),
            "has_ground_truth": metadata.get("has_ground_truth", ""),
            "source": metadata.get("source", ""),
        }
        rows.append(row)
    return rows


def count_csv_rows(path: Path) -> Any:
    if not path.exists():
        return "N/A"
    with open(path, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def count_json_items(path: Path) -> Any:
    if not path.exists():
        return "N/A"
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data) if isinstance(data, list) else "N/A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CoMot experiment outputs into summary tables.")
    parser.add_argument("--output_root", default="outputs", help="Experiment output root.")
    parser.add_argument("--summary_dir", default="outputs/summary", help="Directory for aggregated tables.")
    parser.add_argument("--processed_root", default="data/processed")
    parser.add_argument("--datasets", default="auto", help="Comma/space-separated datasets, all, or auto.")
    parser.add_argument("--methods", default="auto", help="Comma/space-separated methods, all, or auto.")
    parser.add_argument("--seeds", default="auto", help="Comma/space-separated seeds or auto.")
    parser.add_argument("--main_metric", default="motif_f1", help="Primary metric, e.g. motif_f1, edge_f1, hit@100.")
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--bold_comot", action="store_true", default=True)
    parser.add_argument("--no_bold_comot", action="store_false", dest="bold_comot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    summary_dir = resolve_path(args.summary_dir)
    selected_metrics = unique([args.main_metric] + DEFAULT_MAIN_METRICS + EFFICIENCY_METRICS)
    discovered = discover_runs(output_root)
    keys = build_run_keys(
        discovered,
        datasets=parse_list(args.datasets, ALL_DATASETS),
        methods=parse_list(args.methods, ALL_METHODS),
        seeds=parse_seeds(args.seeds),
    )
    rows = [load_run_row(key, discovered.get(key), selected_metrics) for key in keys]
    aggregate = aggregate_rows(rows, selected_metrics, precision=args.precision)

    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, summary_dir / "all_results.csv")

    main_columns = ["dataset", "method", "num_available"] + DEFAULT_MAIN_METRICS
    main_rows = [{col: row.get(col, "N/A") for col in main_columns} for row in aggregate]
    write_csv(main_rows, summary_dir / "main_table.csv", fieldnames=main_columns)
    write_latex_table(main_rows, summary_dir / "main_table_latex.tex", main_columns, "Main experimental results.", "tab:main_results", args.bold_comot)

    eff_columns = ["dataset", "method", "num_available"] + EFFICIENCY_METRICS
    eff_rows = [{col: row.get(col, "N/A") for col in eff_columns} for row in aggregate]
    write_csv(eff_rows, summary_dir / "efficiency_table.csv", fieldnames=eff_columns)
    write_latex_table(eff_rows, summary_dir / "efficiency_table_latex.tex", eff_columns, "Efficiency comparison.", "tab:efficiency", args.bold_comot)

    stats = dataset_statistics(resolve_path(args.processed_root))
    stats_columns = ["dataset", "num_nodes", "num_edges", "num_motif_instances", "directed", "weighted", "has_timestamps", "has_ground_truth", "source"]
    write_csv(stats, summary_dir / "dataset_statistics.csv", fieldnames=stats_columns)
    write_latex_table(stats, summary_dir / "dataset_statistics_latex.tex", stats_columns, "Dataset statistics.", "tab:dataset_statistics", bold_comot=False)

    print(json.dumps({
        "all_results": str(summary_dir / "all_results.csv"),
        "main_table": str(summary_dir / "main_table.csv"),
        "main_table_latex": str(summary_dir / "main_table_latex.tex"),
        "efficiency_table": str(summary_dir / "efficiency_table.csv"),
        "efficiency_table_latex": str(summary_dir / "efficiency_table_latex.tex"),
        "dataset_statistics": str(summary_dir / "dataset_statistics.csv"),
        "dataset_statistics_latex": str(summary_dir / "dataset_statistics_latex.tex"),
        "num_runs": len(rows),
        "num_discovered": len(discovered),
    }, indent=2), flush=True)


def unique(items: Sequence[str]) -> List[str]:
    output = []
    for item in items:
        if item not in output:
            output.append(item)
    return output


if __name__ == "__main__":
    main()
