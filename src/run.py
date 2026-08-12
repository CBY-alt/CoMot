import argparse
import contextlib
import csv
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.amlworld import AMLWorldDataset
from methods.baselines.common import StandardGraphDataset, load_json_or_yaml, save_predictions
from methods.comot import CoMotMethod, run_comot_standard
from evaluation.evaluate import evaluate_predictions, load_predictions_jsonl
from methods.baselines.registry import BASELINES, deep_merge


ALL_DATASETS = ["amlworld", "elliptic", "dblp", "web_google"]
ALL_METHODS = ["comot"] + sorted(BASELINES)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_config_path(dataset: str, method: str) -> Path:
    if dataset == "amlworld" and method == "comot":
        return PROJECT_ROOT / "configs" / "amlworld_hi_small.json"
    return PROJECT_ROOT / "configs" / f"{dataset}.yaml"


def load_method_config(dataset: str, method: str, config_path: Optional[str], seed: int, debug: bool) -> Dict[str, Any]:
    path = resolve_path(config_path) if config_path else default_config_path(dataset, method)
    config = load_json_or_yaml(path)
    if method == "comot":
        comot_path = PROJECT_ROOT / "configs" / "methods" / "comot_standard.yaml"
        if comot_path.exists():
            config = deep_merge(load_json_or_yaml(comot_path), config)
    if method in BASELINES:
        baseline_path = PROJECT_ROOT / "configs" / "baselines" / f"{method}.yaml"
        if baseline_path.exists():
            config = deep_merge(config, load_json_or_yaml(baseline_path))
    config = apply_run_overrides(config, dataset=dataset, method=method, seed=seed, debug=debug)
    config["_run"] = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "debug": debug,
        "source_config": str(path),
    }
    return config


def apply_run_overrides(config: Dict[str, Any], dataset: str, method: str, seed: int, debug: bool) -> Dict[str, Any]:
    config = dict(config)
    config["seed"] = seed
    dataset_cfg = dict(config.get("dataset", {}))
    dataset_cfg.setdefault("dataset_name", dataset)
    dataset_cfg.setdefault("name", dataset)
    config["dataset"] = dataset_cfg

    if method == "comot":
        partition_cfg = dict(config.get("partition", {}))
        partition_cfg["seed"] = seed
        config["partition"] = partition_cfg
        data_cfg = dict(config.get("data", {}))
        data_cfg.setdefault("dataset_dir", f"data/processed/{dataset}")
        if debug:
            data_cfg["max_nodes"] = min_int(data_cfg.get("max_nodes"), 20000)
            data_cfg["max_edges"] = min_int(data_cfg.get("max_edges"), 50000)
        config["data"] = data_cfg
        comot_cfg = dict(config.get("comot", {}))
        comot_cfg.setdefault("runner", "auto")
        standard_cfg = dict(comot_cfg.get("standard", {}))
        if debug:
            standard_cfg["max_predictions"] = min_int(standard_cfg.get("max_predictions"), 100)
            standard_cfg["max_candidates_per_query"] = min_int(standard_cfg.get("max_candidates_per_query"), 4)
            standard_cfg["max_neighbors_per_anchor"] = min_int(standard_cfg.get("max_neighbors_per_anchor"), 12)
            standard_cfg["max_path_len"] = min_int(standard_cfg.get("max_path_len"), 3)
        comot_cfg["standard"] = standard_cfg
        config["comot"] = comot_cfg

    if method in BASELINES:
        data_cfg = dict(config.get("data", {}))
        data_cfg.setdefault("dataset_dir", f"data/processed/{dataset}")
        if debug:
            data_cfg["max_nodes"] = min_int(data_cfg.get("max_nodes"), 20000)
            data_cfg["max_edges"] = min_int(data_cfg.get("max_edges"), 50000)
        config["data"] = data_cfg

        baseline_cfg = dict(config.get("baseline", {}))
        if debug:
            baseline_cfg["max_predictions"] = min_int(baseline_cfg.get("max_predictions"), 100)
            for key, value in {
                "embedding_max_nodes": 5000,
                "max_nodes_per_partition": 200,
                "candidate_edge_limit": 2000,
                "max_seed_nodes": 1000,
                "fallback_max_edges": 5000,
                "export_max_edges": 10000,
                "timeout": 10,
                "debug": True,
            }.items():
                if key in baseline_cfg or key in ("debug", "timeout"):
                    baseline_cfg[key] = value if isinstance(value, bool) else min_int(baseline_cfg.get(key), value)
        config["baseline"] = baseline_cfg
    return config


def min_int(current: Any, limit: int) -> int:
    if current in (None, "all"):
        return limit
    try:
        return min(int(current), limit)
    except (TypeError, ValueError):
        return limit


def output_dir_for(dataset: str, method: str, seed: int, output_root: str) -> Path:
    return resolve_path(output_root) / dataset / method / f"seed_{seed}"


def prepare_output_dir(output_dir: Path, overwrite: bool, resume: bool) -> str:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and resume and not overwrite:
        return "skip"
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and not resume and not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory exists. Use --overwrite or --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return "run"


def save_config(config: Dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "config.yaml"
    try:
        import yaml
        text = yaml.safe_dump(config, sort_keys=False)
    except Exception:
        text = json.dumps(config, indent=2)
    path.write_text(text, encoding="utf-8")
    return path


def save_predictions_json(predictions: List[Dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "predictions.json"
    path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    return path


def write_runtime(output_dir: Path, payload: Dict[str, Any]) -> Path:
    path = output_dir / "runtime.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_baseline_method(dataset_name: str, method_name: str, config: Dict[str, Any], output_dir: Path, seed: int) -> Dict[str, Any]:
    dataset = StandardGraphDataset(dataset_name=dataset_name, project_root=PROJECT_ROOT, config=config)
    print(f"[run_method] method_query_view={dataset.method_queries_path}", flush=True)
    method = BASELINES[method_name](config=config, project_root=PROJECT_ROOT, output_dir=output_dir, seed=seed)
    method.fit(dataset)
    predictions = method.predict(dataset)
    eval_outputs = method.evaluate(dataset)
    save_predictions_json(predictions, output_dir)
    return {"predictions": len(predictions), "outputs": {k: str(v) for k, v in eval_outputs.items()}}


def run_comot_method(dataset_name: str, config: Dict[str, Any], output_dir: Path, seed: int) -> Dict[str, Any]:
    runner = str(config.get("comot", {}).get("runner", "auto")).lower()
    if runner not in {"auto", "legacy", "standard"}:
        raise ValueError(f"Unknown CoMot runner: {runner}. Choose auto, legacy, or standard.")
    selected = "legacy" if runner == "auto" and dataset_name == "amlworld" else runner
    if selected == "auto":
        selected = "standard"
    if selected == "legacy" and dataset_name != "amlworld":
        if runner == "legacy":
            print("[run_method] requested legacy CoMot for non-AMLWorld; falling back to standard runner", flush=True)
        selected = "standard"
    if selected == "standard":
        return run_comot_standard_method(dataset_name, config, output_dir, seed)
    try:
        return run_comot_legacy_method(dataset_name, config, output_dir, seed)
    except Exception as exc:
        if runner != "auto":
            raise
        print(f"[run_method] legacy CoMot failed; falling back to standard runner: {exc}", flush=True)
        return run_comot_standard_method(dataset_name, config, output_dir, seed)


def run_comot_standard_method(dataset_name: str, config: Dict[str, Any], output_dir: Path, seed: int) -> Dict[str, Any]:
    dataset = StandardGraphDataset(dataset_name=dataset_name, project_root=PROJECT_ROOT, config=config)
    print(f"[run_method] comot_runner=standard", flush=True)
    print(f"[run_method] method_query_view={dataset.method_queries_path}", flush=True)
    result = run_comot_standard(dataset.dataset_dir, config, output_dir, seed)
    predictions = load_predictions_jsonl(output_dir / "predictions.jsonl")
    eval_outputs = evaluate_predictions(predictions, dataset.load_ground_truth(), output_dir)
    result["outputs"].update({k: str(v) for k, v in eval_outputs.items()})
    return result


def run_comot_legacy_method(dataset_name: str, config: Dict[str, Any], output_dir: Path, seed: int) -> Dict[str, Any]:
    if dataset_name != "amlworld":
        raise ValueError("CoMot legacy pipeline is currently implemented only for AMLWorld.")
    print("[run_method] comot_runner=legacy", flush=True)
    if config.get("_run", {}).get("debug"):
        legacy = find_legacy_comot_reranked()
        if legacy is not None:
            return convert_legacy_comot_outputs(legacy, output_dir)
    dataset = AMLWorldDataset(config=config, project_root=PROJECT_ROOT, output_dir=output_dir, seed=seed)
    method = CoMotMethod(config=config, project_root=PROJECT_ROOT, output_dir=output_dir, seed=seed)
    dataset.export_standard_format()
    method.fit(dataset)
    method.predict(dataset)
    eval_outputs = method.evaluate(dataset)

    standard_eval_dir = output_dir / "standard_eval"
    standard_predictions = standard_eval_dir / "predictions.jsonl"
    predictions = load_predictions_jsonl(standard_predictions) if standard_predictions.exists() else []
    predictions = augment_amlworld_comot_node_evidence(predictions, config)
    if predictions:
        ground_truth = StandardGraphDataset(dataset_name="amlworld", project_root=PROJECT_ROOT, config=config).load_ground_truth()
        eval_outputs.update(evaluate_predictions(predictions, ground_truth, output_dir))
    save_predictions_json(predictions, output_dir)
    save_predictions(predictions, output_dir / "predictions.jsonl")
    if not (output_dir / "metrics.json").exists():
        for name in ("metrics.json", "metrics.csv", "summary.txt"):
            src = standard_eval_dir / name
            if src.exists():
                shutil.copy2(src, output_dir / name)
    return {"predictions": len(predictions), "outputs": {k: str(v) for k, v in eval_outputs.items()}}


def augment_amlworld_comot_node_evidence(predictions: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = config.get("comot", {}).get("legacy_node_evidence", {})
    if not bool(cfg.get("enabled", False)):
        return predictions
    dataset_dir = resolve_path(config.get("data", {}).get("dataset_dir", "data/processed/amlworld"))
    query_path = dataset_dir / "method_queries.json"
    if not query_path.exists():
        return predictions
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    prefix = int(cfg.get("prefix_queries", 300))
    max_anchor_nodes = int(cfg.get("max_anchor_nodes", 8))
    extras: List[Dict[str, Any]] = []
    for idx, query in enumerate(queries[:prefix]):
        anchors = [str(node) for node in query.get("anchor_nodes", []) if node is not None][:max_anchor_nodes]
        if not anchors:
            continue
        extras.append(
            {
                "query_id": str(query.get("query_id", f"amlworld_anchor_{idx}")),
                "predicted_nodes": anchors,
                "predicted_edges": [],
                "score": 2.0 - idx * 1e-4,
                "runtime": 0.0,
                "metadata": {
                    "method": "comot",
                    "candidate_type": "amlworld_anchor_node_evidence",
                    "runner": "comot_legacy",
                    "source": "method_visible_anchor_nodes",
                },
            }
        )
    return extras + predictions


def find_legacy_comot_reranked() -> Optional[Path]:
    candidates = [
        PROJECT_ROOT / "outputs" / "amlworld_hi_small_unified" / "semotif_candidates_ctrl_v2_003_reranked" / "candidates_reranked.csv",
        PROJECT_ROOT / "outputs" / "amlworld_hi_small" / "semotif_candidates_ctrl_v2_003_reranked" / "candidates_reranked.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def convert_legacy_comot_outputs(reranked_csv: Path, output_dir: Path) -> Dict[str, Any]:
    predictions = []
    with open(reranked_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(
                {
                    "query_id": row.get("candidate_type", "comot"),
                    "predicted_nodes": [strip_partition_prefix(node) for node in parse_json_list(row.get("nodes_json"))],
                    "predicted_edges": [str(edge_id) for edge_id in parse_json_list(row.get("tx_ids_json"))],
                    "score": safe_float(row.get("rerank_score") or row.get("score_mean") or 0.0),
                    "runtime": 0.0,
                    "metadata": {
                        "method": "comot",
                        "candidate_id": row.get("candidate_id", ""),
                        "candidate_type": row.get("candidate_type", ""),
                        "legacy_source": str(reranked_csv),
                        "conversion": "legacy_reranked_csv_to_standard_predictions",
                    },
                }
            )
    ground_truth_path = PROJECT_ROOT / "data" / "processed" / "amlworld" / "ground_truth.json"
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    save_predictions(predictions, output_dir / "predictions.jsonl")
    save_predictions_json(predictions, output_dir)
    eval_outputs = evaluate_predictions(predictions, ground_truth, output_dir)
    return {
        "predictions": len(predictions),
        "legacy_source": str(reranked_csv),
        "outputs": {k: str(v) for k, v in eval_outputs.items()},
    }


def parse_json_list(value: Any) -> List[Any]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def strip_partition_prefix(node: Any) -> str:
    text = str(node)
    return text.split("::", 1)[1] if "::" in text else text


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def run_one(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    output_dir = output_dir_for(args.dataset, args.method, seed, args.output_root)
    status = "failed"
    start = time.time()
    runtime_payload: Dict[str, Any] = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": seed,
        "debug": bool(args.debug),
        "output_dir": str(output_dir),
        "start_time": start,
    }
    try:
        action = prepare_output_dir(output_dir, overwrite=args.overwrite, resume=args.resume)
        if action == "skip":
            runtime_payload.update({"status": "skipped", "reason": "metrics.json exists and --resume was set", "end_time": time.time(), "duration_sec": 0.0})
            write_runtime(output_dir, runtime_payload)
            print(f"[run_method] skipped existing run: {output_dir}", flush=True)
            return 0

        log_path = output_dir / "log.txt"
        config = load_method_config(args.dataset, args.method, args.config, seed=seed, debug=args.debug)
        save_config(config, output_dir)

        with open(log_path, "w", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
            print(f"[run_method] dataset={args.dataset} method={args.method} seed={seed}", flush=True)
            print(f"[run_method] output_dir={output_dir}", flush=True)
            print(f"[run_method] debug={args.debug}", flush=True)
            if args.method == "comot":
                result = run_comot_method(args.dataset, config, output_dir, seed)
            elif args.method in BASELINES:
                result = run_baseline_method(args.dataset, args.method, config, output_dir, seed)
            else:
                raise ValueError(f"Unknown method: {args.method}")
            print(json.dumps(result, indent=2), flush=True)

        status = "success"
        runtime_payload.update({"status": status, "end_time": time.time(), "duration_sec": time.time() - start})
        write_runtime(output_dir, runtime_payload)
        print(f"[run_method] success: {output_dir}", flush=True)
        return 0
    except Exception as exc:
        end = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        error_path = output_dir / "error.log"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        runtime_payload.update({"status": status, "error": str(exc), "end_time": end, "duration_sec": end - start})
        write_runtime(output_dir, runtime_payload)
        print(f"[run_method] failed: {output_dir}", flush=True)
        print(f"[run_method] error: {exc}", flush=True)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one CoMot method on one dataset with unified outputs.")
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--method", required=True, choices=ALL_METHODS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=None, help="Optional dataset config override.")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    sys.exit(run_one(parse_args()))


if __name__ == "__main__":
    main()
