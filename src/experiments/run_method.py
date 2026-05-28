import argparse
import contextlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.standard_graph import StandardGraphDataset, load_json_or_yaml
from methods.comot import run_comot_standard
from experiments.evaluate import evaluate_predictions, load_predictions_jsonl


ALL_DATASETS = ["amlworld", "elliptic", "dblp", "web_google"]
ALL_METHODS = ["comot"]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_config_path(dataset: str, method: str) -> Path:
    return PROJECT_ROOT / "configs" / "datasets.yaml"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_method_config(dataset: str, method: str, config_path: Optional[str], seed: int, debug: bool) -> Dict[str, Any]:
    path = resolve_path(config_path) if config_path else default_config_path(dataset, method)
    loaded = load_json_or_yaml(path)
    config = loaded.get(dataset, loaded) if isinstance(loaded, dict) else loaded
    if method == "comot":
        comot_path = PROJECT_ROOT / "configs" / "comot.yaml"
        if comot_path.exists():
            config = deep_merge(load_json_or_yaml(comot_path), config)
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
        comot_cfg.setdefault("runner", "standard")
        standard_cfg = dict(comot_cfg.get("standard", {}))
        if debug:
            standard_cfg["max_predictions"] = min_int(standard_cfg.get("max_predictions"), 100)
            standard_cfg["max_candidates_per_query"] = min_int(standard_cfg.get("max_candidates_per_query"), 4)
            standard_cfg["max_neighbors_per_anchor"] = min_int(standard_cfg.get("max_neighbors_per_anchor"), 12)
            standard_cfg["max_path_len"] = min_int(standard_cfg.get("max_path_len"), 3)
        comot_cfg["standard"] = standard_cfg
        config["comot"] = comot_cfg

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


def write_runtime(output_dir: Path, payload: Dict[str, Any]) -> Path:
    path = output_dir / "runtime.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_comot_method(dataset_name: str, config: Dict[str, Any], output_dir: Path, seed: int) -> Dict[str, Any]:
    runner = str(config.get("comot", {}).get("runner", "auto")).lower()
    if runner not in {"auto", "standard"}:
        raise ValueError(f"Unknown CoMot runner: {runner}. This release supports the standard processed-data runner.")
    dataset = StandardGraphDataset(dataset_name=dataset_name, project_root=PROJECT_ROOT, config=config)
    print(f"[run_method] comot_runner=standard", flush=True)
    print(f"[run_method] method_query_view={dataset.method_queries_path}", flush=True)
    result = run_comot_standard(dataset.dataset_dir, config, output_dir, seed)
    predictions = load_predictions_jsonl(output_dir / "predictions.jsonl")
    eval_outputs = evaluate_predictions(predictions, dataset.load_ground_truth(), output_dir)
    result["outputs"].update({k: str(v) for k, v in eval_outputs.items()})
    return result


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
