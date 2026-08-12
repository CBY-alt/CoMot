import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.amlworld import AMLWorldDataset
from methods.comot import CoMotMethod


DATASETS = {
    "amlworld": AMLWorldDataset,
}

METHODS = {
    "comot": CoMotMethod,
}


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(config)
    dataset_cfg = dict(config.get("dataset", {}))
    partition_cfg = dict(config.get("partition", {}))

    if args.dataset:
        dataset_cfg["name"] = args.dataset
    if args.seed is not None:
        partition_cfg["seed"] = args.seed
    if args.trans_path:
        dataset_cfg["transaction_path"] = args.trans_path
    if args.pattern_path:
        dataset_cfg["pattern_path"] = args.pattern_path

    config["dataset"] = dataset_cfg
    config["partition"] = partition_cfg
    return config


def _env_int(name: str):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else None


def _env_float(name: str):
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else None


def _env_bool(name: str):
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return value.lower() in ("1", "true", "yes", "y", "on")


def apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(config)
    partition_cfg = dict(config.get("partition", {}))
    local_cfg = dict(config.get("local_encoder", {}))
    orchestrator_cfg = dict(config.get("orchestrator", {}))
    candidate_cfg = dict(config.get("candidate_builder", {}))

    env_partition = {
        "NUM_BANKS": ("num_banks", _env_int),
        "MIN_BANKS_PER_MOTIF": ("min_banks_per_motif", _env_int),
        "CLEAN_INTER_RATIO": ("clean_inter_ratio", _env_float),
        "TIME_MODE": ("time_mode", lambda name: os.environ.get(name)),
        "KEEP_UNKNOWN_LAUNDERING": ("keep_unknown_laundering", _env_bool),
        "MAX_CLEAN_NEIGHBORS_TO_REASSIGN": ("max_clean_neighbors_to_reassign", _env_int),
    }
    for env_name, (cfg_name, parser) in env_partition.items():
        value = parser(env_name)
        if value is not None:
            partition_cfg[cfg_name] = value

    env_local = {
        "K_HOP": ("k_hop", _env_int),
        "NUM_WORKERS": ("num_workers", _env_int),
    }
    for env_name, (cfg_name, parser) in env_local.items():
        value = parser(env_name)
        if value is not None:
            local_cfg[cfg_name] = value

    env_orchestrator = {
        "ALIGN_THRESHOLD": ("threshold", _env_float),
        "MAX_GROUP_SIZE": ("max_group_size", _env_int),
        "TOPK_PER_SENDER": ("topk_per_sender", _env_int),
        "TOPK_PER_RECEIVER": ("topk_per_receiver", _env_int),
    }
    for env_name, (cfg_name, parser) in env_orchestrator.items():
        value = parser(env_name)
        if value is not None:
            orchestrator_cfg[cfg_name] = value

    env_candidate = {
        "CHAIN_MIN_LEN": ("chain_min_len", _env_int),
        "CHAIN_MAX_LEN": ("chain_max_len", _env_int),
        "CYCLE_MIN_LEN": ("cycle_min_len", _env_int),
        "CYCLE_MAX_LEN": ("cycle_max_len", _env_int),
        "FAN_DEGREE_THR": ("fan_degree_thr", _env_int),
    }
    for env_name, (cfg_name, parser) in env_candidate.items():
        value = parser(env_name)
        if value is not None:
            candidate_cfg[cfg_name] = value

    config["partition"] = partition_cfg
    config["local_encoder"] = local_cfg
    config["orchestrator"] = orchestrator_cfg
    config["candidate_builder"] = candidate_cfg
    return config


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        return output_dir

    run_name = os.environ.get("RUN_NAME", "amlworld_hi_small")
    output_root = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "outputs")).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return output_root / run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified CoMot experiment pipeline entrypoint.")
    parser.add_argument("--config", required=True, help="Path to JSON experiment config.")
    parser.add_argument("--dataset", default="amlworld", choices=sorted(DATASETS.keys()))
    parser.add_argument("--method", default="comot", choices=sorted(METHODS.keys()))
    parser.add_argument("--output_dir", default=None, help="Run output directory.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trans_path", default=None, help="Optional AMLWorld transaction CSV override.")
    parser.add_argument("--pattern_path", default=None, help="Optional AMLWorld pattern TXT override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_environment_overrides(apply_cli_overrides(load_config(resolve_config_path(args.config)), args))
    seed = int(args.seed if args.seed is not None else config.get("partition", {}).get("seed", 42))
    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cls = DATASETS[args.dataset]
    method_cls = METHODS[args.method]

    dataset = dataset_cls(config=config, project_root=PROJECT_ROOT, output_dir=output_dir, seed=seed)
    method = method_cls(config=config, project_root=PROJECT_ROOT, output_dir=output_dir, seed=seed)

    print(f"[run_pipeline] dataset={args.dataset} method={args.method} seed={seed}", flush=True)
    print(f"[run_pipeline] output_dir={output_dir}", flush=True)

    print("[1/4] Prepare dataset", flush=True)
    dataset.export_standard_format()

    print("[2/4] Fit method", flush=True)
    method.fit(dataset)

    print("[3/4] Predict candidates", flush=True)
    method.predict(dataset)

    print("[4/4] Evaluate", flush=True)
    outputs = method.evaluate(dataset)

    print("[run_pipeline] completed", flush=True)
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
