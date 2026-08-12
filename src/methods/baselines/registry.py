import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Type


SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from methods.base_method import BaseMethod
from methods.baselines.aa import AdamicAdarBaseline
from methods.baselines.bright import BrightBaseline
from methods.baselines.cn import CommonNeighborsBaseline
from methods.baselines.common import StandardGraphDataset, load_json_or_yaml
from methods.baselines.csgm import CSGMBaseline
from methods.baselines.deepwalk import DeepWalkBaseline
from methods.baselines.fanmod import FanmodBaseline
from methods.baselines.final import FinalBaseline
from methods.baselines.graphsage import GraphSAGEBaseline
from methods.baselines.gspan import GSpanBaseline
from methods.baselines.hlot import HLOTBaseline
from methods.baselines.modern_subgraph import ISONETBaseline, NeuGNBaseline, TPABBaseline
from methods.baselines.node2vec import Node2VecBaseline
from methods.baselines.regal import RegalBaseline
from methods.baselines.seal import SealBaseline
from methods.baselines.subgnn import SubGNNBaseline
from methods.baselines.turboiso import TurboISOBaseline
from methods.baselines.vf2 import VF2Baseline


BASELINES: Dict[str, Type[BaseMethod]] = {
    "aa": AdamicAdarBaseline,
    "bright": BrightBaseline,
    "cn": CommonNeighborsBaseline,
    "CSGM": CSGMBaseline,
    "deepwalk": DeepWalkBaseline,
    "fanmod": FanmodBaseline,
    "final": FinalBaseline,
    "graphsage": GraphSAGEBaseline,
    "gspan": GSpanBaseline,
    "HLOT": HLOTBaseline,
    "ISONET": ISONETBaseline,
    "NeuGN": NeuGNBaseline,
    "node2vec": Node2VecBaseline,
    "regal": RegalBaseline,
    "seal": SealBaseline,
    "subgnn": SubGNNBaseline,
    "TPAB": TPABBaseline,
    "turboiso": TurboISOBaseline,
    "vf2": VF2Baseline,
}


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(config_path: Path, method: str) -> Dict[str, Any]:
    config = load_json_or_yaml(config_path)
    baseline_path = PROJECT_ROOT / "configs" / "baselines" / f"{method}.yaml"
    if baseline_path.exists():
        config = deep_merge(config, load_json_or_yaml(baseline_path))
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CoMot baseline on a standard processed dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset name, for example amlworld.")
    parser.add_argument("--method", required=True, choices=sorted(BASELINES.keys()))
    parser.add_argument("--config", required=True, help="Dataset or experiment config path.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to outputs/baselines/{dataset}/{method}.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return resolve_path(args.output_dir)
    return PROJECT_ROOT / "outputs" / "baselines" / args.dataset / args.method


def main() -> None:
    args = parse_args()
    config = load_experiment_config(resolve_path(args.config), args.method)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = StandardGraphDataset(dataset_name=args.dataset, project_root=PROJECT_ROOT, config=config)
    method = BASELINES[args.method](config=config, project_root=PROJECT_ROOT, output_dir=output_dir, seed=seed)

    print(f"[run_baseline] dataset={args.dataset} method={args.method} seed={seed}", flush=True)
    print(f"[run_baseline] dataset_dir={dataset.dataset_dir}", flush=True)
    print(f"[run_baseline] method_query_view={dataset.method_queries_path}", flush=True)
    print(f"[run_baseline] output_dir={output_dir}", flush=True)

    method.fit(dataset)
    predictions = method.predict(dataset)
    outputs = method.evaluate(dataset)

    print(f"[run_baseline] predictions={len(predictions)}", flush=True)
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
