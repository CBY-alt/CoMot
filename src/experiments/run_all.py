import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.run_method import ALL_DATASETS, ALL_METHODS


def parse_list(value: str, allowed: List[str]) -> List[str]:
    if value == "all":
        return list(allowed)
    items = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown values {unknown}; allowed={allowed}")
    return items


def run_command(args: argparse.Namespace, dataset: str, method: str, seed: int) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "src.experiments.run_method",
        "--dataset",
        dataset,
        "--method",
        method,
        "--seed",
        str(seed),
        "--output_root",
        args.output_root,
    ]
    if args.debug:
        cmd.append("--debug")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.resume:
        cmd.append("--resume")

    start = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "returncode": proc.returncode,
        "status": "success" if proc.returncode == 0 else "failed",
        "duration_sec": time.time() - start,
        "command": " ".join(cmd),
    }


def write_summary(rows: List[dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "run_all_summary.json"
    csv_path = output_root / "run_all_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "seed", "status", "returncode", "duration_sec", "command"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoMot experiments over dataset/method/seed grids.")
    parser.add_argument("--datasets", default="all", help="Comma/space-separated dataset names or all.")
    parser.add_argument("--methods", default="all", help="Comma/space-separated method names or all.")
    parser.add_argument("--seeds", default="0", help="Comma/space-separated integer seeds.")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = parse_list(args.datasets, ALL_DATASETS)
    methods = parse_list(args.methods, ALL_METHODS)
    seeds = [int(item) for item in args.seeds.replace(",", " ").split() if item.strip()]

    rows = []
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                print(f"[run_all] dataset={dataset} method={method} seed={seed}", flush=True)
                rows.append(run_command(args, dataset, method, seed))
    write_summary(rows, PROJECT_ROOT / args.output_root)
    failures = sum(1 for row in rows if row["status"] != "success")
    print(f"[run_all] completed total={len(rows)} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
