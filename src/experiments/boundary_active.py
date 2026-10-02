#!/usr/bin/env python3
"""Workload-facing boundary robustness protocol.

This protocol samples the 10% perturbation
from boundary units actually consumed by the fixed clean workload. Corrupted
endpoint reports receive a fresh non-GT report/edge identifier; otherwise the
the overlap evaluator would still count the corrupted report as a true edge.
No ground-truth data are used to select perturbed units.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments import boundary_common as base  # noqa: E402
from evaluation.evaluate import evaluate_predictions  # noqa: E402
from run import load_method_config  # noqa: E402
from methods.baselines.common import StandardGraphDataset, load_json_or_yaml, save_predictions  # noqa: E402
from methods.comot import (  # noqa: E402
    build_compatibility_graph_standard,
    convert_candidates_to_predictions,
    generate_candidates_standard,
    rerank_candidates_standard,
    write_standard_artifacts,
)


DATASETS = base.DATASETS
SEEDS = base.SEEDS
SETTINGS = base.SETTINGS
RATE = 0.10
PROTOCOL = "active"


def protocol_seed(dataset: str, seed: int, setting: str) -> int:
    # The experiment seed alone controls the sampled 10% subset.  This keeps
    # Missing/Spurious/Endpoint paired on the same boundary units.
    return int(seed)


def load_standard_active_edges(dataset: str, seed: int, boundary_ids: Set[str]) -> List[str]:
    path = PROJECT_ROOT / "outputs" / dataset / "comot" / f"seed_{seed}" / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    active: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            active.update(str(edge_id) for edge_id in row.get("predicted_edges", []) if str(edge_id) in boundary_ids)
    if not active:
        raise RuntimeError(f"No active boundary edges found in {path}")
    return sorted(active)


def perturb_standard_active(
    compatibility: Mapping[str, Any], dataset: str, seed: int, setting: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rng = random.Random(protocol_seed(dataset, seed, setting))
    global_boundary = set(base.boundary_edge_ids(compatibility))
    active_ids = load_standard_active_edges(dataset, seed, global_boundary)
    count = int(round(RATE * len(active_ids)))
    chosen = rng.sample(active_ids, count)
    edges = {str(edge_id): (str(pair[0]), str(pair[1])) for edge_id, pair in compatibility["edge_lookup"].items()}
    node_parts = {str(node): set(parts) for node, parts in compatibility["node_to_partitions"].items()}
    owners = base.node_owner(compatibility)

    if setting == "missing10":
        for edge_id in chosen:
            edges.pop(edge_id, None)
    elif setting == "endpoint10":
        for index, edge_id in enumerate(chosen):
            src, dst = edges.pop(edge_id)
            change_dst = bool(rng.getrandbits(1))
            old = dst if change_dst else src
            synthetic = f"__wrong_endpoint__{dataset}__s{seed}__{index}"
            node_parts[synthetic] = {owners.get(old, "UNKNOWN")}
            corrupted_id = f"__corrupt_report__{dataset}__s{seed}__{index}"
            edges[corrupted_id] = (src, synthetic) if change_dst else (synthetic, dst)
    elif setting == "spurious10":
        for index, donor in enumerate(chosen):
            src, dst = edges[donor]
            retain_src = bool(rng.getrandbits(1))
            retained, replaced = (src, dst) if retain_src else (dst, src)
            synthetic = f"__spurious_endpoint__{dataset}__s{seed}__{index}"
            node_parts[synthetic] = {owners.get(replaced, "UNKNOWN")}
            fake_id = f"__spurious_report__{dataset}__s{seed}__{index}"
            edges[fake_id] = (retained, synthetic) if retain_src else (synthetic, retained)
    else:
        raise ValueError(setting)

    perturbed = base.rebuild_compatibility(edges, node_parts, compatibility["edge_to_partitions"])
    provenance = {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "seed": seed,
        "setting": setting,
        "rate": RATE,
        "random_seed": protocol_seed(dataset, seed, setting),
        "boundary_unit": "workload-facing cross-partition edge used by the fixed clean prediction set",
        "global_boundary_units": len(global_boundary),
        "boundary_units_clean": len(active_ids),
        "boundary_units_changed": count,
        "selection_uses_ground_truth": False,
        "endpoint_corruption_uses_fresh_report_id": setting == "endpoint10",
        "source": f"outputs/{dataset}/comot/seed_{seed}/predictions.jsonl",
        "evaluator": "src/evaluation/evaluate.py:evaluate_predictions",
    }
    return perturbed, provenance


def run_standard(dataset_name: str, seed: int, setting: str, run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    config = load_method_config(dataset_name, "comot", config_path=None, seed=seed, debug=False)
    dataset = StandardGraphDataset(dataset_name=dataset_name, project_root=PROJECT_ROOT, config=config)
    clean = build_compatibility_graph_standard(dataset, config)
    compatibility, provenance = perturb_standard_active(clean, dataset_name, seed, setting)
    candidates = generate_candidates_standard(dataset, compatibility, config, seed)
    ranked = rerank_candidates_standard(candidates, compatibility, config)
    max_predictions = int(config.get("comot", {}).get("standard", {}).get("max_predictions", 1000))
    predictions = convert_candidates_to_predictions(ranked[:max_predictions], runtime=0.0)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(predictions, run_dir / "predictions.jsonl")
    write_standard_artifacts(compatibility, ranked, run_dir)
    evaluate_predictions(predictions, dataset.load_ground_truth(), run_dir)
    provenance.update({"candidate_pool_size": len(ranked), "prediction_count": len(predictions)})
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def aml_active_edges(source_run: Path) -> pd.DataFrame:
    path = source_run / "semotif_orchestrator_five_party" / "compatibility_graph_edges.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if df.empty:
        raise RuntimeError(f"No active AMLWorld compatibility edges in {path}")
    return df


def aml_row_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str, str], int]:
    lookup: Dict[Tuple[str, str, str], int] = {}
    for idx, row in df.iterrows():
        lookup[(str(row["tx_id"]), str(row["evidence_node_id"]), str(row["role"]))] = int(idx)
    return lookup


def edge_report_indices(edge: Mapping[str, Any], lookup: Mapping[Tuple[str, str, str], int]) -> Tuple[int, int] | None:
    left = lookup.get((str(edge["tx_id_sender"]), str(edge["src_node"]), "SENDER"))
    right = lookup.get((str(edge["tx_id_receiver"]), str(edge["dst_node"]), "RECEIVER"))
    if left is None or right is None:
        return None
    return left, right


def perturb_aml_active(seed: int, setting: str, source_run: Path, destination_dir: Path) -> Dict[str, Any]:
    rng = random.Random(protocol_seed("amlworld", seed, setting))
    reports, columns = base.read_aml_evidence(source_run / "semotif_local_evidences_five_party")
    active_edges = aml_active_edges(source_run)
    selected_edge_ids: Set[str] = set()
    with (source_run / "predictions.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                selected_edge_ids.update(str(edge_id) for edge_id in json.loads(line).get("predicted_edges", []))
    active_edges = active_edges[
        active_edges["tx_id_sender"].astype(str).isin(selected_edge_ids)
        | active_edges["tx_id_receiver"].astype(str).isin(selected_edge_ids)
    ].copy()
    lookup = aml_row_lookup(reports)
    units: List[Tuple[Dict[str, Any], int, int]] = []
    for edge in active_edges.to_dict("records"):
        indices = edge_report_indices(edge, lookup)
        if indices is not None:
            units.append((edge, indices[0], indices[1]))
    if not units:
        raise RuntimeError("No AMLWorld active boundary units resolved to source reports")
    count = int(round(RATE * len(units)))
    selected = rng.sample(units, count)
    max_tx = int(pd.to_numeric(reports["tx_id"], errors="coerce").max())

    if setting == "missing10":
        drop_indices = {index for _, left, right in selected for index in (left, right)}
        reports = reports.drop(index=list(drop_indices)).reset_index(drop=True)
    elif setting == "endpoint10":
        for unit_index, (_, left, right) in enumerate(selected):
            fake_tx = str(max_tx + unit_index + 1)
            reports.at[left, "tx_id"] = fake_tx
            reports.at[right, "tx_id"] = fake_tx
            corrupt_index = right if rng.getrandbits(1) else left
            bank = str(reports.at[corrupt_index, "bank_id"])
            account = f"WRONG_ENDPOINT_S{seed}_{unit_index}"
            reports.at[corrupt_index, "local_account_id"] = account
            reports.at[corrupt_index, "evidence_node_id"] = f"{bank}::{account}"
    elif setting == "spurious10":
        additions: List[Dict[str, Any]] = []
        # Groups of four false edges share a synthetic sender, creating false
        # fan structures that the candidate builder can actually consume.
        for unit_index, (_, left, right) in enumerate(selected):
            sender = reports.loc[left].to_dict()
            receiver = reports.loc[right].to_dict()
            group = unit_index // 4
            fake_tx = str(max_tx + unit_index + 1)
            tag = f"__spurious_active__amlworld__s{seed}__{unit_index}"
            sender_account = f"SPURIOUS_CENTER_S{seed}_{group}"
            receiver_account = f"SPURIOUS_LEAF_S{seed}_{unit_index}"
            sender["tx_id"] = fake_tx
            receiver["tx_id"] = fake_tx
            sender["public_tx_tag"] = tag
            receiver["public_tx_tag"] = tag
            sender["local_account_id"] = sender_account
            sender["evidence_node_id"] = f"{sender['bank_id']}::{sender_account}"
            receiver["local_account_id"] = receiver_account
            receiver["evidence_node_id"] = f"{receiver['bank_id']}::{receiver_account}"
            sender["motif_type"] = receiver["motif_type"] = "CLEAN"
            sender["motif_instance_id"] = receiver["motif_instance_id"] = "NONE"
            additions.extend((sender, receiver))
        reports = pd.concat([reports, pd.DataFrame(additions)], ignore_index=True)
    else:
        raise ValueError(setting)

    destination_dir.mkdir(parents=True, exist_ok=True)
    for source_name in sorted(set(reports["__source_file"])):
        reports[reports["__source_file"] == source_name][columns].to_csv(destination_dir / source_name, index=False)
    return {
        "protocol": PROTOCOL,
        "dataset": "amlworld",
        "seed": seed,
        "setting": setting,
        "rate": RATE,
        "random_seed": protocol_seed("amlworld", seed, setting),
        "boundary_unit": "workload-facing matched AMLWorld boundary-report pair used by the fixed clean prediction set",
        "global_boundary_report_rows": len(reports),
        "boundary_units_clean": len(units),
        "boundary_units_changed": count,
        "selection_uses_ground_truth": False,
        "endpoint_corruption_uses_fresh_report_id": setting == "endpoint10",
        "source": str(source_run.relative_to(PROJECT_ROOT)),
        "evaluator": "src/evaluation/evaluate.py:evaluate_predictions",
    }


def run_aml(seed: int, setting: str, run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    source_run = PROJECT_ROOT / "outputs" / "amlworld" / "comot" / f"seed_{seed}"
    evidence_dir = run_dir / "evidence"
    orchestrator_dir = run_dir / "orchestrator"
    candidate_dir = run_dir / "candidates"
    reranked_dir = run_dir / "reranked"
    provenance = perturb_aml_active(seed, setting, source_run, evidence_dir)
    log_path = run_dir / "run.log"
    base.run_logged(
        [sys.executable, str(SRC_ROOT / "methods/comot_pipeline/orchestrator.py"), "--evidence_dir", str(evidence_dir),
         "--output_dir", str(orchestrator_dir), "--threshold", "0.72", "--max_group_size", "5",
         "--topk_per_sender", "1", "--topk_per_receiver", "1"], log_path,
    )
    base.run_logged(
        [sys.executable, str(SRC_ROOT / "methods/comot_pipeline/candidate_builder.py"), "--orchestrator_dir", str(orchestrator_dir),
         "--output_dir", str(candidate_dir), "--chain_min_len", "3", "--chain_max_len", "10",
         "--cycle_min_len", "2", "--cycle_max_len", "8", "--fan_degree_thr", "3"], log_path,
    )
    base.run_logged(
        [sys.executable, str(SRC_ROOT / "methods/comot_pipeline/candidate_reranker.py"), "--candidate_dir", str(candidate_dir),
         "--output_dir", str(reranked_dir)], log_path,
    )
    config = load_json_or_yaml(source_run / "config.yaml")
    predictions = base.aml_predictions(reranked_dir / "candidates_reranked.csv", config)
    save_predictions(predictions, run_dir / "predictions.jsonl")
    ground_truth = json.loads((PROJECT_ROOT / "data/processed/amlworld/ground_truth.json").read_text(encoding="utf-8"))
    evaluate_predictions(predictions, ground_truth, run_dir)
    provenance["prediction_count"] = len(predictions)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/boundary_robustness_active_subset")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=SETTINGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for setting in args.settings:
        for dataset in args.datasets:
            for seed in args.seeds:
                run_dir = root / dataset / setting / f"seed_{seed}"
                print(f"[boundary-active] dataset={dataset} seed={seed} setting={setting}", flush=True)
                if setting == "clean":
                    metrics = base.copy_clean_metrics(dataset, seed, run_dir)
                    provenance = base.load_provenance(run_dir)
                    provenance["protocol"] = PROTOCOL
                    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
                elif dataset == "amlworld":
                    metrics = run_aml(seed, setting, run_dir)
                else:
                    metrics = run_standard(dataset, seed, setting, run_dir)
                rows.append(base.metric_row(dataset, seed, setting, metrics, base.load_provenance(run_dir)))
                gc.collect()
    if set(args.datasets) == set(DATASETS) and set(args.seeds) == set(SEEDS) and set(args.settings) == set(SETTINGS):
        base.summarize(rows, root)
    print(json.dumps({"protocol": PROTOCOL, "runs": len(rows), "output_root": str(root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
