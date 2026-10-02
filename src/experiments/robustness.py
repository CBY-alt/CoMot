#!/usr/bin/env python3
"""Paired boundary robustness at 10%, 30%, and 50%.

The experiment does not change the method,
queries, ground truth, prediction budget, or evaluator.  For each dataset and
seed, the 10% subset is exactly the active subset.  The 30% and 50% subsets
contain that 10% subset and are nested in one another.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments import boundary_common as base  # noqa: E402
from experiments import boundary_active as active  # noqa: E402
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


DATASETS = ["amlworld", "elliptic", "dblp", "web_google"]
SEEDS = [0, 1, 2]
KINDS = ["missing", "spurious", "endpoint"]
RATES = [10, 30, 50]
SETTINGS = ["clean"] + [f"{kind}{rate}" for kind in KINDS for rate in RATES]
PROTOCOL = "paired-nested-rates"


def parse_setting(setting: str) -> Tuple[str, int, float]:
    if setting == "clean":
        return "clean", 0, 0.0
    for kind in KINDS:
        if setting.startswith(kind):
            percent = int(setting[len(kind):])
            if percent not in RATES:
                raise ValueError(setting)
            return kind, percent, percent / 100.0
    raise ValueError(setting)


def extension_seed(dataset: str, seed: int) -> int:
    value = hashlib.sha256(f"{PROTOCOL}|extension|{dataset}|{seed}".encode()).digest()
    return int.from_bytes(value[:8], "big")


def nested_indices(size: int, dataset: str, seed: int, percent: int) -> List[int]:
    """Return nested indices while preserving the 10% sample.

    The 10% condition used ``Random(seed).sample(population, round(.1*N))``.  We retain
    that exact ordered prefix, then deterministically permute the remainder
    using an independent seed so sampling more units cannot alter the 10% arm.
    """
    count10 = int(round(0.10 * size))
    target = int(round((percent / 100.0) * size))
    initial_rng = random.Random(seed)
    prefix = initial_rng.sample(range(size), count10)
    used = set(prefix)
    remainder = [index for index in range(size) if index not in used]
    extension_rng = random.Random(extension_seed(dataset, seed))
    extension_rng.shuffle(remainder)
    return (prefix + remainder)[:target]


def effect_rng(population_size: int, seed: int) -> random.Random:
    """Reproduce the RNG state immediately after the 10% sample."""
    rng = random.Random(seed)
    rng.sample(range(population_size), int(round(0.10 * population_size)))
    return rng


def perturb_standard(
    compatibility: Mapping[str, Any], dataset: str, seed: int, setting: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    kind, percent, rate = parse_setting(setting)
    global_boundary = set(base.boundary_edge_ids(compatibility))
    active_ids = active.load_standard_active_edges(dataset, seed, global_boundary)
    selected = [active_ids[index] for index in nested_indices(len(active_ids), dataset, seed, percent)]
    rng = effect_rng(len(active_ids), seed)
    edges = {str(edge_id): (str(pair[0]), str(pair[1])) for edge_id, pair in compatibility["edge_lookup"].items()}
    node_parts = {str(node): set(parts) for node, parts in compatibility["node_to_partitions"].items()}
    owners = base.node_owner(compatibility)

    if kind == "missing":
        for edge_id in selected:
            edges.pop(edge_id, None)
    elif kind == "endpoint":
        for index, edge_id in enumerate(selected):
            src, dst = edges.pop(edge_id)
            change_dst = bool(rng.getrandbits(1))
            old = dst if change_dst else src
            # IDs depend on the nested unit position, not the rate, so a unit
            # shared by 10/30/50% is byte-identical in every arm.
            synthetic = f"__wrong_endpoint__{dataset}__s{seed}__{index}"
            node_parts[synthetic] = {owners.get(old, "UNKNOWN")}
            corrupted_id = f"__corrupt_report__{dataset}__s{seed}__{index}"
            edges[corrupted_id] = (src, synthetic) if change_dst else (synthetic, dst)
    elif kind == "spurious":
        for index, donor in enumerate(selected):
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
        "perturbation": kind,
        "rate": rate,
        "sampling_seed": seed,
        "extension_seed": extension_seed(dataset, seed),
        "nested_sampling": True,
        "boundary_unit": "workload-facing cross-partition edge used by the fixed clean prediction set",
        "global_boundary_units": len(global_boundary),
        "boundary_units_clean": len(active_ids),
        "boundary_units_changed": len(selected),
        "selection_uses_ground_truth": False,
        "endpoint_corruption_uses_fresh_report_id": kind == "endpoint",
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
    compatibility, provenance = perturb_standard(clean, dataset_name, seed, setting)
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


def perturb_aml(seed: int, setting: str, source_run: Path, destination_dir: Path) -> Dict[str, Any]:
    kind, percent, rate = parse_setting(setting)
    reports, columns = base.read_aml_evidence(source_run / "semotif_local_evidences_five_party")
    active_edges = active.aml_active_edges(source_run)
    selected_edge_ids: Set[str] = set()
    with (source_run / "predictions.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                selected_edge_ids.update(str(edge_id) for edge_id in json.loads(line).get("predicted_edges", []))
    active_edges = active_edges[
        active_edges["tx_id_sender"].astype(str).isin(selected_edge_ids)
        | active_edges["tx_id_receiver"].astype(str).isin(selected_edge_ids)
    ].copy()
    lookup = active.aml_row_lookup(reports)
    units: List[Tuple[Dict[str, Any], int, int]] = []
    for edge in active_edges.to_dict("records"):
        indices = active.edge_report_indices(edge, lookup)
        if indices is not None:
            units.append((edge, indices[0], indices[1]))
    if not units:
        raise RuntimeError("No AMLWorld active boundary units resolved to source reports")
    selected = [units[index] for index in nested_indices(len(units), "amlworld", seed, percent)]
    rng = effect_rng(len(units), seed)
    max_tx = int(pd.to_numeric(reports["tx_id"], errors="coerce").max())

    if kind == "missing":
        drop_indices = {index for _, left, right in selected for index in (left, right)}
        reports = reports.drop(index=list(drop_indices)).reset_index(drop=True)
    elif kind == "endpoint":
        for unit_index, (_, left, right) in enumerate(selected):
            fake_tx = str(max_tx + unit_index + 1)
            reports.at[left, "tx_id"] = fake_tx
            reports.at[right, "tx_id"] = fake_tx
            corrupt_index = right if rng.getrandbits(1) else left
            bank = str(reports.at[corrupt_index, "bank_id"])
            account = f"WRONG_ENDPOINT_S{seed}_{unit_index}"
            reports.at[corrupt_index, "local_account_id"] = account
            reports.at[corrupt_index, "evidence_node_id"] = f"{bank}::{account}"
    elif kind == "spurious":
        additions: List[Dict[str, Any]] = []
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
        "perturbation": kind,
        "rate": rate,
        "sampling_seed": seed,
        "extension_seed": extension_seed("amlworld", seed),
        "nested_sampling": True,
        "boundary_unit": "workload-facing matched AMLWorld boundary-report pair used by the fixed clean prediction set",
        "boundary_units_clean": len(units),
        "boundary_units_changed": len(selected),
        "selection_uses_ground_truth": False,
        "endpoint_corruption_uses_fresh_report_id": kind == "endpoint",
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
    provenance = perturb_aml(seed, setting, source_run, evidence_dir)
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


def run_cell(root: Path, dataset: str, seed: int, setting: str) -> Dict[str, Any]:
    run_dir = root / dataset / setting / f"seed_{seed}"
    if setting == "clean":
        metrics = base.copy_clean_metrics(dataset, seed, run_dir)
        provenance = base.load_provenance(run_dir)
        provenance.update({"protocol": PROTOCOL, "nested_sampling": True})
        (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    elif dataset == "amlworld":
        metrics = run_aml(seed, setting, run_dir)
    else:
        metrics = run_standard(dataset, seed, setting, run_dir)
    provenance = base.load_provenance(run_dir)
    provenance["python_hash_seed"] = os.environ.get("PYTHONHASHSEED", "UNSET")
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return base.metric_row(dataset, seed, setting, metrics, provenance)


def collect_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    missing = []
    for setting in SETTINGS:
        for dataset in DATASETS:
            for seed in SEEDS:
                run_dir = root / dataset / setting / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                if not metrics_path.exists():
                    missing.append(f"{dataset}/{setting}/seed_{seed}")
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                rows.append(base.metric_row(dataset, seed, setting, metrics, base.load_provenance(run_dir)))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} cells: {missing[:10]}")
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], root: Path) -> None:
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(root / "boundary_robustness_by_seed.csv", index=False, float_format="%.9f")
    summary_rows: List[Dict[str, Any]] = []
    metrics = ["motif_f1", "edge_f1", "node_f1"]
    for setting in SETTINGS:
        for dataset in DATASETS:
            group = by_seed[(by_seed["setting"] == setting) & (by_seed["dataset"] == dataset)]
            row: Dict[str, Any] = {"setting": setting, "dataset": dataset, "num_seeds": group["seed"].nunique()}
            for metric in metrics:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=1)
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "boundary_robustness_summary.csv", index=False, float_format="%.9f")

    wide_rows = []
    for setting in SETTINGS:
        wide: Dict[str, Any] = {"setting": setting}
        for dataset in DATASETS:
            cell = summary[(summary["setting"] == setting) & (summary["dataset"] == dataset)].iloc[0]
            for metric, label in [("motif_f1", "M-F1"), ("edge_f1", "E-F1"), ("node_f1", "N-F1")]:
                wide[f"{dataset}_{label}"] = f"{100 * cell[f'{metric}_mean']:.2f} +/- {100 * cell[f'{metric}_std']:.2f}"
        wide_rows.append(wide)
    wide_df = pd.DataFrame(wide_rows)
    wide_df.to_csv(root / "boundary_robustness_full_table.csv", index=False)
    (root / "boundary_robustness_full_table.md").write_text(wide_df.to_markdown(index=False) + "\n", encoding="utf-8")

    motif_rows = []
    labels = {"clean": "Clean"}
    labels.update({f"{kind}{rate}": f"{kind.title()} {rate}%" for kind in KINDS for rate in RATES})
    for setting in SETTINGS:
        row = {"Setting": labels[setting]}
        for dataset in DATASETS:
            cell = summary[(summary["setting"] == setting) & (summary["dataset"] == dataset)].iloc[0]
            row[dataset] = f"{100 * cell['motif_f1_mean']:.2f} +/- {100 * cell['motif_f1_std']:.2f}"
        motif_rows.append(row)
    motif_df = pd.DataFrame(motif_rows)
    motif_df.to_csv(root / "boundary_robustness_motif_f1.csv", index=False)
    (root / "boundary_robustness_motif_f1.md").write_text(motif_df.to_markdown(index=False) + "\n", encoding="utf-8")

    old = PROJECT_ROOT / "outputs/boundary_robustness_active_subset/boundary_robustness_by_seed.csv"
    check: Dict[str, Any] = {"reference": str(old.relative_to(PROJECT_ROOT)), "cells": {}}
    if old.exists():
        old_df = pd.read_csv(old)
        for dataset in DATASETS:
            for seed in SEEDS:
                for kind in KINDS:
                    setting = f"{kind}10"
                    a = by_seed[(by_seed.dataset == dataset) & (by_seed.seed == seed) & (by_seed.setting == setting)].iloc[0]
                    b = old_df[(old_df.dataset == dataset) & (old_df.seed == seed) & (old_df.setting == setting)].iloc[0]
                    diffs = {metric: abs(float(a[metric]) - float(b[metric])) for metric in metrics}
                    check["cells"][f"{dataset}/{seed}/{setting}"] = {"max_abs_diff": max(diffs.values()), "diffs": diffs}
        check["all_exact_to_9_decimals"] = all(v["max_abs_diff"] < 5e-10 for v in check["cells"].values())
    (root / "ten_percent_reproduction_check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "protocol": PROTOCOL,
        "datasets": DATASETS,
        "seeds": SEEDS,
        "settings": SETTINGS,
        "cells": len(rows),
        "method": "CoMot",
        "evaluator": "src/evaluation/evaluate.py:evaluate_predictions",
        "prediction_budget_unchanged": True,
        "ground_truth_unchanged": True,
        "queries_unchanged": True,
        "nested_rates": True,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "UNSET"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/boundary_robustness")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--setting", choices=SETTINGS)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        rows = collect_rows(root)
        summarize(rows, root)
        print(json.dumps({"protocol": PROTOCOL, "cells": len(rows), "output_root": str(root)}, indent=2))
        return
    if args.dataset is None or args.seed is None or args.setting is None:
        raise SystemExit("--dataset, --seed, and --setting are required for a cell run")
    print(f"[boundary-robustness] dataset={args.dataset} seed={args.seed} setting={args.setting}", flush=True)
    row = run_cell(root, args.dataset, args.seed, args.setting)
    gc.collect()
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
