#!/usr/bin/env python3
"""Run the four-dataset boundary-evidence robustness experiment.

The experiment preserves the main-experiment queries, ground truth, method
configuration, prediction budget, and evaluator.  It changes only the boundary
view consumed by CoMot:

* clean: reuse and verify the fixed main-experiment result;
* missing10: remove 10% of boundary edges/reports;
* spurious10: add 10% false boundary edges/reports;
* endpoint10: corrupt one endpoint of 10% of boundary edges/reports.

AMLWorld uses the AMLWorld evidence-record pipeline.  The other datasets use the
standard compatibility-view pipeline, whose boundary edges are exactly the
edges visible in more than one partition.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.evaluate import evaluate_predictions  # noqa: E402
from evaluation.metrics import parse_json_list  # noqa: E402
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
SETTINGS = ["clean", "missing10", "spurious10", "endpoint10"]
RATE = 0.10


def strip_partition_prefix(node: Any) -> str:
    text = str(node)
    return text.split("::", 1)[1] if "::" in text else text


def stable_seed(dataset: str, seed: int, setting: str) -> int:
    digest = hashlib.sha256(f"boundary-robustness|{dataset}|{seed}|{setting}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def motif_f1(metrics: Mapping[str, Any]) -> float:
    motif = metrics.get("motif_level", {})
    precision = float(motif.get("partial_match") or 0.0)
    recall = float(motif.get("motif_recall") or 0.0)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def metric_row(dataset: str, seed: int, setting: str, metrics: Mapping[str, Any], provenance: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": seed,
        "setting": setting,
        "motif_f1": motif_f1(metrics),
        "edge_f1": float(metrics.get("edge_level", {}).get("f1") or 0.0),
        "node_f1": float(metrics.get("node_level", {}).get("f1") or 0.0),
        "num_candidates": int(metrics.get("efficiency", {}).get("num_candidates") or 0),
        "boundary_units_clean": provenance.get("boundary_units_clean", ""),
        "boundary_units_changed": provenance.get("boundary_units_changed", 0),
        "source": provenance.get("source", ""),
    }


def copy_clean_metrics(dataset: str, seed: int, run_dir: Path) -> Dict[str, Any]:
    source = PROJECT_ROOT / "outputs" / dataset / "comot" / f"seed_{seed}" / "metrics.json"
    if not source.exists():
        raise FileNotFoundError(source)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, run_dir / "metrics.json")
    provenance = {
        "dataset": dataset,
        "seed": seed,
        "setting": "clean",
        "rate": 0.0,
        "source": str(source.relative_to(PROJECT_ROOT)),
        "reused_main_result": True,
        "evaluator": "src/evaluation/evaluate.py:evaluate_predictions",
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def clean_parts(values: Iterable[str]) -> Set[str]:
    return {str(value) for value in values if str(value) and str(value) != "UNKNOWN"}


def boundary_edge_ids(compatibility: Mapping[str, Any]) -> List[str]:
    return sorted(
        str(edge_id)
        for edge_id, parts in compatibility["edge_to_partitions"].items()
        if len(clean_parts(parts)) > 1 and str(edge_id) in compatibility["edge_lookup"]
    )


def node_owner(compatibility: Mapping[str, Any]) -> Dict[str, str]:
    owners: Dict[str, str] = {}
    for node, parts in compatibility["node_to_partitions"].items():
        cleaned = sorted(clean_parts(parts))
        if cleaned:
            owners[str(node)] = cleaned[0]
    return owners


def rebuild_compatibility(
    edge_lookup: Mapping[str, Tuple[str, str]],
    node_to_partitions: Mapping[str, Set[str]],
    original_edge_parts: Mapping[str, Set[str]],
) -> Dict[str, Any]:
    directed_adj: MutableMapping[str, List[Tuple[str, str]]] = defaultdict(list)
    reverse_adj: MutableMapping[str, List[Tuple[str, str]]] = defaultdict(list)
    undirected_adj: MutableMapping[str, List[Tuple[str, str]]] = defaultdict(list)
    directed_pair_edge: Dict[Tuple[str, str], str] = {}
    undirected_pair_edge: Dict[frozenset, str] = {}
    edge_to_partitions: MutableMapping[str, Set[str]] = defaultdict(set)

    for edge_id, pair in edge_lookup.items():
        src, dst = str(pair[0]), str(pair[1])
        edge_id = str(edge_id)
        directed_adj[src].append((dst, edge_id))
        reverse_adj[dst].append((src, edge_id))
        undirected_adj[src].append((dst, edge_id))
        undirected_adj[dst].append((src, edge_id))
        directed_pair_edge.setdefault((src, dst), edge_id)
        undirected_pair_edge.setdefault(frozenset((src, dst)), edge_id)
        parts = clean_parts(node_to_partitions.get(src, set())) | clean_parts(node_to_partitions.get(dst, set()))
        edge_to_partitions[edge_id].update(parts or original_edge_parts.get(edge_id, {"UNKNOWN"}))

    for adjacency in (directed_adj, reverse_adj, undirected_adj):
        for node in adjacency:
            adjacency[node].sort(key=lambda item: (item[0], item[1]))

    return {
        "directed_adj": directed_adj,
        "reverse_adj": reverse_adj,
        "undirected_adj": undirected_adj,
        "edge_lookup": dict(edge_lookup),
        "directed_pair_edge": directed_pair_edge,
        "undirected_pair_edge": undirected_pair_edge,
        "node_to_partitions": node_to_partitions,
        "edge_to_partitions": edge_to_partitions,
    }


def perturb_standard_compatibility(
    compatibility: Mapping[str, Any], dataset: str, seed: int, setting: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rng = random.Random(stable_seed(dataset, seed, setting))
    boundary_ids = boundary_edge_ids(compatibility)
    count = int(round(RATE * len(boundary_ids)))
    chosen = set(rng.sample(boundary_ids, count))
    edges = {str(edge_id): (str(pair[0]), str(pair[1])) for edge_id, pair in compatibility["edge_lookup"].items()}
    owners = node_owner(compatibility)
    nodes_by_owner: Dict[str, List[str]] = defaultdict(list)
    for node, owner in owners.items():
        nodes_by_owner[owner].append(node)
    for nodes in nodes_by_owner.values():
        nodes.sort()

    if setting == "missing10":
        for edge_id in chosen:
            edges.pop(edge_id, None)
    elif setting == "endpoint10":
        for edge_id in chosen:
            src, dst = edges[edge_id]
            corrupt_dst = bool(rng.getrandbits(1))
            fixed, old = (src, dst) if corrupt_dst else (dst, src)
            owner = owners.get(old)
            pool = nodes_by_owner.get(owner or "", [])
            if len(pool) > 1:
                replacement = pool[rng.randrange(len(pool))]
                while replacement == old:
                    replacement = pool[rng.randrange(len(pool))]
            else:
                replacement = old
            edges[edge_id] = (fixed, replacement) if corrupt_dst else (replacement, fixed)
    elif setting == "spurious10":
        partitions = sorted(nodes_by_owner)
        existing_pairs = set(edges.values())
        added = 0
        attempts = 0
        while added < count:
            attempts += 1
            if attempts > max(1000, count * 50):
                raise RuntimeError(f"Could not sample {count} distinct false boundary edges")
            left_owner, right_owner = rng.sample(partitions, 2)
            src = rng.choice(nodes_by_owner[left_owner])
            dst = rng.choice(nodes_by_owner[right_owner])
            if src == dst or (src, dst) in existing_pairs:
                continue
            edge_id = f"__spurious_boundary__{dataset}__s{seed}__{added}"
            edges[edge_id] = (src, dst)
            existing_pairs.add((src, dst))
            added += 1
    else:
        raise ValueError(setting)

    perturbed = rebuild_compatibility(
        edges,
        compatibility["node_to_partitions"],
        compatibility["edge_to_partitions"],
    )
    provenance = {
        "dataset": dataset,
        "seed": seed,
        "setting": setting,
        "rate": RATE,
        "random_seed": stable_seed(dataset, seed, setting),
        "boundary_unit": "cross-partition edge in the standard compatibility view",
        "boundary_units_clean": len(boundary_ids),
        "boundary_units_changed": count,
        "source": f"data/processed/{dataset}/partitions.json + edges.csv",
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
    compatibility, provenance = perturb_standard_compatibility(clean, dataset_name, seed, setting)
    start = time.perf_counter()
    candidates = generate_candidates_standard(dataset, compatibility, config, seed)
    ranked = rerank_candidates_standard(candidates, compatibility, config)
    max_predictions = int(config.get("comot", {}).get("standard", {}).get("max_predictions", 1000))
    predictions = convert_candidates_to_predictions(ranked[:max_predictions], runtime=time.perf_counter() - start)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(predictions, run_dir / "predictions.jsonl")
    write_standard_artifacts(compatibility, ranked, run_dir)
    evaluate_predictions(predictions, dataset.load_ground_truth(), run_dir)
    provenance["candidate_pool_size"] = len(ranked)
    provenance["prediction_count"] = len(predictions)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def read_aml_evidence(source_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    frames = []
    columns: List[str] = []
    for path in sorted(source_dir.glob("*_evidences.csv")):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not columns:
            columns = list(frame.columns)
        frame["__source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No AMLWorld evidence CSVs in {source_dir}")
    return pd.concat(frames, ignore_index=True), columns


def perturb_aml_evidence(dataset_seed: int, setting: str, source_dir: Path, destination_dir: Path) -> Dict[str, Any]:
    rng = random.Random(stable_seed("amlworld", dataset_seed, setting))
    df, columns = read_aml_evidence(source_dir)
    clean_count = len(df)
    count = int(round(RATE * clean_count))
    all_indices = list(range(clean_count))

    if setting == "missing10":
        selected = set(rng.sample(all_indices, count))
        df = df.drop(index=list(selected)).reset_index(drop=True)
    elif setting == "endpoint10":
        selected = rng.sample(all_indices, count)
        pools = {
            bank: group[["local_account_id", "evidence_node_id"]].drop_duplicates().values.tolist()
            for bank, group in df.groupby("bank_id", sort=False)
        }
        for idx in selected:
            bank = str(df.at[idx, "bank_id"])
            old_account = str(df.at[idx, "local_account_id"])
            pool = pools[bank]
            if len(pool) <= 1:
                continue
            account, evidence_node = rng.choice(pool)
            while str(account) == old_account:
                account, evidence_node = rng.choice(pool)
            df.at[idx, "local_account_id"] = str(account)
            df.at[idx, "evidence_node_id"] = str(evidence_node)
    elif setting == "spurious10":
        senders = df[df["role"] == "SENDER"]
        receivers = df[df["role"] == "RECEIVER"]
        sender_indices = list(senders.index)
        receiver_indices = list(receivers.index)
        max_tx = pd.to_numeric(df["tx_id"], errors="coerce").max()
        next_tx = int(max_tx if not pd.isna(max_tx) else 0) + 1
        additions: List[Dict[str, Any]] = []
        pairs = count // 2
        for pair_index in range(pairs):
            sender = df.loc[rng.choice(sender_indices)].to_dict()
            receiver = df.loc[rng.choice(receiver_indices)].to_dict()
            attempts = 0
            while receiver["bank_id"] == sender["bank_id"] and attempts < 100:
                receiver = df.loc[rng.choice(receiver_indices)].to_dict()
                attempts += 1
            tag = f"__spurious_boundary__amlworld__s{dataset_seed}__{pair_index}"
            fake_tx = str(next_tx + pair_index)
            for row in (sender, receiver):
                row["public_tx_tag"] = tag
                row["tx_id"] = fake_tx
                row["motif_type"] = "CLEAN"
                row["motif_instance_id"] = "NONE"
                additions.append(row)
        if count % 2:
            orphan = df.loc[rng.choice(sender_indices)].to_dict()
            orphan["public_tx_tag"] = f"__spurious_boundary__amlworld__s{dataset_seed}__orphan"
            orphan["tx_id"] = str(next_tx + pairs)
            orphan["motif_type"] = "CLEAN"
            orphan["motif_instance_id"] = "NONE"
            additions.append(orphan)
        df = pd.concat([df, pd.DataFrame(additions)], ignore_index=True)
    else:
        raise ValueError(setting)

    destination_dir.mkdir(parents=True, exist_ok=True)
    source_names = sorted(set(df["__source_file"]))
    for source_name in source_names:
        out = df[df["__source_file"] == source_name][columns]
        out.to_csv(destination_dir / source_name, index=False)

    return {
        "dataset": "amlworld",
        "seed": dataset_seed,
        "setting": setting,
        "rate": RATE,
        "random_seed": stable_seed("amlworld", dataset_seed, setting),
        "boundary_unit": "AMLWorld local boundary evidence report row",
        "boundary_units_clean": clean_count,
        "boundary_units_changed": count,
        "source": str(source_dir.relative_to(PROJECT_ROOT)),
        "evaluator": "src/evaluation/evaluate.py:evaluate_predictions",
    }


def aml_predictions(reranked_path: Path, config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    with reranked_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            predictions.append(
                {
                    "query_id": row.get("candidate_type", "comot"),
                    "predicted_nodes": [strip_partition_prefix(node) for node in parse_json_list(row.get("nodes_json"))],
                    "predicted_edges": [str(edge_id) for edge_id in parse_json_list(row.get("tx_ids_json"))],
                    "score": float(row.get("rerank_score") or row.get("score_mean") or 0.0),
                    "runtime": 0.0,
                    "metadata": {
                        "method": "comot",
                        "candidate_id": row.get("candidate_id", ""),
                        "candidate_type": row.get("candidate_type", ""),
                        "runner": "comot_boundary_robustness",
                    },
                }
            )
    return predictions


def run_logged(command: Sequence[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(list(command), cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_aml(seed: int, setting: str, run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    source_run = PROJECT_ROOT / "outputs" / "amlworld" / "comot" / f"seed_{seed}"
    source_evidence = source_run / "semotif_local_evidences_five_party"
    evidence_dir = run_dir / "evidence"
    orchestrator_dir = run_dir / "orchestrator"
    candidate_dir = run_dir / "candidates"
    reranked_dir = run_dir / "reranked"
    provenance = perturb_aml_evidence(seed, setting, source_evidence, evidence_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    run_logged(
        [
            sys.executable,
            str(SRC_ROOT / "methods/comot_pipeline/orchestrator.py"),
            "--evidence_dir", str(evidence_dir),
            "--output_dir", str(orchestrator_dir),
            "--threshold", "0.72",
            "--max_group_size", "5",
            "--topk_per_sender", "1",
            "--topk_per_receiver", "1",
        ],
        log_path,
    )
    run_logged(
        [
            sys.executable,
            str(SRC_ROOT / "methods/comot_pipeline/candidate_builder.py"),
            "--orchestrator_dir", str(orchestrator_dir),
            "--output_dir", str(candidate_dir),
            "--chain_min_len", "3",
            "--chain_max_len", "10",
            "--cycle_min_len", "2",
            "--cycle_max_len", "8",
            "--fan_degree_thr", "3",
        ],
        log_path,
    )
    run_logged(
        [
            sys.executable,
            str(SRC_ROOT / "methods/comot_pipeline/candidate_reranker.py"),
            "--candidate_dir", str(candidate_dir),
            "--output_dir", str(reranked_dir),
        ],
        log_path,
    )

    config = load_json_or_yaml(source_run / "config.yaml")
    predictions = aml_predictions(reranked_dir / "candidates_reranked.csv", config)
    save_predictions(predictions, run_dir / "predictions.jsonl")
    ground_truth = json.loads((PROJECT_ROOT / "data" / "processed" / "amlworld" / "ground_truth.json").read_text(encoding="utf-8"))
    evaluate_predictions(predictions, ground_truth, run_dir)
    provenance["prediction_count"] = len(predictions)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_provenance(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "provenance.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def summarize(rows: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(output_root / "boundary_robustness_by_seed.csv", index=False, float_format="%.9f")
    metrics = ["motif_f1", "edge_f1", "node_f1"]
    summary_rows: List[Dict[str, Any]] = []
    for (setting, dataset), group in by_seed.groupby(["setting", "dataset"], sort=False):
        row: Dict[str, Any] = {"setting": setting, "dataset": dataset, "num_seeds": group["seed"].nunique()}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    setting_order = {name: idx for idx, name in enumerate(SETTINGS)}
    dataset_order = {name: idx for idx, name in enumerate(DATASETS)}
    summary["__s"] = summary["setting"].map(setting_order)
    summary["__d"] = summary["dataset"].map(dataset_order)
    summary = summary.sort_values(["__s", "__d"]).drop(columns=["__s", "__d"])
    summary.to_csv(output_root / "boundary_robustness_summary.csv", index=False, float_format="%.9f")

    wide_rows: List[Dict[str, Any]] = []
    for setting in SETTINGS:
        wide: Dict[str, Any] = {"setting": setting}
        for dataset in DATASETS:
            cell = summary[(summary["setting"] == setting) & (summary["dataset"] == dataset)].iloc[0]
            for metric, label in [("motif_f1", "M-F1"), ("edge_f1", "E-F1"), ("node_f1", "N-F1")]:
                wide[f"{dataset}_{label}"] = f"{100 * cell[f'{metric}_mean']:.2f} +/- {100 * cell[f'{metric}_std']:.2f}"
        wide_rows.append(wide)
    wide_df = pd.DataFrame(wide_rows)
    wide_df.to_csv(output_root / "boundary_robustness_table_4x12.csv", index=False)
    (output_root / "boundary_robustness_table.md").write_text(wide_df.to_markdown(index=False) + "\n", encoding="utf-8")

    clean = summary[summary["setting"] == "clean"].set_index("dataset")
    expected = {
        "amlworld": (75.98, 6.51, 20.44),
        "elliptic": (90.35, 73.23, 82.07),
        "dblp": (93.50, 26.21, 50.96),
        "web_google": (51.85, 10.92, 43.05),
    }
    checks = {}
    for dataset, target in expected.items():
        observed = tuple(round(100 * float(clean.loc[dataset, f"{metric}_mean"]), 2) for metric in metrics)
        checks[dataset] = {"expected_percent": target, "observed_percent": observed, "match": observed == target}
    (output_root / "clean_reproduction_check.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "boundary_robustness")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=SETTINGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for setting in args.settings:
        for dataset in args.datasets:
            for seed in args.seeds:
                run_dir = output_root / dataset / setting / f"seed_{seed}"
                print(f"[boundary-robustness] dataset={dataset} seed={seed} setting={setting}", flush=True)
                if setting == "clean":
                    metrics = copy_clean_metrics(dataset, seed, run_dir)
                elif dataset == "amlworld":
                    metrics = run_aml(seed, setting, run_dir)
                else:
                    metrics = run_standard(dataset, seed, setting, run_dir)
                rows.append(metric_row(dataset, seed, setting, metrics, load_provenance(run_dir)))
                gc.collect()
    if set(args.datasets) == set(DATASETS) and set(args.seeds) == set(SEEDS) and set(args.settings) == set(SETTINGS):
        summarize(rows, output_root)
    print(json.dumps({"runs": len(rows), "output_root": str(output_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
