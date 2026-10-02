import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = "amlworld"
METHOD = "comot"
REGIME = "controlled_balanced"
TAG = "five_party"
VARIANTS = [
    "full_generation",
    "no_compatibility_filtering",
    "no_structural_filtering",
    "anchor_only_generation",
    "random_cross_partition_generation",
    "raw_candidate_pool",
]
STUDY_COLUMNS = [
    "dataset",
    "method",
    "regime",
    "seed",
    "generation_variant",
    "status",
    "definition",
    "num_candidates",
    "candidate_purity",
    "motif_recall_all",
    "motif_type_coverage",
    "candidate_type_distribution",
    "avg_candidate_size",
    "runtime",
    "source",
    "notes",
]
DIST_COLUMNS = ["generation_variant", "candidate_type", "num_candidates", "fraction"]


DEFINITIONS = {
    "full_generation": "Existing controlled-balanced CoMot candidate generation: local evidence, compatibility graph threshold/top-k filtering, then chain/fan/cycle structural candidate extraction.",
    "no_compatibility_filtering": "Offline ablation using pre-filter alignment_pairs_scored rows directly as pair candidates, without threshold/top-k compatibility filtering or structural construction.",
    "no_structural_filtering": "Offline ablation using retained compatibility graph edges directly as pair candidates, without chain/fan/cycle structural constraints.",
    "anchor_only_generation": "Weak baseline candidate generator: use method-facing anchor_nodes only, expand to local incident transactions, and do not use the compatibility graph.",
    "random_cross_partition_generation": "Random baseline candidate generator: sample visible cross-partition transactions with a fixed seed and match the full-generation candidate count.",
    "raw_candidate_pool": "Existing unreranked candidate pool emitted before reranking; same candidate set as full_generation, evaluated without rerank-specific changes.",
}
CANDIDATE_COLUMNS = [
    "candidate_id",
    "candidate_type",
    "num_nodes",
    "num_edges",
    "score_mean",
    "score_min",
    "nodes_json",
    "tx_ids_json",
]


def parse_json_list(value: Any) -> List[Any]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.6f}"
    return str(value)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_candidates(candidate_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ["chain_candidates.csv", "fan_candidates.csv", "cycle_candidates.csv"]:
        path = candidate_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No candidate files found in {candidate_dir}")
    return pd.concat(frames, ignore_index=True)


def load_gt(partition_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Set[str]], Dict[str, str], Dict[str, Set[str]]]:
    gt = pd.read_csv(partition_dir / "ground_truth_motifs.csv")
    motif_to_txs = {
        str(row["motif_instance_id"]): {str(x) for x in parse_json_list(row["tx_ids_json"])}
        for _, row in gt.iterrows()
    }
    motif_to_type = {str(row["motif_instance_id"]): str(row["motif_type"]) for _, row in gt.iterrows()}
    tx_to_motifs: Dict[str, Set[str]] = {}
    for motif_id, txs in motif_to_txs.items():
        for tx_id in txs:
            tx_to_motifs.setdefault(tx_id, set()).add(motif_id)
    return gt, motif_to_txs, motif_to_type, tx_to_motifs


def candidate_hits(row: pd.Series, tx_to_motifs: Dict[str, Set[str]]) -> List[str]:
    txs = {str(x) for x in parse_json_list(row.get("tx_ids_json"))}
    hits: Set[str] = set()
    for tx_id in txs:
        hits.update(tx_to_motifs.get(tx_id, set()))
    return sorted(hits)


def evaluate_candidate_pool(
    variant: str,
    df: pd.DataFrame,
    gt_df: pd.DataFrame,
    tx_to_motifs: Dict[str, Set[str]],
    motif_to_type: Dict[str, str],
    source: str,
    runtime: Any,
    notes: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    hit_rows = []
    hit_motifs: Set[str] = set()
    hit_motif_types: Set[str] = set()
    for _, row in df.iterrows():
        hits = candidate_hits(row, tx_to_motifs)
        hit_motifs.update(hits)
        hit_motif_types.update(motif_to_type[motif_id] for motif_id in hits if motif_id in motif_to_type)
        hit_rows.append(1 if hits else 0)

    distribution_rows = []
    counts = df["candidate_type"].astype(str).value_counts().sort_index().to_dict() if len(df) else {}
    for candidate_type, count in counts.items():
        distribution_rows.append({
            "generation_variant": variant,
            "candidate_type": candidate_type,
            "num_candidates": int(count),
            "fraction": float(count / len(df)) if len(df) else 0.0,
        })

    avg_size = 0.0
    if len(df):
        avg_size = float(((pd.to_numeric(df["num_nodes"], errors="coerce").fillna(0) + pd.to_numeric(df["num_edges"], errors="coerce").fillna(0)) / 2.0).mean())

    row = {
        "dataset": DATASET,
        "method": METHOD,
        "regime": REGIME,
        "seed": 0,
        "generation_variant": variant,
        "status": "completed",
        "definition": DEFINITIONS[variant],
        "num_candidates": int(len(df)),
        "candidate_purity": float(sum(hit_rows) / len(hit_rows)) if hit_rows else 0.0,
        "motif_recall_all": float(len(hit_motifs) / len(gt_df)) if len(gt_df) else 0.0,
        "motif_type_coverage": float(len(hit_motif_types) / len(set(motif_to_type.values()))) if motif_to_type else 0.0,
        "candidate_type_distribution": json.dumps({str(k): int(v) for k, v in counts.items()}, sort_keys=True),
        "avg_candidate_size": avg_size,
        "runtime": runtime,
        "source": source,
        "notes": notes,
    }
    return row, distribution_rows


def not_available_row(variant: str, reason: str) -> Dict[str, Any]:
    return {
        "dataset": DATASET,
        "method": METHOD,
        "regime": REGIME,
        "seed": 0,
        "generation_variant": variant,
        "status": "not_available",
        "definition": DEFINITIONS[variant],
        "num_candidates": None,
        "candidate_purity": None,
        "motif_recall_all": None,
        "motif_type_coverage": None,
        "candidate_type_distribution": None,
        "avg_candidate_size": None,
        "runtime": None,
        "source": "N/A",
        "notes": reason,
    }


def prefixed_node(partition_id: str, account_id: str) -> str:
    return f"{partition_id}::{account_id}"


def unique_sorted(values: Iterable[Any]) -> List[str]:
    return sorted({str(value) for value in values if str(value) not in ("", "nan", "None")})


def rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def write_candidate_artifact(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_pair_candidates_from_alignment(alignment_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with alignment_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            tx_ids = unique_sorted([row.get("tx_id_s"), row.get("tx_id_r")])
            nodes = unique_sorted([row.get("evidence_node_id_s"), row.get("evidence_node_id_r")])
            score = float(row.get("compat_score") or 0.0)
            hypothesis = str(row.get("best_hypothesis") or "PAIR").upper()
            rows.append({
                "candidate_id": f"NO_COMPAT_{idx:06d}",
                "candidate_type": f"NO_COMPAT_{hypothesis}",
                "num_nodes": len(nodes),
                "num_edges": len(tx_ids),
                "score_mean": score,
                "score_min": score,
                "nodes_json": json.dumps(nodes),
                "tx_ids_json": json.dumps(tx_ids),
            })
    return rows_to_dataframe(rows)


def build_pair_candidates_from_compatibility(compat_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with compat_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            tx_ids = unique_sorted([row.get("tx_id_sender"), row.get("tx_id_receiver")])
            nodes = unique_sorted([row.get("src_node"), row.get("dst_node")])
            score = float(row.get("weight") or 0.0)
            hypothesis = str(row.get("best_hypothesis") or "PAIR").upper()
            rows.append({
                "candidate_id": f"NO_STRUCT_{idx:06d}",
                "candidate_type": f"NO_STRUCT_{hypothesis}",
                "num_nodes": len(nodes),
                "num_edges": len(tx_ids),
                "score_mean": score,
                "score_min": score,
                "nodes_json": json.dumps(nodes),
                "tx_ids_json": json.dumps(tx_ids),
            })
    return rows_to_dataframe(rows)


def load_anchor_nodes(method_queries_path: Path) -> Set[str]:
    with method_queries_path.open(encoding="utf-8") as f:
        queries = json.load(f)
    anchors: Set[str] = set()
    for query in queries:
        anchors.update(str(node_id) for node_id in query.get("anchor_nodes", []))
    return anchors


def build_anchor_only_candidates(
    transactions_path: Path,
    method_queries_path: Path,
    max_edges_per_anchor: int = 5,
) -> pd.DataFrame:
    anchors = load_anchor_nodes(method_queries_path)
    incident: Dict[str, List[Dict[str, Any]]] = {anchor: [] for anchor in anchors}
    with transactions_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = str(row["Account"])
            dst = str(row["Account.1"])
            touched = []
            if src in incident:
                touched.append(src)
            if dst in incident and dst != src:
                touched.append(dst)
            for anchor in touched:
                incident[anchor].append(row)

    rows: List[Dict[str, Any]] = []
    for anchor in sorted(anchors):
        tx_rows = incident.get(anchor, [])
        if not tx_rows:
            continue
        tx_rows = sorted(
            tx_rows,
            key=lambda r: (
                0 if str(r.get("is_inter_partition")) == "True" else 1,
                int(r["tx_id"]),
            ),
        )[:max_edges_per_anchor]
        nodes: Set[str] = set()
        tx_ids: List[str] = []
        for tx in tx_rows:
            nodes.add(prefixed_node(str(tx["src_bank_partition"]), str(tx["Account"])))
            nodes.add(prefixed_node(str(tx["dst_bank_partition"]), str(tx["Account.1"])))
            tx_ids.append(str(tx["tx_id"]))
        rows.append({
            "candidate_id": f"ANCHOR_{len(rows):06d}",
            "candidate_type": "ANCHOR_ONLY",
            "num_nodes": len(nodes),
            "num_edges": len(set(tx_ids)),
            "score_mean": 0.0,
            "score_min": 0.0,
            "nodes_json": json.dumps(sorted(nodes)),
            "tx_ids_json": json.dumps(unique_sorted(tx_ids)),
        })
    return rows_to_dataframe(rows)


def build_random_cross_partition_candidates(
    transactions_path: Path,
    target_count: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    sample: List[Dict[str, Any]] = []
    seen = 0
    with transactions_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("is_inter_partition")) != "True":
                continue
            seen += 1
            if len(sample) < target_count:
                sample.append(row)
            else:
                j = rng.randrange(seen)
                if j < target_count:
                    sample[j] = row

    sample = sorted(sample, key=lambda r: int(r["tx_id"]))
    rows: List[Dict[str, Any]] = []
    for idx, tx in enumerate(sample):
        nodes = [
            prefixed_node(str(tx["src_bank_partition"]), str(tx["Account"])),
            prefixed_node(str(tx["dst_bank_partition"]), str(tx["Account.1"])),
        ]
        rows.append({
            "candidate_id": f"RANDOM_XPART_{idx:06d}",
            "candidate_type": "RANDOM_CROSS_PARTITION",
            "num_nodes": len(set(nodes)),
            "num_edges": 1,
            "score_mean": 0.0,
            "score_min": 0.0,
            "nodes_json": json.dumps(sorted(set(nodes))),
            "tx_ids_json": json.dumps([str(tx["tx_id"])]),
        })
    return rows_to_dataframe(rows)


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
        ("generation_variant", "Variant"),
        ("status", "Status"),
        ("num_candidates", "Candidates"),
        ("candidate_purity", "Purity"),
        ("motif_recall_all", "Recall"),
        ("motif_type_coverage", "MotifCov"),
        ("avg_candidate_size", "Avg. size"),
        ("runtime", "Runtime"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        " & ".join(label for _, label in columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(key)) for key, _ in columns) + r" \\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{AMLWorld controlled-balanced candidate generation study.}",
        r"\label{tab:amlworld_candidate_generation}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AMLWorld candidate-generation study summary.")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--summary_dir", default="outputs/summary")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = PROJECT_ROOT / args.output_root / DATASET / METHOD / f"seed_{args.seed}"
    candidate_dir = run_dir / f"semotif_candidates_{TAG}"
    partition_dir = run_dir / f"semotif_partitioned_{TAG}"
    orchestrator_dir = run_dir / f"semotif_orchestrator_{TAG}"
    runtime_path = run_dir / "runtime.json"
    data_dir = PROJECT_ROOT / "data" / "processed" / DATASET
    summary_dir = PROJECT_ROOT / args.summary_dir
    variant_dir = summary_dir / "candidate_generation_variants_amlworld"

    candidates = load_candidates(candidate_dir)
    gt_df, motif_to_txs, motif_to_type, tx_to_motifs = load_gt(partition_dir)
    runtime = load_json(runtime_path).get("duration_sec") if runtime_path.exists() else None

    rows = []
    dist_rows = []
    for variant in VARIANTS:
        if variant in ("full_generation", "raw_candidate_pool"):
            row, distribution = evaluate_candidate_pool(
                variant=variant,
                df=candidates,
                gt_df=gt_df,
                tx_to_motifs=tx_to_motifs,
                motif_to_type=motif_to_type,
                source=rel(candidate_dir),
                runtime=runtime if variant == "full_generation" else None,
                notes="Computed from existing candidate files; no new main experiment was run.",
            )
            if variant == "raw_candidate_pool":
                row["runtime"] = None
                row["notes"] = "Existing pre-rerank candidate pool; runtime is not separately logged."
            rows.append(row)
            dist_rows.extend(distribution)
        elif variant == "no_compatibility_filtering":
            source_path = orchestrator_dir / "alignment_pairs_scored.csv"
            if not source_path.exists():
                rows.append(not_available_row(
                    variant,
                    "Pre-filter alignment_pairs_scored.csv was not found; this variant cannot be recovered without rerunning the orchestrator with saved pre-filter pairs.",
                ))
                continue
            start = time.perf_counter()
            df = build_pair_candidates_from_alignment(source_path)
            gen_runtime = time.perf_counter() - start
            artifact = variant_dir / f"{variant}_candidates.csv"
            write_candidate_artifact(df, artifact)
            row, distribution = evaluate_candidate_pool(
                variant=variant,
                df=df,
                gt_df=gt_df,
                tx_to_motifs=tx_to_motifs,
                motif_to_type=motif_to_type,
                source=rel(source_path),
                runtime=gen_runtime,
                notes="Generated offline from pre-filter alignment_pairs_scored.csv without using ground truth during generation.",
            )
            row["source"] = f"{rel(source_path)}; candidates={rel(artifact)}"
            rows.append(row)
            dist_rows.extend(distribution)
        elif variant == "no_structural_filtering":
            source_path = orchestrator_dir / "compatibility_graph_edges.csv"
            if not source_path.exists():
                rows.append(not_available_row(
                    variant,
                    "Retained compatibility_graph_edges.csv was not found; this variant cannot be recovered.",
                ))
                continue
            start = time.perf_counter()
            df = build_pair_candidates_from_compatibility(source_path)
            gen_runtime = time.perf_counter() - start
            artifact = variant_dir / f"{variant}_candidates.csv"
            write_candidate_artifact(df, artifact)
            row, distribution = evaluate_candidate_pool(
                variant=variant,
                df=df,
                gt_df=gt_df,
                tx_to_motifs=tx_to_motifs,
                motif_to_type=motif_to_type,
                source=rel(source_path),
                runtime=gen_runtime,
                notes="Generated offline from retained compatibility graph edges without chain/fan/cycle candidate constraints.",
            )
            row["source"] = f"{rel(source_path)}; candidates={rel(artifact)}"
            rows.append(row)
            dist_rows.extend(distribution)
        elif variant == "anchor_only_generation":
            source_path = partition_dir / "transactions_partitioned.csv"
            method_queries_path = data_dir / "method_queries.json"
            if not source_path.exists() or not method_queries_path.exists():
                rows.append(not_available_row(
                    variant,
                    "transactions_partitioned.csv or method_queries.json was not found; anchor-only candidates cannot be generated.",
                ))
                continue
            start = time.perf_counter()
            df = build_anchor_only_candidates(source_path, method_queries_path)
            gen_runtime = time.perf_counter() - start
            artifact = variant_dir / f"{variant}_candidates.csv"
            write_candidate_artifact(df, artifact)
            row, distribution = evaluate_candidate_pool(
                variant=variant,
                df=df,
                gt_df=gt_df,
                tx_to_motifs=tx_to_motifs,
                motif_to_type=motif_to_type,
                source=f"{rel(source_path)}; {rel(method_queries_path)}",
                runtime=gen_runtime,
                notes="Generated from method-facing anchor_nodes only; hidden_nodes/hidden_edges and compatibility graph are not read during generation.",
            )
            row["source"] = f"{rel(source_path)}; {rel(method_queries_path)}; candidates={rel(artifact)}"
            rows.append(row)
            dist_rows.extend(distribution)
        elif variant == "random_cross_partition_generation":
            source_path = partition_dir / "transactions_partitioned.csv"
            if not source_path.exists():
                rows.append(not_available_row(
                    variant,
                    "transactions_partitioned.csv was not found; random cross-partition candidates cannot be generated.",
                ))
                continue
            start = time.perf_counter()
            df = build_random_cross_partition_candidates(source_path, target_count=len(candidates), seed=args.seed)
            gen_runtime = time.perf_counter() - start
            artifact = variant_dir / f"{variant}_candidates.csv"
            write_candidate_artifact(df, artifact)
            row, distribution = evaluate_candidate_pool(
                variant=variant,
                df=df,
                gt_df=gt_df,
                tx_to_motifs=tx_to_motifs,
                motif_to_type=motif_to_type,
                source=rel(source_path),
                runtime=gen_runtime,
                notes="Generated with fixed-seed reservoir sampling from visible cross-partition transactions; target count matches full_generation.",
            )
            row["source"] = f"{rel(source_path)}; candidates={rel(artifact)}"
            rows.append(row)
            dist_rows.extend(distribution)
        else:
            rows.append(not_available_row(variant, "Variant is not defined in this study script."))

    study_csv = summary_dir / "candidate_assembly_amlworld.csv"
    latex_path = summary_dir / "candidate_assembly_amlworld_latex.tex"
    dist_csv = summary_dir / "candidate_type_distribution_amlworld.csv"
    write_csv(rows, study_csv, STUDY_COLUMNS)
    write_latex(rows, latex_path)
    write_csv(dist_rows, dist_csv, DIST_COLUMNS)

    print(json.dumps({
        "completed_variants": [row["generation_variant"] for row in rows if row["status"] == "completed"],
        "not_available": [row["generation_variant"] for row in rows if row["status"] != "completed"],
        "outputs": [rel(study_csv), rel(latex_path), rel(dist_csv)],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
