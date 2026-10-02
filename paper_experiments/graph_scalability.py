#!/usr/bin/env python3
"""Measure CoMot on the three official AMLWorld graph sizes."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "amlworld"
OUT = ROOT / "outputs" / "paper" / "graph_scale_scalability"
PYTHON = Path(sys.executable)
SCALES = ("Small", "Medium", "Large")
SEEDS = (0, 1, 2)
TAG = "five_party"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], *, time_file: Path | None = None) -> tuple[float, int | None]:
    actual = command
    if time_file is not None:
        time_file.parent.mkdir(parents=True, exist_ok=True)
        actual = ["/usr/bin/time", "-v", "-o", str(time_file), *command]
    print("COMMAND " + " ".join(map(str, actual)), flush=True)
    started = time.perf_counter()
    proc = subprocess.run(actual, cwd=ROOT)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"command exited {proc.returncode}: {' '.join(map(str, command))}")
    peak = parse_peak(time_file) if time_file else None
    return elapsed, peak


def parse_peak(path: Path) -> int:
    prefix = "Maximum resident set size (kbytes):"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(prefix):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"GNU time peak RSS missing: {path}")


def line_records(path: Path) -> int:
    lines = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            lines += block.count(b"\n")
    return max(0, lines - 1)


def global_dir(scale: str) -> Path:
    return OUT / "work" / scale.lower() / "global"


def partition_dir(scale: str, seed: int) -> Path:
    return OUT / "work" / scale.lower() / f"seed_{seed}" / f"semotif_partitioned_{TAG}"


def method_dir(scale: str, seed: int) -> Path:
    return OUT / "raw" / scale.lower() / f"seed_{seed}"


def prepare_global(scale: str) -> dict:
    dest = global_dir(scale)
    summary = dest / "global_summary.json"
    if not summary.exists():
        trans = RAW / f"HI-{scale}_Trans.csv"
        patterns = RAW / f"HI-{scale}_Patterns.txt"
        run([str(PYTHON), str(ROOT / "src" / "data" / "amlworld_preprocess.py"),
             "--trans_path", str(trans), "--pattern_path", str(patterns), "--output_dir", str(dest)])
    result = read_json(summary)
    print(f"GLOBAL READY scale={scale} nodes={result['num_accounts']} edges={result['num_transactions']}", flush=True)
    return result


def prepare_partition(scale: str, seed: int) -> Path:
    dest = partition_dir(scale, seed)
    summary = dest / "partition_summary.json"
    if summary.exists():
        data = read_json(summary)
        if int(data.get("seed", -1)) != seed or int(data.get("num_banks", -1)) != 5:
            raise RuntimeError(f"partition metadata does not match the requested run: {summary}")
        print(f"PARTITION READY scale={scale} seed={seed} path={dest}", flush=True)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    run([str(PYTHON), str(ROOT / "src" / "data" / "amlworld_partition.py"),
         "--global_dir", str(global_dir(scale)), "--output_dir", str(dest),
         "--num_banks", "5", "--min_banks_per_motif", "2", "--clean_inter_ratio", "0.03",
         "--time_mode", "minute", "--seed", str(seed), "--keep_unknown_laundering"])
    if not summary.exists():
        raise RuntimeError(f"partition did not produce {summary}")
    return dest


def preliminary_seed0_table() -> None:
    fields = ["Scale", "#Nodes", "#Edges", "Runtime (s)", "Peak Memory (GB)",
              "Data Volume (MB)", "#Evidence Records", "#Compatibility Relations", "#Candidates"]
    rows = []
    for scale in SCALES:
        path = method_dir(scale, 0) / "result.json"
        if not path.exists():
            continue
        r = read_json(path)
        rows.append({"Scale": "HI-" + scale, "#Nodes": r["nodes"], "#Edges": r["edges"],
                     "Runtime (s)": f"{r['runtime_s']:.2f}",
                     "Peak Memory (GB)": f"{r['peak_memory_gb']:.3f}",
                     "Data Volume (MB)": f"{r['serialized_data_volume_mb']:.2f}",
                     "#Evidence Records": r["evidence_records"],
                     "#Compatibility Relations": r["compatibility_relations"],
                     "#Candidates": r["candidates"]})
    with (OUT / "preliminary_seed0_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    lines += ["| " + " | ".join(str(r[k]) for k in fields) + " |" for r in rows]
    (OUT / "preliminary_seed0_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def measure(scale: str, seed: int, graph: dict) -> dict:
    part = prepare_partition(scale, seed)
    base = method_dir(scale, seed)
    evidence = base / f"semotif_local_evidences_{TAG}"
    orchestrator = base / f"semotif_orchestrator_{TAG}"
    candidates = base / f"semotif_candidates_{TAG}"
    reranked = base / f"semotif_candidates_{TAG}_reranked"
    result_path = base / "result.json"
    if result_path.exists() and read_json(result_path).get("status") == "COMPLETE":
        print(f"MEASUREMENT READY scale={scale} seed={seed}", flush=True)
        return read_json(result_path)
    for path in (evidence, orchestrator, candidates, reranked):
        if path.exists(): shutil.rmtree(path)
    stages = [
        ("local_evidence", [str(PYTHON), str(ROOT / "src" / "methods" / "comot_pipeline" / "local_encoder.py"),
          "--partition_dir", str(part), "--output_dir", str(evidence), "--k_hop", "2", "--num_workers", "1"]),
        ("compatibility", [str(PYTHON), str(ROOT / "src" / "methods/comot_pipeline/orchestrator.py"),
          "--evidence_dir", str(evidence), "--output_dir", str(orchestrator), "--threshold", "0.72",
          "--max_group_size", "5", "--topk_per_sender", "1", "--topk_per_receiver", "1"]),
        ("candidate_assembly", [str(PYTHON), str(ROOT / "src" / "methods" / "comot_pipeline" / "candidate_builder.py"),
          "--orchestrator_dir", str(orchestrator), "--output_dir", str(candidates),
          "--chain_min_len", "3", "--chain_max_len", "10", "--cycle_min_len", "2",
          "--cycle_max_len", "8", "--fan_degree_thr", "3"]),
        ("prioritization", [str(PYTHON), str(ROOT / "src" / "methods/comot_pipeline/candidate_reranker.py"),
          "--candidate_dir", str(candidates), "--output_dir", str(reranked)]),
    ]
    stage_rows = []
    for name, command in stages:
        print(f"STAGE START scale={scale} seed={seed} stage={name}", flush=True)
        elapsed, peak = run(command, time_file=base / "timing" / f"{name}.time")
        stage_rows.append({"stage": name, "wall_seconds": elapsed, "peak_memory_kib": peak})
        print(f"STAGE COMPLETE scale={scale} seed={seed} stage={name} wall={elapsed:.3f} peak_kib={peak}", flush=True)
    evidence_files = sorted(evidence.glob("BANK_*_evidences.csv"))
    if len(evidence_files) != 5:
        raise RuntimeError(f"expected five evidence files, found {len(evidence_files)}")
    evidence_count = sum(line_records(p) for p in evidence_files)
    evidence_bytes = sum(p.stat().st_size for p in evidence_files)
    ostats = read_json(orchestrator / "orchestrator_stats.json")
    cstats = read_json(candidates / "candidate_summary.json")
    candidate_count = sum(int(v) for k, v in cstats.items() if k.startswith("num_") and k.endswith("_candidates"))
    result = {
        "status": "COMPLETE", "scale": scale, "seed": seed,
        "nodes": int(graph["num_accounts"]), "edges": int(graph["num_transactions"]),
        "runtime_s": sum(x["wall_seconds"] for x in stage_rows),
        "peak_memory_gb": max(x["peak_memory_kib"] for x in stage_rows) * 1024 / 1e9,
        "serialized_data_volume_mb": evidence_bytes / 1e6,
        "evidence_records": evidence_count,
        "compatibility_relations": int(ostats["kept_edges"]),
        "compatibility_pairs_scored": int(ostats["total_scored_pairs"]),
        "candidates": candidate_count,
        "stages": stage_rows,
        "definitions": {
            "nodes": "distinct Account IDs in the AMLWorld preprocessing output",
            "edges": "all transaction CSV rows consumed by the pipeline, including self-loops",
            "local_encoder": "src/methods/comot_pipeline/local_encoder.py",
            "cycle_enumeration": "NetworkX simple_cycles uses length_bound=8",
            "runtime": "sum of local evidence, compatibility, candidate assembly, and prioritization wall times",
            "peak_memory": "maximum GNU time max RSS over the same four isolated stage processes; decimal GB",
            "serialized_data_volume": "total bytes of five BANK_*_evidences.csv files; decimal MB",
            "evidence_records": "data rows in the five evidence CSV files",
            "compatibility_relations": "orchestrator kept_edges after threshold and bidirectional Top-K",
            "candidates": "chain + fan + cycle candidate rows before ranking",
        },
    }
    write_json(result_path, result)
    # Retain compact evidence for audit; remove large row artifacts after metrics are sealed.
    write_json(base / "orchestrator_stats.json", ostats)
    write_json(base / "candidate_summary.json", cstats)
    for path in (evidence, orchestrator, candidates, reranked):
        shutil.rmtree(path)
    print(f"MEASUREMENT COMPLETE scale={scale} seed={seed} result={result_path}", flush=True)
    return result


def aggregate() -> None:
    rows = []
    for scale in SCALES:
        for seed in SEEDS:
            p = method_dir(scale, seed) / "result.json"
            if p.exists(): rows.append(read_json(p))
    fields = ["scale", "seed", "status", "nodes", "edges", "runtime_s", "peak_memory_gb",
              "serialized_data_volume_mb", "evidence_records", "compatibility_relations", "candidates"]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in fields})
    numeric = ["runtime_s", "peak_memory_gb", "serialized_data_volume_mb", "evidence_records",
               "compatibility_relations", "candidates"]
    summary = []
    for scale in SCALES:
        group = [r for r in rows if r["scale"] == scale and r["status"] == "COMPLETE"]
        item = {"scale": scale, "completed_seeds": len(group),
                "nodes": group[0]["nodes"] if group else "", "edges": group[0]["edges"] if group else ""}
        for key in numeric:
            values = [float(r[key]) for r in group]
            item[key + "_mean"] = statistics.mean(values) if values else ""
            item[key + "_sample_std"] = statistics.stdev(values) if len(values) >= 2 else ""
        summary.append(item)
    sf = list(summary[0])
    with (OUT / "summary_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sf); w.writeheader(); w.writerows(summary)
    final = []
    for r in summary:
        def pm(key, digits):
            a, b = r[key + "_mean"], r[key + "_sample_std"]
            return "" if a == "" or b == "" else f"{a:.{digits}f} +/- {b:.{digits}f}"
        final.append({"Scale": "HI-" + r["scale"], "#Nodes": r["nodes"], "#Edges": r["edges"],
                      "Runtime (s)": pm("runtime_s", 2), "Peak Memory (GB)": pm("peak_memory_gb", 3),
                      "Data Volume (MB)": pm("serialized_data_volume_mb", 2),
                      "#Evidence Records": pm("evidence_records", 1),
                      "#Compatibility Relations": pm("compatibility_relations", 1),
                      "#Candidates": pm("candidates", 1)})
    ff = list(final[0])
    with (OUT / "final_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ff); w.writeheader(); w.writerows(final)
    lines = ["| " + " | ".join(ff) + " |", "|" + "|".join(["---"] * len(ff)) + "|"]
    lines += ["| " + " | ".join(str(r[k]) for k in ff) + " |" for r in final]
    (OUT / "final_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def full_run() -> None:
    failures = []
    for scale in SCALES:
        graph = prepare_global(scale)
        for seed in SEEDS:
            try:
                measure(scale, seed, graph)
            except BaseException as exc:
                failures.append({"scale": scale, "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
                print(f"CELL FAILED scale={scale} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
            finally:
                aggregate()
                if scale != "Small":
                    part = partition_dir(scale, seed)
                    if part.exists(): shutil.rmtree(part)
        if scale != "Small" and global_dir(scale).exists():
            shutil.rmtree(global_dir(scale))
    write_json(OUT / "failures.json", failures)
    aggregate()
    print(f"ALL DONE failures={len(failures)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["run", "aggregate"])
    args = parser.parse_args()
    if args.command == "run": full_run()
    else: aggregate()


if __name__ == "__main__": main()
