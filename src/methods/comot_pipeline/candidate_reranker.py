import os
import json
import math
import argparse
import pandas as pd


def safe_log(x: float) -> float:
    return math.log(max(x, 1e-8))


def candidate_type_bonus(cand_type: str) -> float:
    """Apply the submitted type prior, reducing the dominant fan candidate weight."""
    cand_type = str(cand_type).upper()
    if cand_type == "CHAIN":
        return 0.15
    if cand_type == "CYCLE":
        return 0.12
    if cand_type in ("FAN_IN", "FAN_OUT"):
        return -0.08
    return 0.0


def size_penalty(cand_type: str, num_nodes: int, num_edges: int) -> float:
    """Apply stricter size penalties to fans than to chains or cycles."""
    cand_type = str(cand_type).upper()

    if cand_type in ("FAN_IN", "FAN_OUT"):
        # Large fan candidates are more likely to be noisy.
        return 0.03 * max(num_nodes - 6, 0) + 0.03 * max(num_edges - 5, 0)

    if cand_type == "CHAIN":
        # Very short chains lack evidence; very long chains may contain assembly noise.
        short_pen = 0.08 if num_edges < 2 else 0.0
        long_pen = 0.02 * max(num_edges - 6, 0)
        return short_pen + long_pen

    if cand_type == "CYCLE":
        # Prefer small and medium closed cycles.
        return 0.025 * max(num_nodes - 5, 0)

    return 0.0


def compactness_bonus(cand_type: str, num_nodes: int, num_edges: int) -> float:
    """Reward compact candidates whose density matches the template."""
    cand_type = str(cand_type).upper()
    if num_nodes <= 0:
        return 0.0

    density = num_edges / max(num_nodes, 1)

    if cand_type == "CHAIN":
        # A chain has approximately num_nodes - 1 edges.
        target = (num_nodes - 1) / max(num_nodes, 1)
        return -abs(density - target)

    if cand_type == "CYCLE":
        # A cycle has approximately num_nodes edges.
        target = 1.0
        return -abs(density - target)

    if cand_type in ("FAN_IN", "FAN_OUT"):
        # A fan should resemble one center with several leaves.
        target = (num_nodes - 1) / max(num_nodes, 1)
        return -0.5 * abs(density - target)

    return 0.0


def rerank_score(row):
    cand_type = str(row["candidate_type"])
    num_nodes = int(row["num_nodes"])
    num_edges = int(row["num_edges"])
    score_mean = float(row["score_mean"])
    score_min = float(row["score_min"])

    # Base term combines mean and minimum compatibility.
    base = 0.60 * score_mean + 0.40 * score_min

    # A smaller mean/min gap indicates greater structural stability.
    stability = -abs(score_mean - score_min)

    # Candidate-type prior.
    type_term = candidate_type_bonus(cand_type)

    # Size penalty.
    penalty = size_penalty(cand_type, num_nodes, num_edges)

    # Compactness reward.
    compact = compactness_bonus(cand_type, num_nodes, num_edges)

    # Slight preference for nontrivial chains and cycles.
    richness = 0.0
    cand_type_u = cand_type.upper()
    if cand_type_u == "CHAIN":
        richness = 0.03 * min(max(num_edges - 2, 0), 4)
    elif cand_type_u == "CYCLE":
        richness = 0.03 * min(max(num_nodes - 2, 0), 3)
    elif cand_type_u in ("FAN_IN", "FAN_OUT"):
        richness = 0.01 * min(max(num_edges - 2, 0), 2)

    final_score = (
        base
        + 0.20 * stability
        + type_term
        + 0.10 * compact
        + richness
        - penalty
    )
    return final_score


def load_candidates(candidate_dir: str) -> pd.DataFrame:
    dfs = []
    for fn in ["chain_candidates.csv", "fan_candidates.csv", "cycle_candidates.csv"]:
        fp = os.path.join(candidate_dir, fn)
        if os.path.exists(fp):
            dfs.append(pd.read_csv(fp))
    if not dfs:
        raise FileNotFoundError(f"No candidate files found in {candidate_dir}")
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="reranked_candidates")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_candidates(args.candidate_dir).copy()
    if len(df) == 0:
        out_fp = os.path.join(args.output_dir, "candidates_reranked.csv")
        pd.DataFrame().to_csv(out_fp, index=False)
        print({"num_candidates": 0})
        return

    # Validate required fields.
    for col in ["candidate_type", "num_nodes", "num_edges", "score_mean", "score_min"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["rerank_score"] = df.apply(rerank_score, axis=1)

    # Write reranked candidates.
    reranked_df = df.sort_values(
        ["rerank_score", "score_mean", "score_min", "num_edges"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    out_fp = os.path.join(args.output_dir, "candidates_reranked.csv")
    reranked_df.to_csv(out_fp, index=False)

    summary = {
        "num_candidates": int(len(reranked_df)),
        "candidate_type_counts": {
            str(k): int(v) for k, v in reranked_df["candidate_type"].value_counts().to_dict().items()
        },
        "output_file": out_fp,
    }

    with open(os.path.join(args.output_dir, "rerank_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
