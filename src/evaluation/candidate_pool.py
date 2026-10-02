import os
import json
import argparse

import pandas as pd


def load_json_list(s):
    if pd.isna(s):
        return []
    return json.loads(s)


def load_candidates(candidate_dir: str) -> pd.DataFrame:
    dfs = []
    for fn in ["chain_candidates.csv", "fan_candidates.csv", "cycle_candidates.csv"]:
        fp = os.path.join(candidate_dir, fn)
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No candidate files found in {candidate_dir}")
    all_df = pd.concat(dfs, ignore_index=True)
    if len(all_df) == 0:
        return all_df

    # Normalize fields used for candidate ordering.
    if "score_mean" not in all_df.columns:
        all_df["score_mean"] = 0.0
    if "score_min" not in all_df.columns:
        all_df["score_min"] = 0.0
    return all_df


def load_gt_motifs(partition_dir: str) -> pd.DataFrame:
    fp = os.path.join(partition_dir, "ground_truth_motifs.csv")
    return pd.read_csv(fp)


def evaluate_candidates(candidate_df: pd.DataFrame, gt_df: pd.DataFrame):
    if len(candidate_df) == 0:
        return {
            "num_candidates": 0,
            "num_hit_candidates": 0,
            "candidate_purity": 0.0,
            "num_gt_motifs": int(len(gt_df)),
            "num_gt_hit": 0,
            "motif_recall_all": 0.0,
            "topk_results": []
        }, pd.DataFrame()

    gt_motif_to_txs = {
        row["motif_instance_id"]: set(load_json_list(row["tx_ids_json"]))
        for _, row in gt_df.iterrows()
    }
    detail_rows = []
    gt_hit_set = set()

    for _, row in candidate_df.iterrows():
        cand_txs = set(load_json_list(row["tx_ids_json"]))
        hits = []
        for motif_id, motif_txs in gt_motif_to_txs.items():
            if len(cand_txs & motif_txs) > 0:
                hits.append(motif_id)

        if hits:
            gt_hit_set.update(hits)

        detail_rows.append({
            "candidate_id": row["candidate_id"],
            "candidate_type": row["candidate_type"],
            "score_mean": row["score_mean"],
            "score_min": row["score_min"],
            "num_nodes": row["num_nodes"],
            "num_edges": row["num_edges"],
            "num_hit_gt_motifs": len(hits),
            "is_hit": 1 if len(hits) > 0 else 0,
            "hit_gt_motif_ids_json": json.dumps(hits),
        })

    detail_df = pd.DataFrame(detail_rows).sort_values(
        ["score_mean", "score_min", "num_edges"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    num_candidates = len(detail_df)
    num_hit_candidates = int(detail_df["is_hit"].sum())
    candidate_purity = num_hit_candidates / num_candidates if num_candidates > 0 else 0.0
    motif_recall_all = len(gt_hit_set) / len(gt_df) if len(gt_df) > 0 else 0.0

    topks = [10, 20, 50, 100, 200, 500, 1000]
    topk_results = []
    for k in topks:
        if k > len(detail_df):
            continue
        sub = detail_df.iloc[:k]
        purity = float(sub["is_hit"].mean()) if len(sub) > 0 else 0.0

        gt_hit_topk = set()
        for _, r in sub.iterrows():
            gt_hit_topk.update(load_json_list(r["hit_gt_motif_ids_json"]))
        recall_topk = len(gt_hit_topk) / len(gt_df) if len(gt_df) > 0 else 0.0

        topk_results.append({
            "top_k": int(k),
            "candidate_purity": purity,
            "num_hit_candidates": int(sub["is_hit"].sum()),
            "num_gt_motifs_hit": int(len(gt_hit_topk)),
            "motif_recall_at_k": recall_topk,
        })

    summary = {
        "num_candidates": int(num_candidates),
        "num_hit_candidates": int(num_hit_candidates),
        "candidate_purity": float(candidate_purity),
        "num_gt_motifs": int(len(gt_df)),
        "num_gt_hit": int(len(gt_hit_set)),
        "motif_recall_all": float(motif_recall_all),
        "topk_results": topk_results,
    }

    return summary, detail_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_dir", type=str, required=True)
    parser.add_argument("--partition_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="candidate_eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    candidate_df = load_candidates(args.candidate_dir)
    gt_df = load_gt_motifs(args.partition_dir)

    summary, detail_df = evaluate_candidates(candidate_df, gt_df)

    detail_df.to_csv(os.path.join(args.output_dir, "candidate_eval_detail.csv"), index=False)
    with open(os.path.join(args.output_dir, "candidate_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
