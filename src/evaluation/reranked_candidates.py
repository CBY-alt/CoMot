import os
import json
import argparse
import pandas as pd


def load_json_list(s):
    if pd.isna(s):
        return []
    return json.loads(s)


def evaluate_reranked(reranked_df: pd.DataFrame, gt_df: pd.DataFrame):
    gt_motif_to_txs = {
        row["motif_instance_id"]: set(load_json_list(row["tx_ids_json"]))
        for _, row in gt_df.iterrows()
    }

    detail_rows = []
    gt_hit_all = set()

    for _, row in reranked_df.iterrows():
        cand_txs = set(load_json_list(row["tx_ids_json"]))
        hits = []

        for motif_id, motif_txs in gt_motif_to_txs.items():
            if len(cand_txs & motif_txs) > 0:
                hits.append(motif_id)

        if hits:
            gt_hit_all.update(hits)

        detail_rows.append({
            "candidate_id": row["candidate_id"],
            "candidate_type": row["candidate_type"],
            "rerank_score": float(row["rerank_score"]),
            "score_mean": float(row["score_mean"]),
            "score_min": float(row["score_min"]),
            "num_nodes": int(row["num_nodes"]),
            "num_edges": int(row["num_edges"]),
            "is_hit": 1 if len(hits) > 0 else 0,
            "num_hit_gt_motifs": len(hits),
            "hit_gt_motif_ids_json": json.dumps(hits),
        })

    detail_df = pd.DataFrame(detail_rows)

    summary = {
        "num_candidates": int(len(detail_df)),
        "num_hit_candidates": int(detail_df["is_hit"].sum()),
        "candidate_purity": float(detail_df["is_hit"].mean()) if len(detail_df) > 0 else 0.0,
        "num_gt_motifs": int(len(gt_df)),
        "num_gt_hit": int(len(gt_hit_all)),
        "motif_recall_all": float(len(gt_hit_all) / len(gt_df)) if len(gt_df) > 0 else 0.0,
        "topk_results": [],
    }

    topks = [10, 20, 50, 100, 200, 500, 1000]
    for k in topks:
        if k > len(detail_df):
            continue
        sub = detail_df.iloc[:k]

        gt_hit_topk = set()
        for _, r in sub.iterrows():
            gt_hit_topk.update(load_json_list(r["hit_gt_motif_ids_json"]))

        summary["topk_results"].append({
            "top_k": int(k),
            "candidate_purity": float(sub["is_hit"].mean()) if len(sub) > 0 else 0.0,
            "num_hit_candidates": int(sub["is_hit"].sum()),
            "num_gt_motifs_hit": int(len(gt_hit_topk)),
            "motif_recall_at_k": float(len(gt_hit_topk) / len(gt_df)) if len(gt_df) > 0 else 0.0,
        })

    return summary, detail_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranked_csv", type=str, required=True)
    parser.add_argument("--partition_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="reranked_eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    reranked_df = pd.read_csv(args.reranked_csv)
    gt_df = pd.read_csv(os.path.join(args.partition_dir, "ground_truth_motifs.csv"))

    summary, detail_df = evaluate_reranked(reranked_df, gt_df)

    detail_df.to_csv(os.path.join(args.output_dir, "reranked_eval_detail.csv"), index=False)
    with open(os.path.join(args.output_dir, "reranked_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()