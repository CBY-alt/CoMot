import os
import json
import argparse
from collections import Counter, defaultdict

import pandas as pd


def load_json_list(s):
    if pd.isna(s):
        return []
    return json.loads(s)


def invert_gt_mapping(gt_df: pd.DataFrame):
    motif_id_to_type = {}
    for _, row in gt_df.iterrows():
        motif_id_to_type[str(row["motif_instance_id"])] = str(row["motif_type"])
    return motif_id_to_type


def summarize_topk(detail_df: pd.DataFrame, motif_id_to_type: dict, ks=(10, 20, 50, 100, 200, 500, 1000)):
    results = []

    for k in ks:
        if k > len(detail_df):
            continue

        sub = detail_df.iloc[:k].copy()

        candidate_type_counter = Counter(sub["candidate_type"].astype(str).tolist())
        hit_candidate_type_counter = Counter(sub[sub["is_hit"] == 1]["candidate_type"].astype(str).tolist())

        gt_hit_ids = []
        for _, row in sub.iterrows():
            gt_hit_ids.extend(load_json_list(row["hit_gt_motif_ids_json"]))

        gt_hit_ids = list(set(gt_hit_ids))
        gt_hit_type_counter = Counter([motif_id_to_type.get(str(mid), "UNKNOWN") for mid in gt_hit_ids])

        results.append({
            "top_k": int(k),
            "num_candidates": int(len(sub)),
            "num_hit_candidates": int(sub["is_hit"].sum()),
            "candidate_type_counts": dict(candidate_type_counter),
            "hit_candidate_type_counts": dict(hit_candidate_type_counter),
            "num_gt_motifs_hit": int(len(gt_hit_ids)),
            "gt_motif_type_counts": dict(gt_hit_type_counter),
        })

    return results


def overall_breakdown(detail_df: pd.DataFrame, motif_id_to_type: dict):
    all_candidate_type_counter = Counter(detail_df["candidate_type"].astype(str).tolist())
    hit_candidate_type_counter = Counter(detail_df[detail_df["is_hit"] == 1]["candidate_type"].astype(str).tolist())

    all_hit_ids = []
    for _, row in detail_df.iterrows():
        all_hit_ids.extend(load_json_list(row["hit_gt_motif_ids_json"]))
    all_hit_ids = list(set(all_hit_ids))

    gt_hit_type_counter = Counter([motif_id_to_type.get(str(mid), "UNKNOWN") for mid in all_hit_ids])

    return {
        "all_candidate_type_counts": dict(all_candidate_type_counter),
        "all_hit_candidate_type_counts": dict(hit_candidate_type_counter),
        "all_gt_hit_motif_type_counts": dict(gt_hit_type_counter),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranked_eval_detail_csv", type=str, required=True)
    parser.add_argument("--partition_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="reranked_breakdown")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    detail_df = pd.read_csv(args.reranked_eval_detail_csv)
    gt_df = pd.read_csv(os.path.join(args.partition_dir, "ground_truth_motifs.csv"))

    motif_id_to_type = invert_gt_mapping(gt_df)

    topk_breakdown = summarize_topk(detail_df, motif_id_to_type)
    overall = overall_breakdown(detail_df, motif_id_to_type)

    summary = {
        "overall": overall,
        "topk_breakdown": topk_breakdown,
    }

    with open(os.path.join(args.output_dir, "reranked_breakdown_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()