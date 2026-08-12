import os
import json
import argparse
from collections import defaultdict

import pandas as pd


def load_tx(partition_dir: str) -> pd.DataFrame:
    fp = os.path.join(partition_dir, "transactions_partitioned.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    return pd.read_csv(fp)


def mark_tx_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _group(row):
        motif_type = str(row.get("motif_type", "CLEAN"))
        is_laundering = int(row.get("Is Laundering", 0))

        if motif_type not in ["CLEAN", "NONE", "nan"] and motif_type != "UNKNOWN_LAUNDERING":
            return "PATTERN_MOTIF"
        if motif_type == "UNKNOWN_LAUNDERING" or (is_laundering == 1 and motif_type in ["CLEAN", "NONE", "nan"]):
            return "UNKNOWN_LAUNDERING"
        return "CLEAN"

    df["tx_group"] = df.apply(_group, axis=1)
    return df


def basic_summary(df: pd.DataFrame):
    out = {
        "num_transactions": int(len(df)),
        "num_clean": int((df["tx_group"] == "CLEAN").sum()),
        "num_pattern_motif": int((df["tx_group"] == "PATTERN_MOTIF").sum()),
        "num_unknown_laundering": int((df["tx_group"] == "UNKNOWN_LAUNDERING").sum()),
        "num_inter_bank": int(df["is_inter_partition"].sum()),
        "num_intra_bank": int((~df["is_inter_partition"].astype(bool)).sum()),
    }
    return out


def inter_intra_by_group(df: pd.DataFrame):
    rows = []
    for group_name, sub in df.groupby("tx_group"):
        total = len(sub)
        inter = int(sub["is_inter_partition"].sum())
        intra = int(total - inter)
        rows.append({
            "tx_group": group_name,
            "num_transactions": int(total),
            "num_inter_bank": inter,
            "num_intra_bank": intra,
            "inter_ratio": inter / total if total > 0 else 0.0,
            "intra_ratio": intra / total if total > 0 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("tx_group").reset_index(drop=True)


def motif_type_summary(df: pd.DataFrame):
    motif_df = df[df["tx_group"] == "PATTERN_MOTIF"].copy()
    if len(motif_df) == 0:
        return pd.DataFrame()

    rows = []
    for motif_type, sub in motif_df.groupby("motif_type"):
        inst_ids = sorted(sub["motif_instance_id"].dropna().unique().tolist())
        inter_sub = sub[sub["is_inter_partition"] == True]

        # Count motif instances containing at least one cross-institution transaction.
        inst_with_inter = inter_sub["motif_instance_id"].dropna().unique().tolist()

        rows.append({
            "motif_type": motif_type,
            "num_instances": int(len(inst_ids)),
            "num_transactions": int(len(sub)),
            "num_inter_bank_transactions": int(len(inter_sub)),
            "num_intra_bank_transactions": int(len(sub) - len(inter_sub)),
            "inter_tx_ratio": len(inter_sub) / len(sub) if len(sub) > 0 else 0.0,
            "num_instances_with_inter": int(len(inst_with_inter)),
            "inter_instance_ratio": len(inst_with_inter) / len(inst_ids) if len(inst_ids) > 0 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("motif_type").reset_index(drop=True)


def motif_instance_summary(df: pd.DataFrame):
    motif_df = df[df["tx_group"] == "PATTERN_MOTIF"].copy()
    if len(motif_df) == 0:
        return pd.DataFrame()

    rows = []
    for motif_id, sub in motif_df.groupby("motif_instance_id"):
        motif_type = str(sub["motif_type"].iloc[0])

        banks = set(sub["src_bank_partition"].astype(str).tolist()) | set(sub["dst_bank_partition"].astype(str).tolist())
        inter_sub = sub[sub["is_inter_partition"] == True]

        rows.append({
            "motif_instance_id": motif_id,
            "motif_type": motif_type,
            "num_transactions": int(len(sub)),
            "num_inter_bank_transactions": int(len(inter_sub)),
            "num_intra_bank_transactions": int(len(sub) - len(inter_sub)),
            "inter_tx_ratio": len(inter_sub) / len(sub) if len(sub) > 0 else 0.0,
            "num_banks_touched": int(len(banks)),
            "has_inter_bank": 1 if len(inter_sub) > 0 else 0,
        })

    out = pd.DataFrame(rows).sort_values(
        ["motif_type", "num_inter_bank_transactions", "num_banks_touched"],
        ascending=[True, False, False]
    ).reset_index(drop=True)
    return out


def motif_bank_span_summary(instance_df: pd.DataFrame):
    if len(instance_df) == 0:
        return pd.DataFrame()

    rows = []
    for motif_type, sub in instance_df.groupby("motif_type"):
        rows.append({
            "motif_type": motif_type,
            "num_instances": int(len(sub)),
            "mean_num_banks_touched": float(sub["num_banks_touched"].mean()),
            "median_num_banks_touched": float(sub["num_banks_touched"].median()),
            "mean_inter_bank_transactions": float(sub["num_inter_bank_transactions"].mean()),
            "median_inter_bank_transactions": float(sub["num_inter_bank_transactions"].median()),
            "instance_has_inter_ratio": float(sub["has_inter_bank"].mean()),
        })
    return pd.DataFrame(rows).sort_values("motif_type").reset_index(drop=True)


def anchor_quality_summary(df: pd.DataFrame):
    """Compare cross-institution edge ratios for clean and motif transactions."""
    clean = df[df["tx_group"] == "CLEAN"]
    motif = df[df["tx_group"] == "PATTERN_MOTIF"]
    unk = df[df["tx_group"] == "UNKNOWN_LAUNDERING"]

    clean_inter = int(clean["is_inter_partition"].sum())
    motif_inter = int(motif["is_inter_partition"].sum())
    unk_inter = int(unk["is_inter_partition"].sum())

    summary = {
        "clean_inter_bank_edges": clean_inter,
        "motif_inter_bank_edges": motif_inter,
        "unknown_inter_bank_edges": unk_inter,
        "clean_to_motif_inter_ratio": (clean_inter / motif_inter) if motif_inter > 0 else None,
        "all_laundering_inter_bank_edges": motif_inter + unk_inter,
        "clean_to_all_laundering_inter_ratio": (
            clean_inter / (motif_inter + unk_inter)
        ) if (motif_inter + unk_inter) > 0 else None,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="partition_protocol_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_tx(args.partition_dir)
    df = mark_tx_group(df)

    basic = basic_summary(df)
    by_group = inter_intra_by_group(df)
    by_type = motif_type_summary(df)
    by_instance = motif_instance_summary(df)
    by_span = motif_bank_span_summary(by_instance)
    anchor_quality = anchor_quality_summary(df)

    by_group.to_csv(os.path.join(args.output_dir, "inter_intra_by_group.csv"), index=False)
    by_type.to_csv(os.path.join(args.output_dir, "motif_type_summary.csv"), index=False)
    by_instance.to_csv(os.path.join(args.output_dir, "motif_instance_summary.csv"), index=False)
    by_span.to_csv(os.path.join(args.output_dir, "motif_bank_span_summary.csv"), index=False)

    summary = {
        "basic_summary": basic,
        "anchor_quality_summary": anchor_quality,
    }

    with open(os.path.join(args.output_dir, "partition_protocol_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\n[inter_intra_by_group]")
    print(by_group)

    print("\n[motif_type_summary]")
    print(by_type.head(20))

    print("\n[motif_bank_span_summary]")
    print(by_span.head(20))


if __name__ == "__main__":
    main()
