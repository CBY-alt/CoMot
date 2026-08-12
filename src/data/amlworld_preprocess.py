import os
import re
import json
import hashlib
import argparse
from typing import Dict, List, Tuple, Optional

import pandas as pd


FULL_TX_KEY_COLS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
]


def normalize_value(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def stable_tx_key_from_row(row: pd.Series) -> Tuple[str, ...]:
    return tuple(normalize_value(row[c]) for c in FULL_TX_KEY_COLS)


def stable_tx_hash_from_key(key: Tuple[str, ...]) -> str:
    raw = "||".join(key)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_pattern_header(line: str) -> Optional[str]:
    m = re.match(r"BEGIN LAUNDERING ATTEMPT - (.*?)(?::.*)?$", line.strip())
    if not m:
        return None
    return m.group(1).strip()


def read_pattern_file(pattern_path: str) -> pd.DataFrame:
    """
    Parse AMLWorld pattern txt and return one row per laundering transaction
    with motif_type and motif_instance_id.
    """
    rows: List[Dict] = []
    current_motif: Optional[str] = None
    current_instance_id: Optional[str] = None
    instance_counter = 0

    with open(pattern_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            motif = parse_pattern_header(line)
            if motif is not None:
                current_motif = motif
                current_instance_id = f"MOTIF_{instance_counter:07d}"
                instance_counter += 1
                continue

            if line.startswith("END LAUNDERING ATTEMPT"):
                current_motif = None
                current_instance_id = None
                continue

            if current_motif is None:
                continue

            if not line or not re.match(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2},", line):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 11:
                continue

            record = {
                "Timestamp": normalize_value(parts[0]),
                "From Bank": normalize_value(parts[1]),
                "Account": normalize_value(parts[2]),
                "To Bank": normalize_value(parts[3]),
                "Account.1": normalize_value(parts[4]),
                "Amount Received": normalize_value(parts[5]),
                "Receiving Currency": normalize_value(parts[6]),
                "Amount Paid": normalize_value(parts[7]),
                "Payment Currency": normalize_value(parts[8]),
                "Payment Format": normalize_value(parts[9]),
                "Is Laundering": normalize_value(parts[10]),
                "motif_type": current_motif,
                "motif_instance_id": current_instance_id,
            }
            key = tuple(normalize_value(record[c]) for c in FULL_TX_KEY_COLS)
            record["tx_hash"] = stable_tx_hash_from_key(key)
            rows.append(record)

    pattern_df = pd.DataFrame(rows)
    if len(pattern_df) == 0:
        raise ValueError(f"No laundering transactions parsed from {pattern_path}")

    return pattern_df

def load_transactions_csv(trans_path: str) -> pd.DataFrame:
    # Read key fields as strings to preserve leading zeros in bank identifiers.
    dtype_map = {
        "Timestamp": "string",
        "From Bank": "string",
        "Account": "string",
        "To Bank": "string",
        "Account.1": "string",
        "Amount Received": "string",
        "Receiving Currency": "string",
        "Amount Paid": "string",
        "Payment Currency": "string",
        "Payment Format": "string",
        "Is Laundering": "string",
    }

    df = pd.read_csv(trans_path, dtype=dtype_map, keep_default_na=False)

    expected_cols = [
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account.1",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
        "Is Laundering",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in transaction csv: {missing}")

    for c in FULL_TX_KEY_COLS + ["Is Laundering"]:
        df[c] = df[c].map(normalize_value)

    df["tx_hash"] = df.apply(stable_tx_key_from_row, axis=1).map(stable_tx_hash_from_key)
    return df

def merge_patterns_into_transactions(
    tx_df: pd.DataFrame,
    pattern_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge motif labels from patterns file into full transaction table.
    If a transaction is laundering but missing from pattern file, mark as UNKNOWN_LAUNDERING.
    """
    pattern_cols = ["tx_hash", "motif_type", "motif_instance_id"]
    pattern_unique = pattern_df[pattern_cols].drop_duplicates()

    merged = tx_df.merge(pattern_unique, on="tx_hash", how="left")
    merged["motif_type"] = merged["motif_type"].fillna("CLEAN")
    merged["motif_instance_id"] = merged["motif_instance_id"].fillna("NONE")

    laundering_mask = merged["Is Laundering"].astype(str) == "1"
    missing_pattern_mask = laundering_mask & (merged["motif_type"] == "CLEAN")
    merged.loc[missing_pattern_mask, "motif_type"] = "UNKNOWN_LAUNDERING"
    merged.loc[missing_pattern_mask, "motif_instance_id"] = "UNKNOWN"

    merged.insert(0, "tx_id", range(len(merged)))
    return merged


def build_account_labels(merged_df: pd.DataFrame) -> pd.DataFrame:
    illicit_tx = merged_df[merged_df["Is Laundering"].astype(str) == "1"]

    src_accounts = illicit_tx[["Account"]].rename(columns={"Account": "account_id"})
    dst_accounts = illicit_tx[["Account.1"]].rename(columns={"Account.1": "account_id"})
    illicit_accounts = pd.concat([src_accounts, dst_accounts], ignore_index=True)
    illicit_accounts["is_illicit"] = 1
    illicit_accounts = illicit_accounts.drop_duplicates()

    all_accounts = pd.DataFrame({
        "account_id": pd.concat([merged_df["Account"], merged_df["Account.1"]], ignore_index=True).unique()
    })
    account_labels = all_accounts.merge(illicit_accounts, on="account_id", how="left")
    account_labels["is_illicit"] = account_labels["is_illicit"].fillna(0).astype(int)
    account_labels["account_int_id"] = range(len(account_labels))
    return account_labels


def build_motif_instances(merged_df: pd.DataFrame) -> pd.DataFrame:
    motif_df = merged_df[merged_df["motif_instance_id"].isin(["NONE"]) == False].copy()

    rows = []
    for motif_instance_id, g in motif_df.groupby("motif_instance_id"):
        row = {
            "motif_instance_id": motif_instance_id,
            "motif_type": g["motif_type"].iloc[0],
            "num_transactions": int(len(g)),
            "num_unique_src_accounts": int(g["Account"].nunique()),
            "num_unique_dst_accounts": int(g["Account.1"].nunique()),
            "num_unique_accounts": int(pd.concat([g["Account"], g["Account.1"]], ignore_index=True).nunique()),
            "tx_ids_json": json.dumps(g["tx_id"].tolist()),
            "tx_hashes_json": json.dumps(g["tx_hash"].tolist()),
            "accounts_json": json.dumps(sorted(set(g["Account"]).union(set(g["Account.1"])))),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def report_match_rate(tx_df: pd.DataFrame, pattern_df: pd.DataFrame):
    tx_hashes = set(tx_df["tx_hash"].tolist())
    pattern_hashes = set(pattern_df["tx_hash"].tolist())
    inter = tx_hashes & pattern_hashes

    print("[preprocess] transaction hashes:", len(tx_hashes))
    print("[preprocess] pattern hashes:", len(pattern_hashes))
    print("[preprocess] matched hashes:", len(inter))
    print("[preprocess] unmatched pattern hashes:", len(pattern_hashes - tx_hashes))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans_path", type=str, required=True)
    parser.add_argument("--pattern_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="semotif_global")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/4] Loading transactions...")
    tx_df = load_transactions_csv(args.trans_path)

    print("[2/4] Parsing patterns...")
    pattern_df = read_pattern_file(args.pattern_path)

    print("[3/4] Merging labels...")
    report_match_rate(tx_df, pattern_df)
    merged_df = merge_patterns_into_transactions(tx_df, pattern_df)
    
    print("[4/4] Building auxiliary tables...")
    account_labels = build_account_labels(merged_df)
    motif_instances = build_motif_instances(merged_df)

    merged_df.to_csv(os.path.join(args.output_dir, "transactions_labeled.csv"), index=False)
    pattern_df.to_csv(os.path.join(args.output_dir, "pattern_transactions.csv"), index=False)
    account_labels.to_csv(os.path.join(args.output_dir, "account_labels.csv"), index=False)
    motif_instances.to_csv(os.path.join(args.output_dir, "motif_instances.csv"), index=False)

    summary = {
        "num_transactions": int(len(merged_df)),
        "num_accounts": int(len(account_labels)),
        "num_laundering_transactions": int((merged_df["Is Laundering"].astype(str) == "1").sum()),
        "num_motif_instances": int(len(motif_instances)),
        "motif_type_counts": merged_df["motif_type"].value_counts().to_dict(),
    }
    with open(os.path.join(args.output_dir, "global_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved global dataset to:", args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
