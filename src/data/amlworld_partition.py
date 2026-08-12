import os
import json
import random
import argparse
from collections import defaultdict, deque, Counter

import pandas as pd
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_public_tx_tag(row, time_mode="minute"):
    ts = str(row["Timestamp"])
    if time_mode == "minute":
        ts_key = ts[:16]
    elif time_mode == "hour":
        ts_key = ts[:13]
    else:
        ts_key = ts

    parts = [
        ts_key,
        str(row["Amount Received"]),
        str(row["Receiving Currency"]),
        str(row["Amount Paid"]),
        str(row["Payment Currency"]),
        str(row["Payment Format"]),
    ]
    return "||".join(parts)


class ControlledPartitionerV2:
    def __init__(
        self,
        global_dir: str,
        output_dir: str,
        num_banks: int = 5,
        min_banks_per_motif: int = 2,
        clean_inter_ratio: float = 0.01,
        time_mode: str = "minute",
        seed: int = 42,
        keep_unknown_laundering: bool = True,
        max_clean_neighbors_to_reassign: int = 3,
    ):
        self.global_dir = global_dir
        self.output_dir = output_dir
        self.num_banks = num_banks
        self.bank_ids = [f"BANK_{i:03d}" for i in range(num_banks)]
        self.min_banks_per_motif = min_banks_per_motif
        self.clean_inter_ratio = clean_inter_ratio
        self.time_mode = time_mode
        self.seed = seed
        self.keep_unknown_laundering = keep_unknown_laundering
        self.max_clean_neighbors_to_reassign = max_clean_neighbors_to_reassign
        self.rng = random.Random(seed)

    def load_global(self):
        fp = os.path.join(self.global_dir, "transactions_labeled.csv")
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        df = pd.read_csv(fp)
        return df

    def mark_tx_group(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        motif_type = df.get("motif_type", pd.Series(index=df.index, dtype=object)).astype(str)
        is_laundering = pd.to_numeric(df.get("Is Laundering", 0), errors="coerce").fillna(0).astype(int)

        mask_pattern = (~motif_type.isin(["CLEAN", "NONE", "nan"])) & (motif_type != "UNKNOWN_LAUNDERING")
        mask_unknown = (motif_type == "UNKNOWN_LAUNDERING") | (
            (is_laundering == 1) & motif_type.isin(["CLEAN", "NONE", "nan"])
        )

        df["tx_group"] = np.where(
            mask_pattern,
            "PATTERN_MOTIF",
            np.where(mask_unknown, "UNKNOWN_LAUNDERING", "CLEAN")
        ).astype(str)
        return df

    def maybe_filter_unknown(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.keep_unknown_laundering:
            return df
        return df[df["tx_group"] != "UNKNOWN_LAUNDERING"].copy()

    def ensure_tx_id(self, df: pd.DataFrame) -> pd.DataFrame:
        if "tx_id" not in df.columns:
            df = df.reset_index(drop=True)
            df["tx_id"] = np.arange(len(df), dtype=int)
        return df

    def assign_motif_accounts(self, df: pd.DataFrame):
        """Assign motif instances across institutions before other accounts."""
        acct_to_bank = {}
        motif_df = df[df["tx_group"] == "PATTERN_MOTIF"].copy()

        # Preserve the original groupby behavior (sort=True).
        for motif_id, sub in motif_df.groupby("motif_instance_id"):
            accounts = sorted(set(sub["Account"].astype(str).tolist()) | set(sub["Account.1"].astype(str).tolist()))
            if len(accounts) == 0:
                continue

            k = min(self.num_banks, max(2, self.min_banks_per_motif))
            chosen_banks = self.rng.sample(self.bank_ids, k)

            # Assign round-robin so each motif spans institutions.
            for i, acct in enumerate(accounts):
                if acct not in acct_to_bank:
                    acct_to_bank[acct] = chosen_banks[i % len(chosen_banks)]

        return acct_to_bank

    def build_clean_graph(self, df: pd.DataFrame):
        """Build an undirected clean graph using only CLEAN edges."""
        clean_df = df[df["tx_group"] == "CLEAN"]
        adj = defaultdict(set)

        for a, b in zip(clean_df["Account"].astype(str), clean_df["Account.1"].astype(str)):
            if a == b:
                adj[a].add(b)
                continue
            adj[a].add(b)
            adj[b].add(a)

        return adj

    def connected_components(self, adj):
        visited = set()
        comps = []

        for node in adj:
            if node in visited:
                continue
            q = deque([node])
            visited.add(node)
            comp = []

            while q:
                cur = q.popleft()
                comp.append(cur)
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)

            comps.append(comp)

        return comps

    def assign_clean_components(self, df: pd.DataFrame, acct_to_bank: dict):
        """Assign each unassigned clean component to one bank to favor intra-bank edges."""
        clean_df = df[df["tx_group"] == "CLEAN"]
        adj = self.build_clean_graph(df)

        # Restrict the subgraph to clean nodes that have not yet been assigned.
        remaining_nodes = set()
        for a, b in zip(clean_df["Account"].astype(str), clean_df["Account.1"].astype(str)):
            if a not in acct_to_bank:
                remaining_nodes.add(a)
            if b not in acct_to_bank:
                remaining_nodes.add(b)

        sub_adj = defaultdict(set)
        for n in remaining_nodes:
            for nb in adj.get(n, []):
                if nb in remaining_nodes:
                    sub_adj[n].add(nb)

        comps = self.connected_components(sub_adj)

        # Assign larger connected components first.
        comps = sorted(comps, key=lambda x: len(x), reverse=True)

        bank_load = Counter(acct_to_bank.values())

        for comp in comps:
            # Select the bank with the smallest current load.
            target_bank = min(self.bank_ids, key=lambda b: bank_load[b])
            for acct in comp:
                acct_to_bank[acct] = target_bank
                bank_load[target_bank] += 1

        # Assign isolated clean nodes and nodes omitted near motif neighborhoods.
        all_clean_accounts = set(clean_df["Account"].astype(str).tolist()) | set(clean_df["Account.1"].astype(str).tolist())
        for acct in all_clean_accounts:
            if acct not in acct_to_bank:
                target_bank = min(self.bank_ids, key=lambda b: bank_load[b])
                acct_to_bank[acct] = target_bank
                bank_load[target_bank] += 1

        return acct_to_bank

    def assign_unknown_accounts(self, df: pd.DataFrame, acct_to_bank: dict):
        """Assign unknown-laundering accounts near existing assignments or randomly."""
        unk_df = df[df["tx_group"] == "UNKNOWN_LAUNDERING"]
        bank_load = Counter(acct_to_bank.values())

        for a, b in zip(unk_df["Account"].astype(str), unk_df["Account.1"].astype(str)):
            a_in = a in acct_to_bank
            b_in = b in acct_to_bank

            if a_in and b_in:
                continue
            elif a_in and not b_in:
                acct_to_bank[b] = acct_to_bank[a]
                bank_load[acct_to_bank[a]] += 1
            elif b_in and not a_in:
                acct_to_bank[a] = acct_to_bank[b]
                bank_load[acct_to_bank[b]] += 1
            else:
                target_bank = min(self.bank_ids, key=lambda x: bank_load[x])
                acct_to_bank[a] = target_bank
                acct_to_bank[b] = target_bank
                bank_load[target_bank] += 2

        return acct_to_bank

    def collect_all_accounts(self, df: pd.DataFrame):
        return sorted(set(df["Account"].astype(str).tolist()) | set(df["Account.1"].astype(str).tolist()))

    def fill_any_remaining(self, df: pd.DataFrame, acct_to_bank: dict):
        all_accts = self.collect_all_accounts(df)
        bank_load = Counter(acct_to_bank.values())

        for acct in all_accts:
            if acct not in acct_to_bank:
                target_bank = min(self.bank_ids, key=lambda x: bank_load[x])
                acct_to_bank[acct] = target_bank
                bank_load[target_bank] += 1

        return acct_to_bank

    def finalize_partition(self, df: pd.DataFrame, acct_to_bank: dict):
        out = df.copy()
        out["Account"] = out["Account"].astype(str)
        out["Account.1"] = out["Account.1"].astype(str)

        out["src_bank_partition"] = out["Account"].map(acct_to_bank)
        out["dst_bank_partition"] = out["Account.1"].map(acct_to_bank)
        out["is_inter_partition"] = (out["src_bank_partition"] != out["dst_bank_partition"])

        ts = out["Timestamp"].astype(str)
        if self.time_mode == "minute":
            ts_key = ts.str.slice(0, 16)
        elif self.time_mode == "hour":
            ts_key = ts.str.slice(0, 13)
        else:
            ts_key = ts

        out["public_tx_tag"] = (
            ts_key
            + "||" + out["Amount Received"].astype(str)
            + "||" + out["Receiving Currency"].astype(str)
            + "||" + out["Amount Paid"].astype(str)
            + "||" + out["Payment Currency"].astype(str)
            + "||" + out["Payment Format"].astype(str)
        )
        return out

    def current_clean_inter_ratio(self, part_df: pd.DataFrame):
        clean_df = part_df[part_df["tx_group"] == "CLEAN"]
        if len(clean_df) == 0:
            return 0.0
        return float(clean_df["is_inter_partition"].mean())

    def reduce_clean_inter_edges(self, part_df: pd.DataFrame, acct_to_bank: dict):
        """Reduce an excessive clean inter-bank ratio by moving low-impact endpoints."""
        clean_df = part_df[part_df["tx_group"] == "CLEAN"]
        if len(clean_df) == 0:
            return acct_to_bank

        # Prefer lower-degree clean nodes to limit effects on adjacent edges.
        clean_deg = Counter()
        for a, b in zip(clean_df["Account"].astype(str), clean_df["Account.1"].astype(str)):
            clean_deg[a] += 1
            clean_deg[b] += 1

        current_ratio = self.current_clean_inter_ratio(part_df)
        if current_ratio <= self.clean_inter_ratio:
            return acct_to_bank

        inter_rows = clean_df[clean_df["is_inter_partition"] == True]
        target_inter_num = int(len(clean_df) * self.clean_inter_ratio)
        current_inter_num = int(inter_rows.shape[0])
        need_reduce = current_inter_num - target_inter_num

        if need_reduce <= 0:
            return acct_to_bank

        # Repair edges in ascending endpoint-degree order.
        candidate_rows = []
        for a, b in zip(inter_rows["Account"].astype(str), inter_rows["Account.1"].astype(str)):
            candidate_rows.append((
                min(clean_deg[a], clean_deg[b]),
                clean_deg[a] + clean_deg[b],
                a,
                b
            ))

        candidate_rows.sort(key=lambda x: (x[0], x[1]))

        fixed = 0
        for _, _, a, b in candidate_rows:
            if fixed >= need_reduce:
                break

            ba = acct_to_bank[a]
            bb = acct_to_bank[b]

            # Move the lower-degree endpoint to the other endpoint's bank.
            move_acct = a if clean_deg[a] <= clean_deg[b] else b
            target_bank = bb if move_acct == a else ba

            old_bank = acct_to_bank[move_acct]
            if old_bank == target_bank:
                continue

            acct_to_bank[move_acct] = target_bank
            fixed += 1

        return acct_to_bank

    def enforce_clean_inter_ratio(self, df: pd.DataFrame, acct_to_bank: dict):
        """Favor intra-bank clean edges, then introduce the requested inter-bank share."""
        part_df = self.finalize_partition(df, acct_to_bank)
        cur_ratio = self.current_clean_inter_ratio(part_df)

        # First reduce the ratio toward the target.
        if cur_ratio > self.clean_inter_ratio:
            acct_to_bank = self.reduce_clean_inter_edges(part_df, acct_to_bank)
            part_df = self.finalize_partition(df, acct_to_bank)
            cur_ratio = self.current_clean_inter_ratio(part_df)

        # If below target, introduce a controlled number of inter-bank clean edges.
        clean_df = part_df[part_df["tx_group"] == "CLEAN"]
        if len(clean_df) == 0:
            return acct_to_bank

        target_inter_num = int(len(clean_df) * self.clean_inter_ratio)
        current_inter_num = int(clean_df["is_inter_partition"].sum())

        if current_inter_num >= target_inter_num:
            return acct_to_bank

        need_add = target_inter_num - current_inter_num
        intra_rows = clean_df[clean_df["is_inter_partition"] == False]

        if len(intra_rows) == 0:
            return acct_to_bank

        # Prefer low-degree endpoints to minimize cascading changes.
        clean_deg = Counter()
        for a, b in zip(clean_df["Account"].astype(str), clean_df["Account.1"].astype(str)):
            clean_deg[a] += 1
            clean_deg[b] += 1

        cand = []
        for a, b in zip(intra_rows["Account"].astype(str), intra_rows["Account.1"].astype(str)):
            cand.append((min(clean_deg[a], clean_deg[b]), clean_deg[a] + clean_deg[b], a, b))

        cand.sort(key=lambda x: (x[0], x[1]))

        added = 0
        for _, _, a, b in cand:
            if added >= need_add:
                break

            move_acct = a if clean_deg[a] <= clean_deg[b] else b
            current_bank = acct_to_bank[move_acct]
            other_banks = [x for x in self.bank_ids if x != current_bank]
            if not other_banks:
                continue

            acct_to_bank[move_acct] = self.rng.choice(other_banks)
            added += 1

        return acct_to_bank

    def build_ground_truth_files(self, part_df: pd.DataFrame):
        all_accts = sorted(set(part_df["Account"].astype(str).tolist()) | set(part_df["Account.1"].astype(str).tolist()))

        illicit_accounts = set(
            pd.concat(
                [
                    part_df.loc[part_df["Is Laundering"] == 1, "Account"].astype(str),
                    part_df.loc[part_df["Is Laundering"] == 1, "Account.1"].astype(str),
                ],
                ignore_index=True
            ).unique().tolist()
        )

        src_map = (
            part_df[["Account", "src_bank_partition"]]
            .copy()
            .assign(Account=lambda x: x["Account"].astype(str))
            .drop_duplicates(subset=["Account"])
            .set_index("Account")["src_bank_partition"]
            .to_dict()
        )
        dst_map = (
            part_df[["Account.1", "dst_bank_partition"]]
            .copy()
            .assign(**{"Account.1": lambda x: x["Account.1"].astype(str)})
            .drop_duplicates(subset=["Account.1"])
            .set_index("Account.1")["dst_bank_partition"]
            .to_dict()
        )

        gt_accounts = []
        for acct in all_accts:
            bank_partition = src_map.get(acct, dst_map.get(acct))
            gt_accounts.append({
                "account_id": acct,
                "is_illicit": 1 if acct in illicit_accounts else 0,
                "bank_partition": bank_partition,
            })

        pd.DataFrame(gt_accounts).to_csv(os.path.join(self.output_dir, "ground_truth_accounts.csv"), index=False)

        gt_edges_df = part_df[[
            "tx_id", "Timestamp",
            "Account", "Account.1",
            "src_bank_partition", "dst_bank_partition",
            "is_inter_partition",
            "Is Laundering",
            "motif_type",
            "motif_instance_id",
            "tx_group"
        ]].copy()
        gt_edges_df.to_csv(os.path.join(self.output_dir, "ground_truth_edges.csv"), index=False)

        motif_df = part_df[part_df["tx_group"] == "PATTERN_MOTIF"].copy()
        motif_rows = []
        for motif_id, sub in motif_df.groupby("motif_instance_id"):
            motif_rows.append({
                "motif_instance_id": motif_id,
                "motif_type": str(sub["motif_type"].iloc[0]),
                "num_transactions": int(len(sub)),
                "num_inter_bank_transactions": int(sub["is_inter_partition"].sum()),
                "num_banks_touched": int(
                    len(set(sub["src_bank_partition"].astype(str).tolist()) | set(sub["dst_bank_partition"].astype(str).tolist()))
                ),
                "tx_ids_json": json.dumps(sorted(sub["tx_id"].astype(int).tolist())),
            })

        pd.DataFrame(motif_rows).to_csv(os.path.join(self.output_dir, "ground_truth_motifs.csv"), index=False)

    def build_bank_views(self, part_df: pd.DataFrame):
        views_dir = os.path.join(self.output_dir, "bank_views")
        ensure_dir(views_dir)

        base_cols = [
            "tx_id", "Timestamp", "Account", "Account.1",
            "Amount Received", "Receiving Currency",
            "Amount Paid", "Payment Currency", "Payment Format",
            "Is Laundering", "motif_type", "motif_instance_id", "tx_group",
            "src_bank_partition", "dst_bank_partition",
            "is_inter_partition", "public_tx_tag"
        ]

        for bank in self.bank_ids:
            sender_mask = part_df["src_bank_partition"] == bank
            receiver_mask = part_df["dst_bank_partition"] == bank

            sender_df = part_df.loc[sender_mask, base_cols].copy()
            if len(sender_df) > 0:
                sender_df["local_role"] = "SENDER"
                sender_df["local_account_id"] = sender_df["Account"].astype(str)
                sender_df["counterparty_account_masked"] = (
                    "REMOTE::"
                    + sender_df["dst_bank_partition"].astype(str)
                    + "::"
                    + sender_df["Account.1"].astype(str)
                )

            receiver_df = part_df.loc[receiver_mask, base_cols].copy()
            if len(receiver_df) > 0:
                receiver_df["local_role"] = "RECEIVER"
                receiver_df["local_account_id"] = receiver_df["Account.1"].astype(str)
                receiver_df["counterparty_account_masked"] = (
                    "REMOTE::"
                    + receiver_df["src_bank_partition"].astype(str)
                    + "::"
                    + receiver_df["Account"].astype(str)
                )

            visible_df = pd.concat([sender_df, receiver_df], ignore_index=True)

            if len(visible_df) > 0:
                visible_df["Account"] = visible_df["Account"].astype(str)
                visible_df["Account.1"] = visible_df["Account.1"].astype(str)
                visible_df["Is Laundering"] = visible_df["Is Laundering"].astype(int)
                visible_df["is_inter_partition"] = visible_df["is_inter_partition"].astype(bool)
                visible_df["tx_id"] = visible_df["tx_id"].astype(int)

            visible_df.to_csv(os.path.join(views_dir, f"{bank}_view.csv"), index=False)

    def summarize(self, part_df: pd.DataFrame):
        clean_df = part_df[part_df["tx_group"] == "CLEAN"]
        motif_df = part_df[part_df["tx_group"] == "PATTERN_MOTIF"]
        unk_df = part_df[part_df["tx_group"] == "UNKNOWN_LAUNDERING"]

        summary = {
            "num_transactions": int(len(part_df)),
            "num_banks": self.num_banks,
            "clean_inter_ratio_actual": float(clean_df["is_inter_partition"].mean()) if len(clean_df) > 0 else 0.0,
            "motif_inter_ratio_actual": float(motif_df["is_inter_partition"].mean()) if len(motif_df) > 0 else 0.0,
            "unknown_inter_ratio_actual": float(unk_df["is_inter_partition"].mean()) if len(unk_df) > 0 else 0.0,
            "clean_inter_edges": int(clean_df["is_inter_partition"].sum()),
            "motif_inter_edges": int(motif_df["is_inter_partition"].sum()),
            "unknown_inter_edges": int(unk_df["is_inter_partition"].sum()),
            "clean_to_motif_inter_ratio": (
                float(clean_df["is_inter_partition"].sum()) / float(motif_df["is_inter_partition"].sum())
            ) if int(motif_df["is_inter_partition"].sum()) > 0 else None,
            "target_clean_inter_ratio": self.clean_inter_ratio,
            "min_banks_per_motif": self.min_banks_per_motif,
            "time_mode": self.time_mode,
            "seed": self.seed,
        }

        with open(os.path.join(self.output_dir, "partition_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(summary, indent=2))

    def run(self):
        ensure_dir(self.output_dir)

        df = self.load_global()
        df = self.mark_tx_group(df)
        df = self.maybe_filter_unknown(df)
        df = self.ensure_tx_id(df)

        # 1) Establish cross-institution exposure for motifs.
        acct_to_bank = self.assign_motif_accounts(df)

        # 2) Assign each clean connected component to a single bank.
        acct_to_bank = self.assign_clean_components(df, acct_to_bank)

        # 3) Assign remaining unknown-laundering accounts.
        acct_to_bank = self.assign_unknown_accounts(df, acct_to_bank)

        # 4) Assign any remaining accounts.
        acct_to_bank = self.fill_any_remaining(df, acct_to_bank)

        # 5) enforce controlled clean inter ratio
        acct_to_bank = self.enforce_clean_inter_ratio(df, acct_to_bank)

        part_df = self.finalize_partition(df, acct_to_bank)
        part_df.to_csv(os.path.join(self.output_dir, "transactions_partitioned.csv"), index=False)

        with open(os.path.join(self.output_dir, "account_partition_map.json"), "w", encoding="utf-8") as f:
            json.dump(acct_to_bank, f, indent=2)

        self.build_ground_truth_files(part_df)
        self.build_bank_views(part_df)
        self.summarize(part_df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="semotif_partitioned_controlled_v2")
    parser.add_argument("--num_banks", type=int, default=5)
    parser.add_argument("--min_banks_per_motif", type=int, default=2)
    parser.add_argument("--clean_inter_ratio", type=float, default=0.01)
    parser.add_argument("--time_mode", type=str, default="minute", choices=["minute", "hour", "full"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_unknown_laundering", action="store_true")
    parser.add_argument("--max_clean_neighbors_to_reassign", type=int, default=3)
    args = parser.parse_args()

    partitioner = ControlledPartitionerV2(
        global_dir=args.global_dir,
        output_dir=args.output_dir,
        num_banks=args.num_banks,
        min_banks_per_motif=args.min_banks_per_motif,
        clean_inter_ratio=args.clean_inter_ratio,
        time_mode=args.time_mode,
        seed=args.seed,
        keep_unknown_laundering=args.keep_unknown_laundering,
        max_clean_neighbors_to_reassign=args.max_clean_neighbors_to_reassign,
    )
    partitioner.run()


if __name__ == "__main__":
    main()
