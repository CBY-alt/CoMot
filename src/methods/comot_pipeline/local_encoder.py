import os
import json
import math
import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor

import pandas as pd


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def log_bucket(x: float, base: float = 2.0) -> int:
    if x <= 0:
        return 0
    return int(math.log(x + 1.0, base))


def time_to_hour_bucket(ts: str) -> int:
    # Expected input format: 2022/09/01 13:24.
    try:
        hh = int(str(ts)[11:13])
        return hh
    except Exception:
        return 0


class LocalEncoder:
    def __init__(self, k_hop: int = 2):
        self.k_hop = k_hop

    def _build_graph(self, df: pd.DataFrame):
        out_adj = defaultdict(list)
        in_adj = defaultdict(list)
        local_nodes = set(df["local_account_id"].unique().tolist())

        roles = df["local_role"].to_numpy(copy=False)
        local = df["local_account_id"].to_numpy(copy=False)
        remote = df["counterparty_account_masked"].to_numpy(copy=False)
        for role, local_id, remote_id in zip(roles, local, remote):
            if role == "SENDER":
                src, dst = local_id, remote_id
            else:
                src, dst = remote_id, local_id
            out_adj[src].append(dst)
            in_adj[dst].append(src)

        all_nodes = set(out_adj.keys()) | set(in_adj.keys()) | local_nodes
        return all_nodes, out_adj, in_adj

    def _degree_tables(self, all_nodes, out_adj, in_adj):
        out_deg = {n: len(out_adj.get(n, [])) for n in all_nodes}
        in_deg = {n: len(in_adj.get(n, [])) for n in all_nodes}
        return out_deg, in_deg

    def _hop_reach_stats(self, node, out_adj, in_adj, direction="out", k_hop=2):
        visited = {node}
        q = deque([(node, 0)])
        reached = set()
        edge_cnt = 0

        while q:
            cur, depth = q.popleft()
            if depth >= k_hop:
                continue

            neighbors = out_adj.get(cur, []) if direction == "out" else in_adj.get(cur, [])
            edge_cnt += len(neighbors)
            for nxt in neighbors:
                reached.add(nxt)
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, depth + 1))

        return len(reached), edge_cnt

    def _precompute_node_stats(self, nodes, out_adj, in_adj, out_deg, in_deg):
        """Cache row-independent node features to avoid repeated BFS traversals."""
        node_stats = {}

        for node in nodes:
            one_hop_out_nodes, one_hop_out_edges = self._hop_reach_stats(
                node, out_adj, in_adj, direction="out", k_hop=1
            )
            two_hop_out_nodes, two_hop_out_edges = self._hop_reach_stats(
                node, out_adj, in_adj, direction="out", k_hop=self.k_hop
            )
            one_hop_in_nodes, one_hop_in_edges = self._hop_reach_stats(
                node, out_adj, in_adj, direction="in", k_hop=1
            )
            two_hop_in_nodes, two_hop_in_edges = self._hop_reach_stats(
                node, out_adj, in_adj, direction="in", k_hop=self.k_hop
            )

            node_stats[node] = {
                "local_out_deg": out_deg.get(node, 0),
                "local_in_deg": in_deg.get(node, 0),
                "one_hop_out_nodes": one_hop_out_nodes,
                "two_hop_out_nodes": two_hop_out_nodes,
                "one_hop_in_nodes": one_hop_in_nodes,
                "two_hop_in_nodes": two_hop_in_nodes,
                # Retain edge statistics even though the current feature vector does not
                # consume them directly, preserving the configured computation.
                "one_hop_out_edges": one_hop_out_edges,
                "two_hop_out_edges": two_hop_out_edges,
                "one_hop_in_edges": one_hop_in_edges,
                "two_hop_in_edges": two_hop_in_edges,
            }

        return node_stats

    def _feature_for_anchor_from_cache(self, node, row_dict, node_stats, role):
        """Extract outward features for SENDER and inward features for RECEIVER."""
        amt_recv = safe_float(row_dict["Amount Received"])
        amt_paid = safe_float(row_dict["Amount Paid"])
        hour_bucket = time_to_hour_bucket(row_dict["Timestamp"])

        st = node_stats[node]

        local_out_deg = st["local_out_deg"]
        local_in_deg = st["local_in_deg"]

        one_hop_out_nodes = st["one_hop_out_nodes"]
        two_hop_out_nodes = st["two_hop_out_nodes"]
        one_hop_in_nodes = st["one_hop_in_nodes"]
        two_hop_in_nodes = st["two_hop_in_nodes"]

        # role-sensitive summary
        if role == "SENDER":
            primary_deg = local_out_deg
            secondary_deg = local_in_deg
            primary_reach_1 = one_hop_out_nodes
            primary_reach_2 = two_hop_out_nodes
            opposite_reach_1 = one_hop_in_nodes
            opposite_reach_2 = two_hop_in_nodes
        else:
            primary_deg = local_in_deg
            secondary_deg = local_out_deg
            primary_reach_1 = one_hop_in_nodes
            primary_reach_2 = two_hop_in_nodes
            opposite_reach_1 = one_hop_out_nodes
            opposite_reach_2 = two_hop_out_nodes

        feature = {
            "role": role,
            "primary_deg": primary_deg,
            "secondary_deg": secondary_deg,
            "primary_deg_bucket": log_bucket(primary_deg, base=2.0),
            "secondary_deg_bucket": log_bucket(secondary_deg, base=2.0),
            "primary_reach_1": primary_reach_1,
            "primary_reach_2": primary_reach_2,
            "opposite_reach_1": opposite_reach_1,
            "opposite_reach_2": opposite_reach_2,
            "primary_reach_1_bucket": log_bucket(primary_reach_1, base=2.0),
            "primary_reach_2_bucket": log_bucket(primary_reach_2, base=2.0),
            "opposite_reach_1_bucket": log_bucket(opposite_reach_1, base=2.0),
            "opposite_reach_2_bucket": log_bucket(opposite_reach_2, base=2.0),
            "amount_recv_bucket": log_bucket(amt_recv, base=10.0),
            "amount_paid_bucket": log_bucket(amt_paid, base=10.0),
            "hour_bucket": hour_bucket,
            "payment_format": str(row_dict["Payment Format"]),
            "recv_currency": str(row_dict["Receiving Currency"]),
            "paid_currency": str(row_dict["Payment Currency"]),
            "is_inter_partition": int(bool(row_dict["is_inter_partition"])),
        }
        return feature

    def encode_one_bank(self, bank_view_path: str, output_dir: str):
        bank_df = pd.read_csv(bank_view_path)
        bank_name = os.path.basename(bank_view_path).replace("_view.csv", "")

        all_nodes, out_adj, in_adj = self._build_graph(bank_df)
        out_deg, in_deg = self._degree_tables(all_nodes, out_adj, in_adj)

        inter_df = bank_df[bank_df["is_inter_partition"] == True].copy()
        rows = []

        # Cache only local nodes present in cross-partition evidence rows.
        unique_local_nodes = inter_df["local_account_id"].unique().tolist()
        node_stats = self._precompute_node_stats(unique_local_nodes, out_adj, in_adj, out_deg, in_deg)

        print(f"[{bank_name}] encoding {len(inter_df)} inter-partition evidences...")

        # Preserve field semantics; record dictionaries are stable for this iteration.
        for row_dict in inter_df.to_dict("records"):
            local_node = row_dict["local_account_id"]
            role = row_dict["local_role"]

            feat = self._feature_for_anchor_from_cache(
                local_node, row_dict, node_stats, role
            )

            rows.append({
                "bank_id": bank_name,
                "tx_id": int(row_dict["tx_id"]),
                "public_tx_tag": row_dict["public_tx_tag"],
                "evidence_node_id": f"{bank_name}::{local_node}",
                "local_account_id": local_node,
                "role": role,
                "motif_type": row_dict["motif_type"],
                "motif_instance_id": row_dict["motif_instance_id"],
                "feature_json": json.dumps(feat),
            })

        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, f"{bank_name}_evidences.csv"),
            index=False
        )
        print(f"[{bank_name}] saved to {output_dir}")


def _process_one_file(args_tuple):
    bank_view_path, output_dir, k_hop = args_tuple
    encoder = LocalEncoder(k_hop=k_hop)
    encoder.encode_one_bank(bank_view_path, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="semotif_local_evidences")
    parser.add_argument("--k_hop", type=int, default=2)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of CPU worker processes. Default=1 keeps original single-process behavior."
    )
    args = parser.parse_args()

    views_dir = os.path.join(args.partition_dir, "bank_views")
    files = sorted([
        os.path.join(views_dir, f)
        for f in os.listdir(views_dir)
        if f.endswith("_view.csv")
    ])

    if args.num_workers <= 1:
        encoder = LocalEncoder(k_hop=args.k_hop)
        for fp in files:
            encoder.encode_one_bank(fp, args.output_dir)
    else:
        task_args = [(fp, args.output_dir, args.k_hop) for fp in files]
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            list(ex.map(_process_one_file, task_args))


if __name__ == "__main__":
    main()
