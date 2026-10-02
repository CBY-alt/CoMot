import os
import csv
import json
import argparse
from collections import defaultdict


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def bucket_closeness(a: int, b: int) -> float:
    return 1.0 / (1.0 + abs(a - b))


def categorical_match(a, b) -> float:
    return 1.0 if a == b else 0.0


def parse_feature_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def asymmetry_score(primary_deg: int, secondary_deg: int) -> float:
    # Reward a clear directional imbalance.
    denom = primary_deg + secondary_deg + 1.0
    return abs(primary_deg - secondary_deg) / denom


def direction_bias_score(primary_reach_2: int, opposite_reach_2: int) -> float:
    denom = primary_reach_2 + opposite_reach_2 + 1.0
    return (primary_reach_2 - opposite_reach_2) / denom


def role_balance(sender_feat, receiver_feat):
    # Score higher when the sender is output-heavy and the receiver is input-heavy.
    s_bias = direction_bias_score(sender_feat["primary_reach_2"], sender_feat["opposite_reach_2"])
    r_bias = direction_bias_score(receiver_feat["primary_reach_2"], receiver_feat["opposite_reach_2"])
    # Reward both endpoints for favoring their primary directions.
    # sender primary=out, receiver primary=in
    return 0.5 * clamp01((s_bias + 1.0) / 2.0) + 0.5 * clamp01((r_bias + 1.0) / 2.0)


def anti_balance(sender_feat, receiver_feat):
    # Reward sender/receiver role asymmetry.
    s_asym = asymmetry_score(sender_feat["primary_deg"], sender_feat["secondary_deg"])
    r_asym = asymmetry_score(receiver_feat["primary_deg"], receiver_feat["secondary_deg"])
    return 0.5 * s_asym + 0.5 * r_asym


def symmetry_score(sender_feat, receiver_feat):
    # For cycles, reward bidirectional reachability and balance.
    s_bal = 1.0 - asymmetry_score(sender_feat["primary_deg"], sender_feat["secondary_deg"])
    r_bal = 1.0 - asymmetry_score(receiver_feat["primary_deg"], receiver_feat["secondary_deg"])

    s_reach_bal = 1.0 - abs(
        direction_bias_score(sender_feat["primary_reach_2"], sender_feat["opposite_reach_2"])
    )
    r_reach_bal = 1.0 - abs(
        direction_bias_score(receiver_feat["primary_reach_2"], receiver_feat["opposite_reach_2"])
    )

    return 0.25 * s_bal + 0.25 * r_bal + 0.25 * s_reach_bal + 0.25 * r_reach_bal


def flow_compat(sender_feat, receiver_feat):
    # Compare amount buckets.
    a = bucket_closeness(sender_feat["amount_paid_bucket"], receiver_feat["amount_paid_bucket"])
    b = bucket_closeness(sender_feat["amount_recv_bucket"], receiver_feat["amount_recv_bucket"])
    return 0.5 * a + 0.5 * b


def context_compat(sender_feat, receiver_feat):
    # Compare time, payment format, and currency context.
    time_score = bucket_closeness(sender_feat["hour_bucket"], receiver_feat["hour_bucket"])
    fmt_score = categorical_match(sender_feat["payment_format"], receiver_feat["payment_format"])
    recv_ccy_score = categorical_match(sender_feat["recv_currency"], receiver_feat["recv_currency"])
    paid_ccy_score = categorical_match(sender_feat["paid_currency"], receiver_feat["paid_currency"])
    ccy_score = 0.5 * recv_ccy_score + 0.5 * paid_ccy_score
    return 0.4 * time_score + 0.2 * fmt_score + 0.4 * ccy_score


def degree_scale_compat(sender_feat, receiver_feat):
    # Use degree-scale proximity conservatively to avoid over-rewarding ordinary edges.
    return 0.5 * bucket_closeness(
        sender_feat["primary_deg_bucket"], receiver_feat["primary_deg_bucket"]
    ) + 0.5 * bucket_closeness(
        sender_feat["secondary_deg_bucket"], receiver_feat["secondary_deg_bucket"]
    )


def chain_hypothesis_score(sender_feat, receiver_feat):
    """Score a chain-like pair using role bias, asymmetry, flow, and context."""
    rb = role_balance(sender_feat, receiver_feat)
    asym = anti_balance(sender_feat, receiver_feat)
    flow = flow_compat(sender_feat, receiver_feat)
    ctx = context_compat(sender_feat, receiver_feat)

    score = 0.35 * rb + 0.25 * asym + 0.25 * flow + 0.15 * ctx
    return float(score)


def fan_hypothesis_score(sender_feat, receiver_feat):
    """Score a fan-like pair using endpoint asymmetry, direction, flow, and context."""
    s_asym = asymmetry_score(sender_feat["primary_deg"], sender_feat["secondary_deg"])
    r_asym = asymmetry_score(receiver_feat["primary_deg"], receiver_feat["secondary_deg"])
    max_asym = max(s_asym, r_asym)

    s_dir = clamp01((direction_bias_score(sender_feat["primary_reach_2"], sender_feat["opposite_reach_2"]) + 1.0) / 2.0)
    r_dir = clamp01((direction_bias_score(receiver_feat["primary_reach_2"], receiver_feat["opposite_reach_2"]) + 1.0) / 2.0)
    dir_score = 0.5 * s_dir + 0.5 * r_dir

    flow = flow_compat(sender_feat, receiver_feat)
    ctx = context_compat(sender_feat, receiver_feat)

    score = 0.40 * max_asym + 0.25 * dir_score + 0.20 * flow + 0.15 * ctx
    return float(score)


def cycle_hypothesis_score(sender_feat, receiver_feat):
    """Score a cycle-like pair using symmetry, degree scale, flow, and context."""
    sym = symmetry_score(sender_feat, receiver_feat)
    flow = flow_compat(sender_feat, receiver_feat)
    ctx = context_compat(sender_feat, receiver_feat)
    deg = degree_scale_compat(sender_feat, receiver_feat)

    score = 0.40 * sym + 0.20 * deg + 0.20 * flow + 0.20 * ctx
    return float(score)


def compute_hypothesis_scores(sender_feat, receiver_feat):
    chain_score = chain_hypothesis_score(sender_feat, receiver_feat)
    fan_score = fan_hypothesis_score(sender_feat, receiver_feat)
    cycle_score = cycle_hypothesis_score(sender_feat, receiver_feat)

    scores = {
        "CHAIN": chain_score,
        "FAN": fan_score,
        "CYCLE": cycle_score,
    }
    best_hypothesis = max(scores, key=scores.get)
    compat_score = scores[best_hypothesis]
    return scores, best_hypothesis, compat_score


class StreamingOrchestrator:
    def __init__(
        self,
        evidence_dir: str,
        output_dir: str,
        threshold: float = 0.72,
        max_group_size: int = 5,
        topk_per_sender: int = 1,
        topk_per_receiver: int = 1,
    ):
        self.evidence_dir = evidence_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.max_group_size = max_group_size
        self.topk_per_sender = topk_per_sender
        self.topk_per_receiver = topk_per_receiver
        os.makedirs(self.output_dir, exist_ok=True)

    def _iter_evidence_files(self):
        for fn in sorted(os.listdir(self.evidence_dir)):
            if fn.endswith("_evidences.csv"):
                yield os.path.join(self.evidence_dir, fn)

    def _load_tag_buckets(self):
        tag_buckets = defaultdict(lambda: {"SENDER": [], "RECEIVER": []})
        total_rows = 0

        for fp in self._iter_evidence_files():
            print(f"[*] Reading evidence file: {fp}")
            with open(fp, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_rows += 1
                    tag = row["public_tx_tag"]
                    role = row["role"]

                    if role not in ("SENDER", "RECEIVER"):
                        continue

                    if len(tag_buckets[tag][role]) < self.max_group_size:
                        tag_buckets[tag][role].append({
                            "bank_id": row["bank_id"],
                            "tx_id": int(row["tx_id"]),
                            "evidence_node_id": row["evidence_node_id"],
                            "local_account_id": row["local_account_id"],
                            "motif_type": row.get("motif_type", ""),
                            "motif_instance_id": row.get("motif_instance_id", ""),
                            "feature_json": row["feature_json"],
                        })

            print(f"    current unique tags in memory: {len(tag_buckets)}")

        return tag_buckets, {"total_rows_read": total_rows, "num_tags_loaded": len(tag_buckets)}

    def run(self):
        tag_buckets, stats = self._load_tag_buckets()

        scored_path = os.path.join(self.output_dir, "alignment_pairs_scored.csv")
        edge_path = os.path.join(self.output_dir, "compatibility_graph_edges.csv")
        node_path = os.path.join(self.output_dir, "compatibility_graph_nodes.csv")
        stats_path = os.path.join(self.output_dir, "orchestrator_stats.json")

        node_seen = set()
        total_candidate_pairs = 0
        total_scored_pairs = 0
        kept_edges = 0
        valid_tags = 0

        with open(scored_path, "w", encoding="utf-8", newline="") as scored_f, \
             open(edge_path, "w", encoding="utf-8", newline="") as edge_f, \
             open(node_path, "w", encoding="utf-8", newline="") as node_f:

            scored_writer = csv.writer(scored_f)
            edge_writer = csv.writer(edge_f)
            node_writer = csv.writer(node_f)

            scored_writer.writerow([
                "public_tx_tag",
                "bank_id_s", "bank_id_r",
                "evidence_node_id_s", "evidence_node_id_r",
                "local_account_id_s", "local_account_id_r",
                "tx_id_s", "tx_id_r",
                "chain_score", "fan_score", "cycle_score",
                "compat_score", "best_hypothesis",
                "motif_type_s", "motif_type_r",
                "motif_instance_id_s", "motif_instance_id_r",
            ])

            edge_writer.writerow([
                "src_node", "dst_node", "weight",
                "best_hypothesis",
                "public_tx_tag", "tx_id_sender", "tx_id_receiver"
            ])

            node_writer.writerow([
                "node_id", "bank_id", "local_account_id"
            ])

            for idx, (tag, group) in enumerate(tag_buckets.items()):
                senders = group["SENDER"]
                receivers = group["RECEIVER"]

                if len(senders) == 0 or len(receivers) == 0:
                    continue

                valid_tags += 1
                pair_rows = []

                # 1) score all candidate pairs
                for s in senders:
                    s_feat = parse_feature_json(s["feature_json"])
                    for r in receivers:
                        if s["bank_id"] == r["bank_id"]:
                            continue

                        r_feat = parse_feature_json(r["feature_json"])
                        total_candidate_pairs += 1

                        scores, best_hypothesis, compat_score = compute_hypothesis_scores(s_feat, r_feat)
                        total_scored_pairs += 1

                        pair = {
                            "tag": tag,
                            "bank_id_s": s["bank_id"],
                            "bank_id_r": r["bank_id"],
                            "evidence_node_id_s": s["evidence_node_id"],
                            "evidence_node_id_r": r["evidence_node_id"],
                            "local_account_id_s": s["local_account_id"],
                            "local_account_id_r": r["local_account_id"],
                            "tx_id_s": s["tx_id"],
                            "tx_id_r": r["tx_id"],
                            "chain_score": scores["CHAIN"],
                            "fan_score": scores["FAN"],
                            "cycle_score": scores["CYCLE"],
                            "compat_score": compat_score,
                            "best_hypothesis": best_hypothesis,
                            "motif_type_s": s["motif_type"],
                            "motif_type_r": r["motif_type"],
                            "motif_instance_id_s": s["motif_instance_id"],
                            "motif_instance_id_r": r["motif_instance_id"],
                        }
                        pair_rows.append(pair)

                        scored_writer.writerow([
                            tag,
                            s["bank_id"], r["bank_id"],
                            s["evidence_node_id"], r["evidence_node_id"],
                            s["local_account_id"], r["local_account_id"],
                            s["tx_id"], r["tx_id"],
                            scores["CHAIN"], scores["FAN"], scores["CYCLE"],
                            compat_score, best_hypothesis,
                            s["motif_type"], r["motif_type"],
                            s["motif_instance_id"], r["motif_instance_id"],
                        ])

                # 2) keep only top-k per sender and top-k per receiver
                sender_to_pairs = defaultdict(list)
                receiver_to_pairs = defaultdict(list)

                for p in pair_rows:
                    sender_to_pairs[p["evidence_node_id_s"]].append(p)
                    receiver_to_pairs[p["evidence_node_id_r"]].append(p)

                sender_keep = set()
                for sid, plist in sender_to_pairs.items():
                    plist_sorted = sorted(plist, key=lambda x: x["compat_score"], reverse=True)
                    for p in plist_sorted[:self.topk_per_sender]:
                        if p["compat_score"] >= self.threshold:
                            sender_keep.add((p["evidence_node_id_s"], p["evidence_node_id_r"], p["tx_id_s"], p["tx_id_r"]))

                receiver_keep = set()
                for rid, plist in receiver_to_pairs.items():
                    plist_sorted = sorted(plist, key=lambda x: x["compat_score"], reverse=True)
                    for p in plist_sorted[:self.topk_per_receiver]:
                        if p["compat_score"] >= self.threshold:
                            receiver_keep.add((p["evidence_node_id_s"], p["evidence_node_id_r"], p["tx_id_s"], p["tx_id_r"]))

                final_keep = sender_keep & receiver_keep

                # 3) write final edges
                for p in pair_rows:
                    key = (p["evidence_node_id_s"], p["evidence_node_id_r"], p["tx_id_s"], p["tx_id_r"])
                    if key not in final_keep:
                        continue

                    edge_writer.writerow([
                        p["evidence_node_id_s"],
                        p["evidence_node_id_r"],
                        p["compat_score"],
                        p["best_hypothesis"],
                        p["tag"],
                        p["tx_id_s"],
                        p["tx_id_r"],
                    ])
                    kept_edges += 1

                    if p["evidence_node_id_s"] not in node_seen:
                        node_writer.writerow([
                            p["evidence_node_id_s"],
                            p["bank_id_s"],
                            p["local_account_id_s"],
                        ])
                        node_seen.add(p["evidence_node_id_s"])

                    if p["evidence_node_id_r"] not in node_seen:
                        node_writer.writerow([
                            p["evidence_node_id_r"],
                            p["bank_id_r"],
                            p["local_account_id_r"],
                        ])
                        node_seen.add(p["evidence_node_id_r"])

                if (idx + 1) % 5000 == 0:
                    print(
                        f"[*] processed {idx+1} tags | "
                        f"valid_tags={valid_tags} | "
                        f"candidate_pairs={total_candidate_pairs} | "
                        f"scored_pairs={total_scored_pairs} | "
                        f"kept_edges={kept_edges}"
                    )

        stats.update({
            "valid_tags_with_both_roles": valid_tags,
            "total_candidate_pairs": total_candidate_pairs,
            "total_scored_pairs": total_scored_pairs,
            "kept_edges": kept_edges,
            "kept_nodes": len(node_seen),
            "threshold": self.threshold,
            "max_group_size": self.max_group_size,
            "topk_per_sender": self.topk_per_sender,
            "topk_per_receiver": self.topk_per_receiver,
        })

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="semotif_orchestrator")
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--max_group_size", type=int, default=5)
    parser.add_argument("--topk_per_sender", type=int, default=1)
    parser.add_argument("--topk_per_receiver", type=int, default=1)
    args = parser.parse_args()

    orchestrator = StreamingOrchestrator(
        evidence_dir=args.evidence_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        max_group_size=args.max_group_size,
        topk_per_sender=args.topk_per_sender,
        topk_per_receiver=args.topk_per_receiver,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
