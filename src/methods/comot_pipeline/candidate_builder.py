import os
import json
import argparse
from collections import defaultdict

import pandas as pd
import networkx as nx


def dedup_candidate_signature(nodes, tx_ids, cand_type):
    nodes_key = tuple(sorted(set(nodes)))
    tx_key = tuple(sorted(set(tx_ids)))
    return (cand_type, nodes_key, tx_key)


class MotifCandidateBuilder:
    def __init__(
        self,
        orchestrator_dir: str,
        output_dir: str,
        chain_min_len: int = 3,
        chain_max_len: int = 10,
        cycle_min_len: int = 2,
        cycle_max_len: int = 8,
        fan_degree_thr: int = 3,
    ):
        self.orchestrator_dir = orchestrator_dir
        self.output_dir = output_dir
        self.chain_min_len = chain_min_len
        self.chain_max_len = chain_max_len
        self.cycle_min_len = cycle_min_len
        self.cycle_max_len = cycle_max_len
        self.fan_degree_thr = fan_degree_thr

        os.makedirs(self.output_dir, exist_ok=True)

    def load_edges(self):
        edge_path = os.path.join(self.orchestrator_dir, "compatibility_graph_edges.csv")
        if not os.path.exists(edge_path):
            raise FileNotFoundError(edge_path)
        return pd.read_csv(edge_path)

    def build_graph_from_edges(self, df: pd.DataFrame) -> nx.DiGraph:
        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_edge(
                row["src_node"],
                row["dst_node"],
                weight=float(row["weight"]),
                best_hypothesis=row.get("best_hypothesis", "UNKNOWN"),
                public_tx_tag=row["public_tx_tag"],
                tx_id_sender=int(row["tx_id_sender"]),
                tx_id_receiver=int(row["tx_id_receiver"]),
            )
        return G

    def extract_chain_candidates(self, df: pd.DataFrame):
        sub_df = df[df["best_hypothesis"] == "CHAIN"].copy()
        G = self.build_graph_from_edges(sub_df)

        candidates = []
        seen = set()
        cid = 0

        for comp in nx.weakly_connected_components(G):
            sub = G.subgraph(comp).copy()
            if sub.number_of_nodes() < self.chain_min_len:
                continue

            # Approximate a DAG by retaining high-weight edges that do not create cycles.
            dag = nx.DiGraph()
            dag.add_nodes_from(sub.nodes(data=True))
            for u, v, d in sorted(sub.edges(data=True), key=lambda x: (-x[2].get("weight", 0.0), x[0], x[1])):
                if not nx.has_path(dag, v, u):
                    dag.add_edge(u, v, **d)

            if dag.number_of_edges() == 0:
                continue

            try:
                path = nx.dag_longest_path(dag)
            except Exception:
                continue

            if not (self.chain_min_len <= len(path) <= self.chain_max_len):
                continue

            tx_ids = []
            edge_scores = []
            for i in range(len(path) - 1):
                d = dag[path[i]][path[i + 1]]
                tx_ids.extend([d["tx_id_sender"], d["tx_id_receiver"]])
                edge_scores.append(float(d["weight"]))

            sig = dedup_candidate_signature(path, tx_ids, "CHAIN")
            if sig in seen:
                continue
            seen.add(sig)

            candidates.append({
                "candidate_id": f"CHAIN_{cid:06d}",
                "candidate_type": "CHAIN",
                "num_nodes": len(path),
                "num_edges": len(path) - 1,
                "score_mean": sum(edge_scores) / len(edge_scores) if edge_scores else 0.0,
                "score_min": min(edge_scores) if edge_scores else 0.0,
                "nodes_json": json.dumps(path),
                "tx_ids_json": json.dumps(sorted(set(tx_ids))),
            })
            cid += 1

        return pd.DataFrame(candidates)

    def extract_fan_candidates(self, df: pd.DataFrame):
        sub_df = df[df["best_hypothesis"] == "FAN"].copy()
        G = self.build_graph_from_edges(sub_df)

        candidates = []
        seen = set()
        cid = 0

        for node in G.nodes():
            indeg = G.in_degree(node)
            outdeg = G.out_degree(node)

            if indeg >= self.fan_degree_thr:
                preds = list(G.predecessors(node))
                tx_ids = []
                edge_scores = []
                for p in preds:
                    d = G[p][node]
                    tx_ids.extend([d["tx_id_sender"], d["tx_id_receiver"]])
                    edge_scores.append(float(d["weight"]))

                nodes = [node] + preds
                sig = dedup_candidate_signature(nodes, tx_ids, "FAN_IN")
                if sig not in seen:
                    seen.add(sig)
                    candidates.append({
                        "candidate_id": f"FANIN_{cid:06d}",
                        "candidate_type": "FAN_IN",
                        "num_nodes": len(nodes),
                        "num_edges": len(preds),
                        "score_mean": sum(edge_scores) / len(edge_scores) if edge_scores else 0.0,
                        "score_min": min(edge_scores) if edge_scores else 0.0,
                        "nodes_json": json.dumps(nodes),
                        "tx_ids_json": json.dumps(sorted(set(tx_ids))),
                    })
                    cid += 1

            if outdeg >= self.fan_degree_thr:
                succs = list(G.successors(node))
                tx_ids = []
                edge_scores = []
                for s in succs:
                    d = G[node][s]
                    tx_ids.extend([d["tx_id_sender"], d["tx_id_receiver"]])
                    edge_scores.append(float(d["weight"]))

                nodes = [node] + succs
                sig = dedup_candidate_signature(nodes, tx_ids, "FAN_OUT")
                if sig not in seen:
                    seen.add(sig)
                    candidates.append({
                        "candidate_id": f"FANOUT_{cid:06d}",
                        "candidate_type": "FAN_OUT",
                        "num_nodes": len(nodes),
                        "num_edges": len(succs),
                        "score_mean": sum(edge_scores) / len(edge_scores) if edge_scores else 0.0,
                        "score_min": min(edge_scores) if edge_scores else 0.0,
                        "nodes_json": json.dumps(nodes),
                        "tx_ids_json": json.dumps(sorted(set(tx_ids))),
                    })
                    cid += 1

        return pd.DataFrame(candidates)

    def extract_cycle_candidates(self, df: pd.DataFrame):
        sub_df = df[df["best_hypothesis"] == "CYCLE"].copy()
        G = self.build_graph_from_edges(sub_df)

        candidates = []
        seen = set()
        cid = 0

        try:
            all_cycles = list(nx.simple_cycles(G))
        except Exception:
            all_cycles = []

        for cyc in all_cycles:
            if not (self.cycle_min_len <= len(cyc) <= self.cycle_max_len):
                continue

            tx_ids = []
            edge_scores = []

            valid = True
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                if not G.has_edge(u, v):
                    valid = False
                    break
                d = G[u][v]
                tx_ids.extend([d["tx_id_sender"], d["tx_id_receiver"]])
                edge_scores.append(float(d["weight"]))

            if not valid:
                continue

            sig = dedup_candidate_signature(cyc, tx_ids, "CYCLE")
            if sig in seen:
                continue
            seen.add(sig)

            candidates.append({
                "candidate_id": f"CYCLE_{cid:06d}",
                "candidate_type": "CYCLE",
                "num_nodes": len(cyc),
                "num_edges": len(cyc),
                "score_mean": sum(edge_scores) / len(edge_scores) if edge_scores else 0.0,
                "score_min": min(edge_scores) if edge_scores else 0.0,
                "nodes_json": json.dumps(cyc),
                "tx_ids_json": json.dumps(sorted(set(tx_ids))),
            })
            cid += 1

        return pd.DataFrame(candidates)

    def run(self):
        df = self.load_edges()

        chain_df = self.extract_chain_candidates(df)
        fan_df = self.extract_fan_candidates(df)
        cycle_df = self.extract_cycle_candidates(df)

        chain_df.to_csv(os.path.join(self.output_dir, "chain_candidates.csv"), index=False)
        fan_df.to_csv(os.path.join(self.output_dir, "fan_candidates.csv"), index=False)
        cycle_df.to_csv(os.path.join(self.output_dir, "cycle_candidates.csv"), index=False)

        summary = {
            "num_chain_candidates": int(len(chain_df)),
            "num_fan_candidates": int(len(fan_df)),
            "num_cycle_candidates": int(len(cycle_df)),
        }
        with open(os.path.join(self.output_dir, "candidate_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orchestrator_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="semotif_candidates")
    parser.add_argument("--chain_min_len", type=int, default=3)
    parser.add_argument("--chain_max_len", type=int, default=10)
    parser.add_argument("--cycle_min_len", type=int, default=2)
    parser.add_argument("--cycle_max_len", type=int, default=8)
    parser.add_argument("--fan_degree_thr", type=int, default=3)
    args = parser.parse_args()

    builder = MotifCandidateBuilder(
        orchestrator_dir=args.orchestrator_dir,
        output_dir=args.output_dir,
        chain_min_len=args.chain_min_len,
        chain_max_len=args.chain_max_len,
        cycle_min_len=args.cycle_min_len,
        cycle_max_len=args.cycle_max_len,
        fan_degree_thr=args.fan_degree_thr,
    )
    builder.run()


if __name__ == "__main__":
    main()
