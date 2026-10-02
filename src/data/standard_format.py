import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


STANDARD_FILES = [
    "nodes.csv",
    "edges.csv",
    "node_features.csv",
    "edge_features.csv",
    "labels.csv",
    "partitions.json",
    "query_motifs.json",
    "ground_truth.json",
    "metadata.json",
]

NODES_COLUMNS = ["node_id", "original_id", "node_type", "label"]
EDGES_COLUMNS = ["edge_id", "src", "dst", "timestamp", "weight", "edge_type", "label"]
LABELS_COLUMNS = ["object_id", "object_type", "label", "label_name"]
NODE_FEATURE_COLUMNS = ["node_id"]
EDGE_FEATURE_COLUMNS = ["edge_id"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def _json_list(value) -> List:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    return json.loads(value)


def _edge_label(row) -> int:
    try:
        return int(row.get("Is Laundering", 0))
    except Exception:
        return 0


def _optional_float(value):
    try:
        return float(value)
    except Exception:
        return None


def export_amlworld_standard_format(
    partition_dir: Path,
    output_dir: Path,
    dataset_name: str = "amlworld_hi_small",
    source: str = "AMLWorld HI-Small",
    citation: str = "",
    chunksize: int = 250_000,
) -> Dict[str, Path]:
    """Export AMLWorld partition outputs to the CoMot standard format."""
    partition_dir = Path(partition_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    tx_path = partition_dir / "transactions_partitioned.csv"
    accounts_path = partition_dir / "ground_truth_accounts.csv"
    motifs_path = partition_dir / "ground_truth_motifs.csv"

    accounts_df = _read_csv(accounts_path, dtype={"account_id": "string"})
    motifs_df = _read_csv(motifs_path)

    export_nodes(accounts_df, output_dir / "nodes.csv")
    export_node_features(accounts_df, output_dir / "node_features.csv")
    export_edges_and_edge_features(tx_path, output_dir / "edges.csv", output_dir / "edge_features.csv", chunksize)
    export_labels(accounts_df, tx_path, motifs_df, output_dir / "labels.csv", chunksize)
    export_partitions(partition_dir, accounts_df, output_dir / "partitions.json", chunksize)
    export_amlworld_queries_and_ground_truth(
        motifs_df,
        partition_dir,
        output_dir / "query_motifs.json",
        output_dir / "ground_truth.json",
    )
    export_metadata(
        output_dir / "metadata.json",
        dataset_name=dataset_name,
        source=source,
        citation=citation,
        num_nodes=int(len(accounts_df)),
        num_motifs=int(len(motifs_df)),
    )

    return {name: output_dir / name for name in STANDARD_FILES}


def export_nodes(accounts_df: pd.DataFrame, output_path: Path) -> None:
    out = pd.DataFrame({
        "node_id": accounts_df["account_id"].astype(str),
        "original_id": accounts_df["account_id"].astype(str),
        "node_type": "account",
        "label": accounts_df.get("is_illicit", "").astype(str),
    })
    out.to_csv(output_path, index=False, columns=NODES_COLUMNS)


def export_node_features(accounts_df: pd.DataFrame, output_path: Path) -> None:
    pd.DataFrame({"node_id": accounts_df["account_id"].astype(str)}).to_csv(
        output_path, index=False, columns=NODE_FEATURE_COLUMNS
    )


def export_edges_and_edge_features(
    tx_path: Path,
    edges_path: Path,
    edge_features_path: Path,
    chunksize: int,
) -> None:
    first = True
    for chunk in pd.read_csv(tx_path, chunksize=chunksize):
        edges = pd.DataFrame({
            "edge_id": chunk["tx_id"].astype(str),
            "src": chunk["Account"].astype(str),
            "dst": chunk["Account.1"].astype(str),
            "timestamp": chunk.get("Timestamp", ""),
            "weight": chunk.get("Amount Paid", "").map(_optional_float),
            "edge_type": chunk.get("tx_group", chunk.get("Payment Format", "")),
            "label": chunk.apply(_edge_label, axis=1),
        })
        edges.to_csv(edges_path, index=False, mode="w" if first else "a", header=first, columns=EDGES_COLUMNS)

        edge_features = pd.DataFrame({"edge_id": chunk["tx_id"].astype(str)})
        for col in [
            "Amount Received",
            "Receiving Currency",
            "Amount Paid",
            "Payment Currency",
            "Payment Format",
            "public_tx_tag",
            "is_inter_partition",
        ]:
            if col in chunk.columns:
                edge_features[col] = chunk[col]
        edge_features.to_csv(
            edge_features_path,
            index=False,
            mode="w" if first else "a",
            header=first,
        )
        first = False


def export_labels(
    accounts_df: pd.DataFrame,
    tx_path: Path,
    motifs_df: pd.DataFrame,
    output_path: Path,
    chunksize: int,
) -> None:
    node_labels = pd.DataFrame({
        "object_id": accounts_df["account_id"].astype(str),
        "object_type": "node",
        "label": accounts_df.get("is_illicit", 0).astype(str),
        "label_name": accounts_df.get("is_illicit", 0).map(lambda x: "illicit" if int(x) == 1 else "benign"),
    })
    node_labels.to_csv(output_path, index=False, columns=LABELS_COLUMNS)

    for chunk in pd.read_csv(tx_path, chunksize=chunksize):
        edge_labels = pd.DataFrame({
            "object_id": chunk["tx_id"].astype(str),
            "object_type": "edge",
            "label": chunk.get("Is Laundering", 0).astype(str),
            "label_name": chunk.get("motif_type", ""),
        })
        edge_labels.to_csv(output_path, index=False, mode="a", header=False, columns=LABELS_COLUMNS)

    motif_labels = pd.DataFrame({
        "object_id": motifs_df["motif_instance_id"].astype(str),
        "object_type": "motif",
        "label": motifs_df["motif_type"].astype(str),
        "label_name": motifs_df["motif_type"].astype(str),
    })
    motif_labels.to_csv(output_path, index=False, mode="a", header=False, columns=LABELS_COLUMNS)


def export_partitions(
    partition_dir: Path,
    accounts_df: pd.DataFrame,
    output_path: Path,
    chunksize: int,
) -> None:
    partition_ids = sorted(accounts_df["bank_partition"].dropna().astype(str).unique().tolist())
    visible_nodes: Dict[str, Set[str]] = {pid: set() for pid in partition_ids}
    visible_edges: Dict[str, List[str]] = {pid: [] for pid in partition_ids}

    for _, row in accounts_df.iterrows():
        bank = str(row["bank_partition"])
        if bank in visible_nodes:
            visible_nodes[bank].add(str(row["account_id"]))

    tx_path = partition_dir / "transactions_partitioned.csv"
    for chunk in pd.read_csv(tx_path, chunksize=chunksize):
        values = zip(
            chunk["tx_id"],
            chunk["src_bank_partition"],
            chunk["dst_bank_partition"],
            chunk["Account"],
            chunk["Account.1"],
        )
        for tx_id, src_bank_value, dst_bank_value, src_value, dst_value in values:
            edge_id = str(tx_id)
            src_bank = str(src_bank_value)
            dst_bank = str(dst_bank_value)
            src = str(src_value)
            dst = str(dst_value)
            if src_bank in visible_edges:
                visible_edges[src_bank].append(edge_id)
                visible_nodes[src_bank].add(src)
                visible_nodes[src_bank].add(dst)
            if dst_bank != src_bank and dst_bank in visible_edges:
                visible_edges[dst_bank].append(edge_id)
                visible_nodes[dst_bank].add(src)
                visible_nodes[dst_bank].add(dst)

    payload = []
    for pid in partition_ids:
        payload.append({
            "partition_id": pid,
            "visible_nodes": sorted(visible_nodes[pid]),
            "visible_edges": visible_edges[pid],
        })
    write_json(output_path, payload)


def export_queries(motifs_df: pd.DataFrame, output_path: Path) -> None:
    raise RuntimeError("export_queries requires transaction topology. Use export_amlworld_queries_and_ground_truth.")


def export_ground_truth(motifs_df: pd.DataFrame, partition_dir: Path, output_path: Path) -> None:
    raise RuntimeError("export_ground_truth requires transaction topology. Use export_amlworld_queries_and_ground_truth.")


def export_amlworld_queries_and_ground_truth(
    motifs_df: pd.DataFrame,
    partition_dir: Path,
    query_output_path: Path,
    gt_output_path: Path,
) -> None:
    tx_lookup = build_transaction_lookup(partition_dir / "transactions_partitioned.csv")
    queries = []
    ground_truth = []
    for _, row in motifs_df.iterrows():
        motif_instance_id = str(row["motif_instance_id"])
        motif_type = str(row["motif_type"])
        edge_ids = [str(x) for x in _json_list(row["tx_ids_json"])]
        tx_rows = [tx_lookup[edge_id] for edge_id in edge_ids if edge_id in tx_lookup]
        true_nodes = sorted({tx["src"] for tx in tx_rows} | {tx["dst"] for tx in tx_rows})
        true_edges = [tx["edge_id"] for tx in tx_rows]
        true_partitions = sorted({p for tx in tx_rows for p in [tx["src_bank"], tx["dst_bank"]] if p})
        node_to_query = {node_id: f"q{idx}" for idx, node_id in enumerate(true_nodes)}
        query_edges = [
            {
                "src": node_to_query[tx["src"]],
                "dst": node_to_query[tx["dst"]],
                "edge_id": tx["edge_id"],
            }
            for tx in tx_rows
        ]
        hidden_edges = [
            tx["edge_id"]
            for tx in tx_rows
            if tx["src_bank"] and tx["dst_bank"] and tx["src_bank"] != tx["dst_bank"]
        ]
        if not hidden_edges:
            hidden_edges = true_edges[:1]
        hidden_edge_set = set(hidden_edges)
        hidden_nodes = sorted({
            node
            for tx in tx_rows
            if tx["edge_id"] in hidden_edge_set
            for node in [tx["src"], tx["dst"]]
        })
        anchor_nodes = sorted(set(true_nodes) - set(hidden_nodes))
        if not anchor_nodes and true_nodes:
            anchor_nodes = [true_nodes[0]]
            hidden_nodes = sorted(set(true_nodes) - set(anchor_nodes))

        queries.append({
            "query_id": motif_instance_id,
            "motif_type": motif_type,
            "nodes": [node_to_query[node_id] for node_id in true_nodes],
            "edges": query_edges,
            "anchor_nodes": anchor_nodes,
            "hidden_nodes": hidden_nodes,
            "hidden_edges": hidden_edges,
            "node_mapping": {query_node: node_id for node_id, query_node in node_to_query.items()},
            "description": (
                f"AMLWorld {motif_type} motif instance {motif_instance_id}; "
                "query topology is anonymized from laundering-pattern transactions, "
                "anchor_nodes are observed real account IDs, hidden_edges are cross-partition transactions."
            ),
        })
        ground_truth.append({
            "query_id": motif_instance_id,
            "true_nodes": true_nodes,
            "true_edges": true_edges,
            "true_partitions": true_partitions,
            "motif_instance_id": motif_instance_id,
        })
    write_json(query_output_path, queries)
    write_json(gt_output_path, ground_truth)


def build_transaction_lookup(tx_path: Path, chunksize: int = 250_000) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    usecols = ["tx_id", "Account", "Account.1", "src_bank_partition", "dst_bank_partition"]
    for chunk in pd.read_csv(tx_path, usecols=usecols, chunksize=chunksize):
        for row in chunk.to_dict("records"):
            edge_id = str(row["tx_id"])
            lookup[edge_id] = {
                "edge_id": edge_id,
                "src": str(row["Account"]),
                "dst": str(row["Account.1"]),
                "src_bank": str(row["src_bank_partition"]),
                "dst_bank": str(row["dst_bank_partition"]),
            }
    return lookup


def build_edge_partition_lookup(tx_path: Path, chunksize: int = 250_000) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    usecols = ["tx_id", "src_bank_partition", "dst_bank_partition"]
    for chunk in pd.read_csv(tx_path, usecols=usecols, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            banks = [str(row.src_bank_partition)]
            if str(row.dst_bank_partition) != str(row.src_bank_partition):
                banks.append(str(row.dst_bank_partition))
            lookup[str(row.tx_id)] = banks
    return lookup


def export_metadata(
    output_path: Path,
    dataset_name: str,
    source: str,
    citation: str,
    num_nodes: int,
    num_motifs: int,
) -> None:
    payload = {
        "dataset_name": dataset_name,
        "graph_type": "transaction",
        "directed": True,
        "weighted": True,
        "has_timestamps": True,
        "has_node_features": True,
        "has_edge_features": True,
        "has_ground_truth": True,
        "source": source,
        "citation": citation,
        "num_nodes": num_nodes,
        "num_motif_instances": num_motifs,
        "format_version": "1.0",
    }
    write_json(output_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a dataset into the CoMot standard processed format.")
    parser.add_argument("--dataset", required=True, choices=["amlworld"])
    parser.add_argument("--partition_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", default="amlworld_hi_small")
    parser.add_argument("--source", default="AMLWorld HI-Small")
    parser.add_argument("--citation", default="")
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "amlworld":
        outputs = export_amlworld_standard_format(
            partition_dir=Path(args.partition_dir),
            output_dir=Path(args.output_dir),
            dataset_name=args.dataset_name,
            source=args.source,
            citation=args.citation,
            chunksize=args.chunksize,
        )
        print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
