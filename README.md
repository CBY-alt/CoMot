# CoMot

CoMot retrieves cross-partition graph motifs under partial graph observability. This artifact contains the preprocessing, partition/query construction, submitted CoMot implementation, 19 task-adapted baselines, and common evaluation code used in the paper. Datasets and generated results are intentionally excluded.

## Layout

```text
CoMot/
├── README.md
├── requirements.txt
├── configs/
│   ├── amlworld_hi_small.json
│   ├── dblp.yaml
│   ├── elliptic.yaml
│   ├── web_google.yaml
│   ├── methods/comot_standard.yaml
│   └── baselines/
├── scripts/
│   ├── preprocess_amlworld.sh
│   ├── run.sh
│   ├── run_all.sh
│   └── run_baselines.sh
├── src/
│   ├── data/
│   ├── methods/
│   │   ├── comot.py
│   │   ├── comot_pipeline/
│   │   └── baselines/
│   ├── evaluation/
│   ├── amlworld_pipeline.py
│   ├── run.py
│   └── run_all.py
└── data/
    ├── raw/
    └── processed/
```

## Installation

```bash
pip install -r requirements.txt
```

## Data preprocessing

The common artifact flow is:

```text
public raw dataset
  -> processed nodes/edges/features/labels
  -> five-party partition view
  -> query_motifs.json + ground_truth.json
  -> method_queries.json (evaluation-only GT fields removed)
  -> CoMot or baseline candidates
  -> common evaluation
```

Place the public source files as follows:

```text
data/raw/amlworld/HI-Small_Trans.csv
data/raw/amlworld/HI-Small_Patterns.txt
data/raw/elliptic/elliptic_bitcoin_dataset/elliptic_txs_classes.csv
data/raw/elliptic/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv
data/raw/elliptic/elliptic_bitcoin_dataset/elliptic_txs_features.csv
data/raw/dblp/com-dblp.ungraph.txt
data/raw/dblp/com-dblp.top5000.cmty.txt
data/raw/web_google/web-Google.txt
```

Obtain these files from their public providers and comply with their licenses.

### AMLWorld

The submitted AMLWorld chain constructs the global graph, controlled five-party partition, partition summary, local evidence, cross-party alignment, motif candidates, reranking, and evaluation:

```bash
bash scripts/preprocess_amlworld.sh
```

The implementation is split between `src/data/amlworld_*.py`, `src/methods/comot_pipeline/`, and `src/evaluation/`.

### Elliptic, DBLP, and Web-Google

```bash
python -m src.data.elliptic --raw_dir data/raw/elliptic --processed_dir data/processed/elliptic
python -m src.data.elliptic_protocol --dataset_dir data/processed/elliptic

python -m src.data.dblp --raw_dir data/raw/dblp --processed_dir data/processed/dblp
python -m src.data.web_google --raw_dir data/raw/web_google --processed_dir data/processed/web_google
python -m src.data.structural_protocol --dataset all --seed 42

python -m src.data.query_view all
python -m src.data.validate_dataset --dataset_dir data/processed/elliptic
```

The protocol builders write processed files. Use a fresh `data/processed/` tree for a clean reconstruction.

### Processed-data contract

Each `data/processed/<dataset>/` directory contains:

```text
nodes.csv                 node_id, original_id, optional type/label
edges.csv                 edge_id, src, dst, optional time/weight/type/label
node_features.csv         node_id plus dataset-specific features
edge_features.csv         edge_id plus dataset-specific features
labels.csv                object_id, object_type, label, optional label_name
partitions.json           partition_id, visible_nodes, visible_edges
query_motifs.json         full benchmark-construction queries
method_queries.json       method-facing query view
ground_truth.json         query_id, true_nodes, true_edges, true_partitions, motif_instance_id
metadata.json             graph and construction metadata
```

`method_queries.json` excludes evaluation-only fields such as hidden/true node and edge sets. It is derived from benchmark preprocessing and is not a separate raw-data source.

## Ground-truth construction

| Dataset | Instances | Actual construction |
|---|---:|---|
| AMLWorld HI-Small | 370 | Dataset-native laundering patterns parsed by `amlworld_preprocess.py` and retained by `amlworld_partition.py`. Types are FAN-IN, FAN-OUT, CYCLE, BIPARTITE, STACK, RANDOM, SCATTER-GATHER, and GATHER-SCATTER. |
| Elliptic | 1,000 | Illicit-centered directed-chain, fan-in, fan-out, and cycle candidates constructed by `elliptic_protocol.py`. |
| DBLP | 1,000 | Collaboration motifs selected/generated from processed edges by `structural_protocol.py`. |
| Web-Google | 1,000 | Directed hyperlink motifs selected/generated from processed edges by `structural_protocol.py`. |

AMLWorld uses five bank partitions, seed 42, at least two banks per native motif, clean inter-partition target 0.03, minute time buckets, and retention of unknown laundering transactions. All 370 instances enter the evaluator. The native set includes 57 motifs disconnected under the processed topology: 31 BIPARTITE and 26 STACK.

Elliptic assigns nodes deterministically to `P[(time_step + md5(node_id)) mod 5]`. It visits illicit centers in stable order, prioritizes cross-partition candidates, sorts by type/center/edge IDs, and selects round-robin in the fixed order directed-chain, fan-in, fan-out, cycle until 1,000.

DBLP initially assigns shuffled, sorted node IDs round-robin to five partitions with `random.Random(42)`. The final protocol accepts motifs with at least three nodes, two edges, two partitions, one hidden edge and hidden node, and a nontrivial anchor view. It deterministically supplements cross-partition three-edge bridge motifs with seed 42, sorts by type/query ID, and retains 1,000: 652 bridge, 116 cycle-4, 72 dense, 110 star, and 50 triangle motifs.

Web-Google reads the configured deterministic capped edge prefix and assigns shuffled, sorted nodes round-robin to five partitions with seed 42. The final protocol applies the same validity conditions, supplements three-edge cross-partition directed chains with `random.Random(seed + 17)`, sorts by type/query ID, and retains 1,000: 725 directed-chain, 88 directed-star, 12 directed-cycle, 100 feed-forward, and 75 dense-block motifs.

## Running CoMot and baselines

Run one method:

```bash
bash scripts/run.sh --dataset elliptic --method comot --seed 0 --output_root outputs
bash scripts/run.sh --dataset elliptic --method cn --seed 0 --output_root outputs
```

Run the configured grids:

```bash
bash scripts/run_all.sh --seeds "0 1 2"
DATASETS="elliptic dblp web_google" SEEDS="0 1 2" bash scripts/run_baselines.sh
```

All methods emit `predictions.jsonl` rows with `query_id`, `predicted_nodes`, `predicted_edges`, `score`, `runtime`, and `metadata`.

### Baselines

| Baseline | Paradigm | Submitted task adaptation | Input |
|---|---|---|---|
| CN | link heuristic | anchor-constrained candidate ranking | topology + queries |
| AA | link heuristic | anchor-constrained candidate ranking | topology + queries |
| DeepWalk | embedding | random-walk co-occurrence random indexing | topology + queries |
| node2vec | embedding | biased walks with random indexing | topology + queries |
| GraphSAGE | GNN embedding | deterministic structural features and mean aggregation | topology + queries |
| VF2 | exact matching | NetworkX matching for explicit query topology | topology + queries |
| FINAL | graph alignment | structural/attribute partition alignment | topology + partitions + queries |
| REGAL | graph alignment | structural-role partition alignment | topology + partitions + queries |
| BRIGHT | graph alignment | bridge-aware signature alignment | topology + partitions + queries |
| SEAL | link prediction | enclosing-subgraph features without GNN training | topology + queries |
| TurboISO | subgraph matching | external wrapper; NetworkX fallback in submitted config | topology + queries |
| FANMOD | motif mining | external wrapper; structural fallback in submitted config | topology + queries |
| gSpan | frequent-subgraph mining | export wrapper; frequent-pattern fallback | topology + queries |
| SubGNN | subgraph representation | ego-subgraph representation scorer | topology + queries |
| NeuGN | learned subgraph matching | anchor-seeded local navigation | topology + partitions + queries |
| TPAB | top-k subgraph matching | topology-aware beam retrieval | topology + partitions + queries |
| ISONET | neural graph retrieval | topology/edge-alignment scorer | topology + partitions + queries |
| HLOT | higher-order alignment | structural-signature alignment | topology + partitions + queries |
| CSGM | collaborative mining | bilateral scatter-gather search | topology + partitions + queries |

Each baseline writes the common output schema and is evaluated by `src/evaluation/evaluate.py`. “Adapted” implementations are the implementations used in the submitted experiments. Several consume the materialized processed topology through `StandardGraphDataset`, consistent with the submitted evaluation protocol. Ground truth is loaded after candidate generation by the evaluator.

## Evaluation protocol

Evaluate an existing prediction file:

```bash
python -m src.evaluation.evaluate \
  --predictions outputs/elliptic/cn/seed_0/predictions.jsonl \
  --ground_truth data/processed/elliptic/ground_truth.json \
  --output_dir outputs/elliptic/cn/seed_0
```

A partial motif hit is nonempty predicted-node overlap or predicted-edge overlap with a GT instance. Each prediction is paired with its best-scoring GT row. Motif recall counts unique recovered GT IDs, so duplicate predictions do not increase recall; duplicates remain candidate rows and affect the hit-prediction fraction. M-F1 is the harmonic mean of `motif_level.partial_match` and `motif_level.motif_recall`. Exact match is reported separately.

For AMLWorld, Fig. 3 reports the newly assembled candidate pool from `candidate_builder.py`, before reranking and end-to-end composition. The Table II output contains observable anchor-only hypotheses and template-assembled hypotheses. `configs/amlworld_hi_small.json` enables `comot.legacy_node_evidence` with `prefix_queries: 300` and `max_anchor_nodes: 8`; `augment_amlworld_comot_node_evidence()` uses these limits when reading anchors from `method_queries.json`. It does not read `ground_truth.json`, hidden edges, or labels.

Template candidate scoring also does not use hidden edges or labels. The orchestrator scores structural, direction, amount-bucket, time-bucket, payment-format, and currency compatibility; the reranker uses compatibility, candidate type, size, density, and compactness. Motif type/instance columns carried by preprocessing are not consumed by these scoring functions.
