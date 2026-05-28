# CoMot

CoMot is a research codebase for cross-partition motif retrieval under partial graph observability.

This open-source tree contains code, configs, and minimal run scripts only. It does not include datasets, experiment outputs, logs, or generated artifacts.

## Repository Layout

- `src/`: CoMot, standard data loading, evaluation, and result aggregation.
- `configs/`: dataset and method configuration files.
- `scripts/`: small wrappers for query-view construction, running, and aggregating.
- `data/`: placeholder directory for processed datasets.
- `outputs/`: placeholder directory for experiment outputs.

## Data

This release uses a simple standard processed-data interface. Prepare each dataset under `data/processed/<dataset>/` before running CoMot.

The experiments use:

- AMLWorld HI-Small transaction data.
- Elliptic transaction graph.
- DBLP graph data.
- Web-Google graph data.

Each processed dataset should contain:

```text
data/processed/<dataset>/nodes.csv
data/processed/<dataset>/edges.csv
data/processed/<dataset>/partitions.json
data/processed/<dataset>/method_queries.json
data/processed/<dataset>/ground_truth.json
```

## Installation

```bash
pip install -r requirements.txt
```

## Example Usage

Run CoMot on one dataset:

```bash
python src/experiments/run_method.py \
  --dataset elliptic \
  --method comot \
  --seed 0 \
  --output_root outputs
```

Run CoMot on a dataset through the wrapper script:

```bash
bash scripts/run_dataset.sh elliptic
```

Aggregate results:

```bash
python src/experiments/aggregate_results.py --output_root outputs --summary_dir outputs/summary
```

## Notes

- This release intentionally excludes experiment results.
- Dataset defaults are collected in `configs/datasets.yaml`; CoMot defaults are in `configs/comot.yaml`.
