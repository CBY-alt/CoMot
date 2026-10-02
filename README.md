# CoMot

This repository contains the CoMot code.

## Layout

```text
configs/            Dataset, CoMot, and baseline configuration
src/data/           Dataset preprocessing and partitioning
src/methods/        CoMot and baseline implementations
src/evaluation/     Shared evaluator and metrics
src/experiments/    Ablation, robustness, ranking, and analysis code
paper_experiments/  Paper-table and scalability evaluation entry points
scripts/            Shell entry points
```

Raw data are not included. Place each dataset under `data/raw/` and generate the processed files under `data/processed/`.

## Run

```bash
pip install -r requirements.txt

# One dataset, method, and seed
bash scripts/run.sh --dataset amlworld --method comot --seed 0 --output_root outputs

# AMLWorld preprocessing, five-party partitioning, CoMot, and evaluation
bash scripts/preprocess_amlworld.sh

# All datasets, methods, and paper seeds
bash scripts/run_all.sh --seeds "0 1 2" --output_root outputs

# Built-in baselines only
bash scripts/run_baselines.sh

# Aggregate the main results
bash scripts/make_paper_tables.sh

# Paper analyses
python -m src.experiments.assembly
python -m src.experiments.prioritization
python -m src.experiments.ablation
python -m src.experiments.robustness
python scripts/eval_strict_overlap.py --output-dir outputs/paper/strict_overlap
python paper_experiments/retrieval_budget.py
python paper_experiments/graph_scalability.py run

# Task-adapted gMatch and MixMatch
python paper_experiments/external_baselines.py \
  --gmatch /path/to/gMatch \
  --mixmatch /path/to/MixMatch \
  --seeds 0 1 2
```

Use `python -m src.run --help` for all single-run options.
