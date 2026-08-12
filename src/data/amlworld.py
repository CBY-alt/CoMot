import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from data.base_dataset import BaseDataset
from data.standard_format import export_amlworld_standard_format


class AMLWorldDataset(BaseDataset):
    """AMLWorld HI-Small dataset wrapper for the submitted pipeline."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        super().__init__(config=config, project_root=project_root, output_dir=output_dir, seed=seed)
        dataset_cfg = config.get("dataset", {})
        partition_cfg = config.get("partition", {})

        self.variant = dataset_cfg.get("variant", "HI-Small")
        self.transaction_path = self._resolve_path(
            os.environ.get("AMLWORLD_TRANS_PATH", dataset_cfg.get("transaction_path", "data/raw/amlworld/HI-Small_Trans.csv"))
        )
        self.pattern_path = self._resolve_path(
            os.environ.get("AMLWORLD_PATTERN_PATH", dataset_cfg.get("pattern_path", "data/raw/amlworld/HI-Small_Patterns.txt"))
        )

        self.partition_tag = partition_cfg.get("tag", "ctrl_v2_003")
        self.global_dir = self.output_dir / "semotif_global"
        self.partition_dir = self.output_dir / f"semotif_partitioned_{self.partition_tag}"
        self.partition_analysis_dir = self.output_dir / f"partition_protocol_analysis_{self.partition_tag}"

        standard_cfg = config.get("standard_format", {})
        self.standard_dataset_name = standard_cfg.get("dataset_name", "amlworld")
        self.standard_dir = self._resolve_path(
            standard_cfg.get("output_dir", f"data/processed/{self.standard_dataset_name}")
        )

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    def _run(self, args: List[str]) -> None:
        print("+ " + " ".join(args), flush=True)
        subprocess.run(args, cwd=self.project_root, check=True)

    def load_raw(self) -> Dict[str, Path]:
        if not self.transaction_path.exists():
            raise FileNotFoundError(
                f"Missing AMLWorld transaction CSV: {self.transaction_path}. "
                "Set AMLWORLD_TRANS_PATH or place the file under data/raw."
            )
        if not self.pattern_path.exists():
            raise FileNotFoundError(
                f"Missing AMLWorld pattern TXT: {self.pattern_path}. "
                "Set AMLWORLD_PATTERN_PATH or place the file under data/raw."
            )
        return {"transactions": self.transaction_path, "patterns": self.pattern_path}

    def preprocess(self) -> Path:
        self.load_raw()
        self._run([
            sys.executable,
            str(self.project_root / "src" / "data" / "amlworld_preprocess.py"),
            "--trans_path",
            str(self.transaction_path),
            "--pattern_path",
            str(self.pattern_path),
            "--output_dir",
            str(self.global_dir),
        ])
        return self.global_dir

    def build_graph(self) -> Path:
        # The AMLWorld pipeline emits graph-ready files during partitioning.
        return self.partition_dir

    def generate_partitions(self) -> Path:
        partition_cfg = self.config.get("partition", {})
        args = [
            sys.executable,
            str(self.project_root / "src" / "data" / "amlworld_partition.py"),
            "--global_dir",
            str(self.global_dir),
            "--output_dir",
            str(self.partition_dir),
            "--num_banks",
            str(partition_cfg.get("num_banks", 5)),
            "--min_banks_per_motif",
            str(partition_cfg.get("min_banks_per_motif", 2)),
            "--clean_inter_ratio",
            str(partition_cfg.get("clean_inter_ratio", 0.03)),
            "--time_mode",
            str(partition_cfg.get("time_mode", "minute")),
            "--seed",
            str(self.seed),
        ]
        if partition_cfg.get("keep_unknown_laundering", True):
            args.append("--keep_unknown_laundering")
        if "max_clean_neighbors_to_reassign" in partition_cfg:
            args.extend([
                "--max_clean_neighbors_to_reassign",
                str(partition_cfg["max_clean_neighbors_to_reassign"]),
            ])
        self._run(args)

        self._run([
            sys.executable,
            str(self.project_root / "src" / "data" / "amlworld_partition_summary.py"),
            "--partition_dir",
            str(self.partition_dir),
            "--output_dir",
            str(self.partition_analysis_dir),
        ])
        return self.partition_dir

    def get_query_motifs(self) -> Path:
        return self.partition_dir / "ground_truth_motifs.csv"

    def get_ground_truth(self) -> Path:
        return self.partition_dir / "ground_truth_motifs.csv"

    def export_standard_format(self) -> Dict[str, Path]:
        self.preprocess()
        self.generate_partitions()
        self.build_graph()
        standard_outputs = export_amlworld_standard_format(
            partition_dir=self.partition_dir,
            output_dir=self.standard_dir,
            dataset_name=self.standard_dataset_name,
            source=self.config.get("dataset", {}).get("source", "AMLWorld HI-Small"),
            citation=self.config.get("dataset", {}).get("citation", ""),
        )
        return {
            "global_dir": self.global_dir,
            "partition_dir": self.partition_dir,
            "partition_analysis_dir": self.partition_analysis_dir,
            "ground_truth": self.get_ground_truth(),
            "standard_dir": self.standard_dir,
            "standard_files": standard_outputs,
        }
