from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class BaseDataset(ABC):
    """Abstract interface for datasets used by CoMot experiments."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        self.config = config
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.seed = seed

    @abstractmethod
    def load_raw(self) -> Any:
        """Load or validate raw dataset inputs."""

    @abstractmethod
    def preprocess(self) -> Any:
        """Convert raw inputs into dataset-specific processed artifacts."""

    @abstractmethod
    def build_graph(self) -> Any:
        """Build graph objects or graph-ready files."""

    @abstractmethod
    def generate_partitions(self) -> Any:
        """Generate train/test, institution, or visibility partitions."""

    @abstractmethod
    def get_query_motifs(self) -> Any:
        """Return query motif definitions or paths."""

    @abstractmethod
    def get_ground_truth(self) -> Any:
        """Return ground-truth labels or paths."""

    @abstractmethod
    def export_standard_format(self) -> Any:
        """Export this dataset into the standard experiment format."""
