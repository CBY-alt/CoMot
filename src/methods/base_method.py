from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class BaseMethod(ABC):
    """Abstract interface for graph recovery or baseline methods."""

    def __init__(self, config: Dict[str, Any], project_root: Path, output_dir: Path, seed: int):
        self.config = config
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.seed = seed

    @abstractmethod
    def fit(self, dataset: Any) -> Any:
        """Fit or prepare the method on a dataset."""

    @abstractmethod
    def predict(self, dataset: Any) -> Any:
        """Generate motif or link recovery predictions."""

    @abstractmethod
    def evaluate(self, dataset: Any) -> Any:
        """Evaluate method predictions under the experiment protocol."""
