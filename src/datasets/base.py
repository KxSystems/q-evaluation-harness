"""Abstract base classes for dataset loading."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseDataLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self, path: str) -> List[Dict[str, Any]]:
        """Load dataset from given path.

        Args:
            path: Path to the dataset file

        Returns:
            List of problem dictionaries
        """
        pass

    @abstractmethod
    def load_sample(self, path: str, n: int = 1) -> Dict[str, Any]:
        """Load a sample from the dataset for schema detection.

        Args:
            path: Path to the dataset file
            n: Number of samples to load

        Returns:
            Sample problem dictionary
        """
        pass

    @abstractmethod
    def validate_schema(self, data: Dict[str, Any]) -> bool:
        """Validate that data conforms to expected schema.

        Args:
            data: Problem dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        pass
