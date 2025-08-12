"""Dataset loaders for different file formats."""

import json
from typing import List, Dict, Any
from .base import BaseDataLoader


class JSONLLoader(BaseDataLoader):
    """Loader for JSONL (JSON Lines) format datasets."""

    def load(self, path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSONL file.

        Args:
            path: Path to JSONL file

        Returns:
            List of problem dictionaries
        """
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        error_msg = f"Invalid JSON in line: {line}"
                        raise ValueError(error_msg) from e
        return data

    def load_sample(self, path: str, n: int = 1) -> Dict[str, Any]:
        """Load first sample from JSONL file.

        Args:
            path: Path to JSONL file
            n: Number of samples (only first one returned)

        Returns:
            First problem dictionary
        """
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if not line:
                raise ValueError("Empty dataset file")
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("Expected dictionary in JSONL")
                return data
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in first line: {line}"
                raise ValueError(error_msg) from e

    def validate_schema(self, data: Dict[str, Any]) -> bool:
        """Validate basic structure.

        Args:
            data: Problem dictionary

        Returns:
            True if has basic required fields
        """
        required_fields = ["task_id", "prompt", "tests", "entry_point"]
        return all(field in data for field in required_fields)


def get_loader(format_type: str) -> BaseDataLoader:
    """Get appropriate loader for the format.

    Args:
        format_type: Format type string

    Returns:
        Loader instance

    Raises:
        ValueError: If format not supported
    """
    loaders = {
        "jsonl": JSONLLoader,
    }

    if format_type not in loaders:
        raise ValueError(f"Unsupported format: {format_type}")

    return loaders[format_type]()
