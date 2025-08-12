"""IO utilities for data handling."""

import json
from typing import List, Dict, Any
from pathlib import Path


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """Save data to JSONL format.

    Args:
        data: List of dictionaries to save
        path: Output file path
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def append_to_jsonl(record: Dict[str, Any], path: str) -> None:
    """Append a single record to JSONL format.

    Args:
        record: Dictionary to append
        path: Output file path
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL format.

    Args:
        path: Input file path

    Returns:
        List of dictionaries
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(data: Any, path: str) -> None:
    """Save data to JSON format.

    Args:
        data: Data to save
        path: Output file path
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
