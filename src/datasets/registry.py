"""Dataset registry and configuration management."""

from typing import Dict, Any
from pathlib import Path


# Predefined dataset configurations for Q code evaluation
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "q-humaneval": {
        "path": "./datasets/q_humaneval.jsonl",
        "format": "jsonl",
        "schema": "humaneval",
        "language": "q",
        "test_language": "python",
        "prompt_template": "q_humaneval",
    },
    # TODO: Enable q-mbpp dataset when dataset file is available
    # "q-mbpp": {
    #     "path": "./datasets/q_mbpp.jsonl",
    #     "format": "jsonl",
    #     "schema": "mbpp",
    #     "language": "q",
    #     "test_language": "python",
    #     "prompt_template": "q_mbpp"
    # }
}


def get_dataset_config(name: str) -> Dict[str, Any]:
    """Get dataset configuration by name or auto-detect from path.

    Args:
        name: Dataset name (registered) or file path

    Returns:
        Dataset configuration dictionary

    Raises:
        ValueError: If dataset not found and auto-detection fails
    """
    if name in DATASET_CONFIGS:
        return DATASET_CONFIGS[name].copy()
    else:
        # Treat as file path and auto-detect
        return auto_detect_config(name)


def auto_detect_config(path: str) -> Dict[str, Any]:
    """Auto-detect dataset configuration from file path.

    Args:
        path: Path to dataset file

    Returns:
        Auto-detected configuration dictionary

    Raises:
        ValueError: If file doesn't exist or format can't be detected
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"Dataset file not found: {path}")

    # Import here to avoid circular imports
    from .router import DatasetRouter

    router = DatasetRouter()
    return router.auto_detect_config(path)


def register_dataset(name: str, config: Dict[str, Any]) -> None:
    """Register a new dataset configuration.

    Args:
        name: Dataset name
        config: Dataset configuration dictionary
    """
    DATASET_CONFIGS[name] = config


def list_datasets() -> Dict[str, str]:
    """List all registered datasets.

    Returns:
        Dictionary mapping dataset names to their descriptions
    """
    return {
        name: (
            f"{config['language']} -> {config['test_language']} "
            f"({config['schema']})"
        )
        for name, config in DATASET_CONFIGS.items()
    }
