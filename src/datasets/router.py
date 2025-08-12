"""Dataset routing and auto-detection logic."""

import json
from typing import Dict, Any
from pathlib import Path


class DatasetRouter:
    """Router for dataset format and schema detection."""

    def detect_format(self, path: str) -> str:
        """Detect data format based on file extension and content.

        Args:
            path: Path to dataset file

        Returns:
            Detected format ('jsonl', 'json', etc.)
        """
        file_path = Path(path)

        if file_path.suffix == ".jsonl":
            return "jsonl"
        elif file_path.suffix == ".json":
            return "json"
        else:
            # Sample first few lines to detect format
            return self._sample_and_detect_format(path)

    def _sample_and_detect_format(self, path: str) -> str:
        """Sample file content to detect format.

        Args:
            path: Path to dataset file

        Returns:
            Detected format
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()

                # Try to parse as JSON to see if it's JSONL
                try:
                    json.loads(first_line)
                    # If first line is valid JSON, likely JSONL
                    return "jsonl"
                except json.JSONDecodeError:
                    pass

        except Exception:
            pass

        # Default fallback
        return "jsonl"

    def detect_schema(self, sample_data: Dict[str, Any]) -> str:
        """Detect schema based on column names.

        Args:
            sample_data: Sample data dictionary

        Returns:
            Detected schema type
        """
        required_cols = set(sample_data.keys())

        # Q-specific schemas (our focus)
        if "entry_point" in required_cols and "prompt" in required_cols:
            if "test_setup_code" in required_cols:
                return "mbpp"
            elif "canonical_solution" in required_cols:
                return "humaneval"

        # Check for Q-specific fields
        if "q_tests" in required_cols:
            return (
                "q_mbpp"
                if "test_setup_code" in required_cols
                else "q_humaneval"
            )

        # Default/generic schema
        return "generic"

    def auto_detect_config(self, path: str) -> Dict[str, Any]:
        """Auto-detect configuration for unknown datasets.

        Args:
            path: Path to dataset file

        Returns:
            Auto-detected configuration dictionary
        """
        from .loaders import get_loader

        format_type = self.detect_format(path)
        loader = get_loader(format_type)
        sample = loader.load_sample(path)
        schema = self.detect_schema(sample)

        # Determine language based on schema
        if schema.startswith("q_") or "q_tests" in sample:
            language = "q"
            # Q code tested with Python assertions
            test_language = "python"
            prompt_template = schema
        else:
            # Default to Q since this is Q evaluation system
            language = "q"
            test_language = "python"
            prompt_template = "generic"

        return {
            "path": path,
            "format": format_type,
            "schema": schema,
            "language": language,
            "test_language": test_language,
            "prompt_template": prompt_template,
        }
