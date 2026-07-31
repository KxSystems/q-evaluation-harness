"""Helpers for model backends with optional local-inference dependencies."""

from typing import NoReturn


_EXTRAS = {
    "HuggingFace": "huggingface",
    "vLLM": "vllm",
    "hardware profiling": "vllm",
}

_OPTIONAL_PACKAGES = {
    "HuggingFace": {"accelerate", "torch", "transformers"},
    "vLLM": {"accelerate", "torch", "transformers", "vllm"},
    "hardware profiling": {"torch", "vllm"},
}


def raise_optional_dependency_error(
    feature: str, error: ModuleNotFoundError
) -> NoReturn:
    """Raise an actionable error when a local-inference import is unavailable."""
    missing = (error.name or "").partition(".")[0]
    if missing not in _OPTIONAL_PACKAGES[feature]:
        raise error

    extra = _EXTRAS[feature]
    raise RuntimeError(
        f"{feature} requires optional local-inference dependencies; "
        f"'{missing}' is not installed. Install them with "
        f"`poetry install -E {extra}` or "
        f"`pip install 'q-evaluation-harness[{extra}]'`."
    ) from error
