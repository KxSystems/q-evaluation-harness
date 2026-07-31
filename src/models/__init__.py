"""Model interfaces and implementations."""

from typing import Any

from .base import BaseModel
from .factory import create_model

__all__ = [
    "BaseModel",
    "LiteLLMModel",
    "MCPModel",
    "HuggingFaceModel",
    "VLLMModel",
    "create_model",
]


def __getattr__(name: str) -> Any:
    """Load concrete model classes only when callers request them."""
    if name == "LiteLLMModel":
        from .litellm_model import LiteLLMModel

        return LiteLLMModel
    if name == "MCPModel":
        from .mcp_model import MCPModel

        return MCPModel
    if name == "HuggingFaceModel":
        try:
            from .huggingface_model import HuggingFaceModel
        except ModuleNotFoundError as error:
            from .optional_dependencies import raise_optional_dependency_error

            raise_optional_dependency_error("HuggingFace", error)
        return HuggingFaceModel
    if name == "VLLMModel":
        try:
            from .vllm_model import VLLMModel
        except ModuleNotFoundError as error:
            from .optional_dependencies import raise_optional_dependency_error

            raise_optional_dependency_error("vLLM", error)
        return VLLMModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
