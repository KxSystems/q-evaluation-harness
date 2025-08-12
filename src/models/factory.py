"""Model factory for creating model instances."""

import os
import torch
from typing import Any
from .base import BaseModel
from .litellm_model import LiteLLMModel
from .huggingface_model import HuggingFaceModel
from .vllm_model import VLLMModel


def create_model(model_name: str, model_type: str = "auto", **kwargs: Any) -> BaseModel:
    """Create a model instance based on the model name and type.

    Args:
        model_name: Model identifier
        model_type: Type of model ('auto', 'litellm', 'huggingface', 'vllm')
        **kwargs: Additional model parameters

    Returns:
        Model instance

    Raises:
        ValueError: If model type not supported
    """
    if model_type == "auto":
        # Auto-detect model type
        model_type = _detect_model_type(model_name)

    if model_type == "huggingface":
        return HuggingFaceModel(model_name, **kwargs)
    elif model_type == "litellm":
        return LiteLLMModel(model_name, **kwargs)
    elif model_type == "vllm":
        return VLLMModel(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _detect_model_type(model_name: str) -> str:
    """Detect the appropriate model type based on model name patterns.

    Uses elegant pattern recognition:
    1. vLLM: Models optimized for high-performance inference (multi-GPU setups)
    2. HuggingFace: Contains '/' (org/model format) or local paths
    3. APIs: Known provider patterns without slashes
    4. Default: vLLM for large models, HuggingFace otherwise

    Args:
        model_name: Model identifier

    Returns:
        Model type ('litellm', 'huggingface', or 'vllm')
    """

    # 1. Check for vLLM preference (multi-GPU environments)
    if _should_use_vllm(model_name):
        return "vllm"
    
    # 2. Local path check (highest priority for HF)
    if os.path.exists(model_name) and os.path.isdir(model_name):
        return "huggingface"

    # 3. HuggingFace pattern: org/model format (very strong indicator)
    if ("/" in model_name) and (not "xai" in model_name) and (not "gemini" in model_name):
        return "huggingface"

    # 4. Known API patterns (only for models WITHOUT slashes)
    api_patterns = [
        # OpenAI models
        "gpt-3",
        "gpt-4",
        "gpt-35",
        'gpt-5',
        "text-davinci",
        "text-curie",
        "text-babbage",
        "text-ada",
        "davinci",
        "curie",
        "babbage",
        "ada",
        "o3",
        "o1",
        "o4-mini",
        "o4-mini-high",
        # Anthropic models
        "claude-",
        "claude_",
        # Google models
        "gemini",
        "gemini-",
        "palm-",
        "chat-bison",
        "text-bison",
        # Other providers
        "command-",
        "j2-",
        "ai21",
        "cohere-",
        "grok-",
        # Platform indicators
        "azure-",
        "bedrock-",
        "vertex-",
        "openai-",
    ]
    

    model_lower = model_name.lower()
    
    if any(
        model_lower.startswith(pattern) or pattern in model_lower
        for pattern in api_patterns
    ):
        return "litellm"

    # 5. Known HuggingFace model patterns (models without slashes)
    hf_indicators = [
        "starcoder",
        "codellama",
        "deepseek",
        "bigcode",
        "codet5",
        "unixcoder",
        "codebert",
        "roberta",
        "bert-",
        "distilbert",
        "gpt2",
        "bloom",
        "flan-",
        "t5-",
        "opt-",
        "pythia",
    ]

    if any(indicator in model_lower for indicator in hf_indicators):
        return "huggingface"

    # 6. Default to vLLM for unknown models (optimized inference)
    return "vllm"


def _should_use_vllm(model_name: str) -> bool:
    """Determine if vLLM should be used based on model characteristics and hardware.

    Args:
        model_name: Model identifier

    Returns:
        True if vLLM is recommended, False otherwise
    """
    # Check if we have multiple GPUs (vLLM's strength)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus < 2:
        return False

    # Check for large models that benefit from vLLM optimization
    model_lower = model_name.lower()
    large_model_patterns = [
        "70b",
        "65b",
        "30b",
        "20b",
        "13b",
        "7b",  # Parameter counts
        "large",
        "xl",
        "xxl",  # Size indicators
        "code-",
        "starcoder",
        "codellama",
        "deepseek",  # Code models
        "llama",
        "mistral",
        "mixtral",
        "qwen",  # Popular large models
        "falcon",
        "mpt",
        "wizard",
        "vicuna",
    ]

    # vLLM is excellent for these model types
    if any(pattern in model_lower for pattern in large_model_patterns):
        return True

    # For API models, don't use vLLM
    api_indicators = ["gpt-", "claude-", "gemini-", "palm-", "grok-"]
    if any(api in model_lower for api in api_indicators):
        return False

    # Default: use vLLM for multi-GPU setups with unknown large models
    return True
