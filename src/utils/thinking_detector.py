"""Detector for open-source thinking/reasoning models that need extended token limits."""

from typing import Optional


class OpenSourceThinkingDetector:
    """Detector for open-source thinking/reasoning models that need extended token limits."""
    
    THINKING_PATTERNS = [
        "gpt-oss",           # Specific thinking model
        "thinking",          # Generic thinking models  
        "reasoning",         # Reasoning variants
        "cot",              # Chain-of-thought models
        "chain-of-thought",
        "deepseek-r1",       # Full name variants
        "o1-",               # OpenAI O1 series models (o1-preview, o1-mini, etc.)
        "gpt-o1",            # GPT O1 variants
    ]
    
    @classmethod
    def is_thinking_model(cls, model_name: str) -> bool:
        """Check if model is an open-source thinking model needing extended limits.
        
        Args:
            model_name: Model identifier to check
            
        Returns:
            True if model is an open-source thinking model, False otherwise
        """
        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in cls.THINKING_PATTERNS)
    
    @classmethod
    def get_default_max_tokens(cls, model_name: str, user_override: Optional[int] = None) -> int:
        """Get appropriate token limit for model type.
        
        Args:
            model_name: Model identifier to check
            user_override: User-provided token limit override
            
        Returns:
            Appropriate max_tokens value for the model type
        """
        # Import here to avoid circular imports
        from ..constants import DEFAULT_MAX_TOKENS, DEFAULT_MAX_THINKING_TOKENS
        
        if user_override is not None:
            return user_override
        return DEFAULT_MAX_THINKING_TOKENS if cls.is_thinking_model(model_name) else DEFAULT_MAX_TOKENS


