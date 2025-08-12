"""Shared chat template handling utilities for consistent formatting."""

import logging
from typing import Optional

from transformers import AutoTokenizer


class ChatTemplateHandler:
    """Handles chat template formatting consistently across models."""

    def __init__(self, model_name: str, logger: logging.Logger) -> None:
        """Initialize chat template handler.

        Args:
            model_name: Model identifier for tokenizer loading
            logger: Logger instance for error reporting
        """
        self.model_name = model_name
        self.logger = logger
        self._tokenizer: Optional[AutoTokenizer] = None
        self._supports_chat_template: Optional[bool] = None

        # Load tokenizer and detect chat template support
        self.load_tokenizer()

    def load_tokenizer(self) -> None:
        """Load tokenizer for chat template support."""
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._supports_chat_template = self._detect_chat_template_support()
        except Exception as e:
            msg = f"Failed to load tokenizer for {self.model_name}: {str(e)}"
            self.logger.warning(msg)
            self._tokenizer = None
            self._supports_chat_template = False

    def _detect_chat_template_support(self) -> bool:
        """Detect if the model supports chat templates."""
        if self._tokenizer is None:
            return False

        try:
            # Check if tokenizer has a chat template
            if (
                hasattr(self._tokenizer, "chat_template")
                and self._tokenizer.chat_template
            ):
                return True

            # Check if apply_chat_template method exists and works
            if hasattr(self._tokenizer, "apply_chat_template"):
                test_messages = [{"role": "user", "content": "test"}]
                try:
                    self._tokenizer.apply_chat_template(
                        test_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    return True
                except Exception:
                    pass

            return False
        except Exception:
            return False

    def format_as_conversation(self, prompt: str) -> str:
        """Format prompt as a conversation for chat models.

        Args:
            prompt: Raw instruction prompt

        Returns:
            Formatted conversation prompt
        """
        if not self.supports_chat_template or self._tokenizer is None:
            return prompt

        # Create conversation format for instruction following
        messages = [{"role": "user", "content": prompt}]

        try:
            # Apply chat template
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return formatted_prompt
        except Exception as e:
            # Fallback to raw prompt if chat template fails
            self.logger.warning(f"Chat template formatting failed: {e}")
            return prompt

    @property
    def supports_chat_template(self) -> bool:
        """Whether the model supports chat templates."""
        return self._supports_chat_template or False

    @property
    def tokenizer(self) -> Optional[AutoTokenizer]:
        """Access to the underlying tokenizer."""
        return self._tokenizer