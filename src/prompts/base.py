"""Abstract base class for prompt templates."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BasePromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @abstractmethod
    def format(self, problem: Dict[str, Any]) -> str:
        """Format a problem into a prompt for the model.

        Args:
            problem: Problem dictionary containing task information

        Returns:
            Formatted prompt string
        """
        pass

    @property
    @abstractmethod
    def required_fields(self) -> List[str]:
        """Return list of required fields in the problem dictionary.

        Returns:
            List of required field names
        """
        pass

    def validate_problem(self, problem: Dict[str, Any]) -> bool:
        """Validate that problem contains all required fields.

        Args:
            problem: Problem dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        return all(field in problem for field in self.required_fields)
