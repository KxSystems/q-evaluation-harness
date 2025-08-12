"""Prompt templates for different datasets."""

from pathlib import Path
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader
from .base import BasePromptTemplate


class JinjaTemplateBase(BasePromptTemplate):
    """Base class for Jinja2-based templates."""

    def __init__(self) -> None:
        """Initialize Jinja2 environment."""
        template_dir = Path(__file__).parent / "jinja_templates"
        self.env = Environment(loader=FileSystemLoader(template_dir))

    @property
    def template_name(self) -> str:
        """Return the Jinja2 template filename."""
        raise NotImplementedError("Subclasses must implement template_name")

    def format(self, problem: Dict[str, Any]) -> str:
        """Format a problem into a prompt using Jinja2 template.

        Args:
            problem: Problem dictionary containing task information

        Returns:
            Formatted prompt string
        """
        if not self.validate_problem(problem):
            missing = [f for f in self.required_fields if f not in problem]
            raise ValueError(f"Missing required fields: {missing}")

        template = self.env.get_template(self.template_name)
        return template.render(**problem)

    @property
    def required_fields(self) -> List[str]:
        """Return list of required fields in the problem dictionary.

        Returns:
            List of required field names
        """
        return ["prompt"]


class QHumanEvalTemplate(JinjaTemplateBase):
    """Prompt template for Q HumanEval dataset."""

    @property
    def template_name(self) -> str:
        """Return the Jinja2 template filename."""
        return "q_humaneval.j2"

    @property
    def required_fields(self) -> List[str]:
        """Required fields for Q HumanEval problems."""
        return ["prompt", "entry_point"]

    def format(self, problem: Dict[str, Any]) -> str:
        """Format Q HumanEval problem into prompt.

        Args:
            problem: Problem dictionary

        Returns:
            Formatted prompt string
        """
        if not self.validate_problem(problem):
            missing = [f for f in self.required_fields if f not in problem]
            raise ValueError(f"Missing required fields: {missing}")

        template = self.env.get_template(self.template_name)
        return template.render(prompt=problem["prompt"])


class QMBPPTemplate(JinjaTemplateBase):
    """Prompt template for Q MBPP dataset."""

    @property
    def template_name(self) -> str:
        """Return the Jinja2 template filename."""
        return "q_mbpp.j2"

    @property
    def required_fields(self) -> List[str]:
        """Required fields for Q MBPP problems."""
        return ["prompt", "entry_point"]

    def format(self, problem: Dict[str, Any]) -> str:
        """Format Q MBPP problem into prompt.

        Args:
            problem: Problem dictionary

        Returns:
            Formatted prompt string
        """
        if not self.validate_problem(problem):
            missing = [f for f in self.required_fields if f not in problem]
            raise ValueError(f"Missing required fields: {missing}")

        # Follow MBPP approach with first test as example
        description = problem["text"]
        q_tests = problem["q_test_list"]
        test_example = q_tests[0] if q_tests else ""

        template = self.env.get_template(self.template_name)
        return template.render(text=description, test_example=test_example)


class GenericTemplate(JinjaTemplateBase):
    """Generic prompt template for unknown datasets."""

    @property
    def template_name(self) -> str:
        """Return the Jinja2 template filename."""
        return "generic.j2"

    @property
    def required_fields(self) -> List[str]:
        """Required fields for generic problems."""
        return ["prompt"]

    def format(self, problem: Dict[str, Any]) -> str:
        """Format generic problem into prompt.

        Args:
            problem: Problem dictionary

        Returns:
            Formatted prompt string
        """
        if not self.validate_problem(problem):
            missing = [f for f in self.required_fields if f not in problem]
            raise ValueError(f"Missing required fields: {missing}")

        # Simple prompt formatting
        prompt_text: str = problem["prompt"]

        # Add Q programming context if not present
        needs_context = (
            "q (KDB+)" not in prompt_text and "q programmer" not in prompt_text
        )

        template = self.env.get_template(self.template_name)
        return template.render(prompt=prompt_text, needs_context=needs_context)


def get_prompt_template(template_name: str) -> BasePromptTemplate:
    """Get prompt template by name.

    Args:
        template_name: Template identifier

    Returns:
        Template instance

    Raises:
        ValueError: If template not found
    """
    templates = {
        "q_humaneval": QHumanEvalTemplate,
        "q_mbpp": QMBPPTemplate,
        "generic": GenericTemplate,
    }

    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}")

    return templates[template_name]()
