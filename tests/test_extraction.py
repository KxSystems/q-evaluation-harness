"""Tests for Q function extraction utilities."""

import pytest
from src.utils.extraction import extract_function_from_content

@pytest.mark.parametrize(
    "content,function_name,expected",
    [
        # User example 1: Function in code block with explanation
        (
            (
                "```q\nmax_element:{[x]\n    max x\n}\n```\n\n"
                "In this KDB+ function, `max_element` takes a list `x` of float "
                "numbers as input and uses the built-in `max` function to return "
                "the maximum value in the list."
            ),
            "max_element",
            "{[x]\n    max x\n}",
        ),
        # User example 2: Function in code block with detailed explanation
        (
            (
                "Here's how you can define the `max_element` function in q (KDB+) "
                "to return the maximum element in a list of floats:\n\n"
                "```q\nmax_element:{[x]\n    max x\n}\n```\n\n"
                "In this function, `max` is a built-in function in q that returns "
                "the maximum value from a list. The parameter `x` is expected to "
                "be a list of floats. The function simply applies the `max` "
                "function to the input list `x` and returns the maximum value."
            ),
            "max_element",
            "{[x]\n    max x\n}",
        ),
        # Additional example 1: Inline function without code blocks
        ("sum_list:{[lst] +/lst}", "sum_list", "{[lst] +/lst}"),
        # Additional example 2: Multi-line function with explanation
        (
            (
                "To calculate factorial in Q:\n\n"
                "factorial:{[n]\n    if[n<=1; :1];\n    n * factorial[n-1]\n}\n\n"
                "This uses recursion to compute factorial."
            ),
            "factorial",
            "{[n]\n    if[n<=1; :1];\n    n * factorial[n-1]\n}",
        ),
    ],
)
def test_extract_function_from_content(
    content: str, function_name: str, expected: str
) -> None:
    """Test function extraction with various content formats."""
    result = extract_function_from_content(content, function_name)
    assert result == expected

def test_extract_function_fallback() -> None:
    """Test fallback behavior when function not found."""
    content = "This content doesn't contain any function definition."
    function_name = "nonexistent_function"
    result = extract_function_from_content(content, function_name)
    # Should return original content as fallback
    assert result == content
