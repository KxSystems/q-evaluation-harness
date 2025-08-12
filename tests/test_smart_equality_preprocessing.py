"""
Comprehensive test suite for smart equality preprocessing in QPythonExecutor.

Tests cover:
- Basic equality assertions (should be converted)
- Inequality operators (should be preserved)
- Assert not statements (should be preserved)
- Variable assignments (should be preserved)
- Multi-line assertions (should be handled correctly)
- Complex expressions (should be handled safely)
- Edge cases and malformed input
"""

import pytest
from typing import Generator
from src.evaluation.q_python_executor import QPythonExecutor

class TestSmartEqualityPreprocessing:
    """Test cases for _preprocess_test_for_smart_equality method."""

    @pytest.fixture
    def executor(self) -> Generator[QPythonExecutor, None, None]:
        """Create a QPythonExecutor instance for testing."""
        yield QPythonExecutor()

    def test_basic_equality_conversion(
        self, executor: QPythonExecutor
    ) -> None:
        """Test basic == assertions are converted to smart_equal."""
        test_input = """def check(candidate):
    assert candidate([1, 2, 3]) == [1, 2, 3]
    assert candidate('hello') == 'hello'"""

        expected = """def check(candidate):
    assert smart_equal(candidate([1, 2, 3]), [1, 2, 3])
    assert smart_equal(candidate('hello'), 'hello')"""

        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == expected

    def test_inequality_operators_preserved(
        self, executor: QPythonExecutor
    ) -> None:
        """Test that inequality operators are NOT converted."""
        test_input = """def check(candidate):
    assert abs(candidate(1.33) - 0.33) < 1e-6
    assert candidate([1, 2]) != [2, 1]
    assert len(candidate('test')) > 0
    assert candidate(5) <= 10
    assert candidate(10) >= 5"""

        # Should remain unchanged
        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == test_input

    def test_assert_not_statements_preserved(
        self, executor: QPythonExecutor
    ) -> None:
        """Test that 'assert not' statements are NOT converted."""
        test_input = """def check(candidate):
    assert not candidate([1, 20, 4, 10], 5)
    assert not candidate("<<<><>>>>")
    assert not candidate(")((")")"""

        # Should remain unchanged
        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == test_input

    def test_variable_assignments_preserved(
        self, executor: QPythonExecutor
    ) -> None:
        """Test that variable assignments within check functions are preserved."""
        test_input = """def check(candidate):
    import math
    import random
    rng = random.Random(42)
    coeffs = [1, 2, 3, 4]
    solution = candidate(coeffs)
    assert math.fabs(poly(coeffs, solution)) < 1e-4"""

        # Should remain unchanged
        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == test_input

    def test_multi_line_assertions(self, executor: QPythonExecutor) -> None:
        """Test that multi-line assertions are handled correctly."""
        test_input = """def check(candidate):
    assert candidate('test') == [
        'item1', 'item2', 'item3'
    ]"""

        # AST implementation correctly converts multi-line assertions
        expected = """def check(candidate):
    assert smart_equal(candidate('test'), ['item1', 'item2', 'item3'])"""

        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == expected

    def test_single_line_complex_assertions(
        self, executor: QPythonExecutor
    ) -> None:
        """Test single-line assertions with complex expressions."""
        test_input = """def check(candidate):
    assert candidate([3, 5, 7]) == (3 + 5 + 7, 3 * 5 * 7)
    assert candidate('abcde' + 'cade' + 'CADE') == 5"""

        expected = """def check(candidate):
    assert smart_equal(candidate([3, 5, 7]), (3 + 5 + 7, 3 * 5 * 7))
    assert smart_equal(candidate('abcde' + 'cade' + 'CADE'), 5)"""

        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == expected

    def test_mixed_assert_types(self, executor: QPythonExecutor) -> None:
        """Test a mix of different assertion types."""
        test_input = """def check(candidate):
    assert candidate([1, 2, 3]) == [1, 2, 3]
    assert not candidate([1, 20, 4, 10], 5)
    assert abs(candidate(1.33) - 0.33) < 1e-6
    assert candidate('hello') == 'hello'
    solution = candidate([1, 2, 3])
    assert solution == [1, 2, 3]"""

        # AST normalizes number formatting (1e-6 -> 1e-06)
        expected = """def check(candidate):
    assert smart_equal(candidate([1, 2, 3]), [1, 2, 3])
    assert not candidate([1, 20, 4, 10], 5)
    assert abs(candidate(1.33) - 0.33) < 1e-06
    assert smart_equal(candidate('hello'), 'hello')
    solution = candidate([1, 2, 3])
    assert smart_equal(solution, [1, 2, 3])"""

        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == expected

    def test_no_check_function(self, executor: QPythonExecutor) -> None:
        """Test that code without check function is unchanged."""
        test_input = """def other_function():
    assert x == y"""

        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == test_input

    def test_no_equality_operators(self, executor: QPythonExecutor) -> None:
        """Test that code without == operators is unchanged."""
        test_input = """def check(candidate):
    assert candidate([1, 2, 3])
    assert len(candidate('test')) > 0"""

        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == test_input

    def test_indentation_preservation(self, executor: QPythonExecutor) -> None:
        """Test that indentation is properly preserved."""
        test_input = """def check(candidate):
    if True:
        assert candidate([1, 2]) == [1, 2]
        if nested:
            assert candidate('test') == 'test'"""

        expected = """def check(candidate):
    if True:
        assert smart_equal(candidate([1, 2]), [1, 2])
        if nested:
            assert smart_equal(candidate('test'), 'test')"""

        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == expected

    def test_comments_preservation(self, executor: QPythonExecutor) -> None:
        """Test that comments are handled by AST (stripped during unparse)."""
        test_input = """def check(candidate):
    assert candidate([1, 2]) == [1, 2]  # This should be converted
    # This is a comment
    assert candidate('test') == 'test'  # Another comment"""

        # AST strips comments during unparse - acceptable for test execution
        expected = """def check(candidate):
    assert smart_equal(candidate([1, 2]), [1, 2])
    assert smart_equal(candidate('test'), 'test')"""

        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == expected

    def test_edge_case_malformed_input(
        self, executor: QPythonExecutor
    ) -> None:
        """Test handling of edge cases and malformed input."""
        test_input = """def check(candidate):
    assert candidate(== ==) == invalid
    assert == candidate() ==
    assert"""

        # Should handle gracefully and not crash
        result = executor._preprocess_test_for_smart_equality(test_input)
        # The malformed lines should remain unchanged
        assert "def check(candidate):" in result

    def test_complex_equality_with_nested_operators(
        self, executor: QPythonExecutor
    ) -> None:
        """Test complex expressions that contain != in sub-expressions."""
        test_input = """def check(candidate):
    assert candidate([x for x in range(10) if x != 5]) == [0, 1, 2, 3, 4, 6, 7, 8, 9]"""

        # AST correctly identifies main == operator and converts it
        # The != in list comprehension doesn't affect main comparison
        expected = """def check(candidate):
    assert smart_equal(candidate([x for x in range(10) if x != 5]), [0, 1, 2, 3, 4, 6, 7, 8, 9])"""

        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == expected

    def test_string_with_equality_signs(
        self, executor: QPythonExecutor
    ) -> None:
        """Test strings containing equality signs."""
        test_input = """def check(candidate):
    assert candidate('a==b') == 'a==b'
    assert candidate('test != fail') == 'test != fail'"""

        expected = """def check(candidate):
    assert smart_equal(candidate('a==b'), 'a==b')
    assert smart_equal(candidate('test != fail'), 'test != fail')"""

        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == expected

class TestSmartEqualityEdgeCases:
    """Additional edge case tests for smart equality preprocessing."""

    @pytest.fixture
    def executor(self) -> Generator[QPythonExecutor, None, None]:
        """Create a QPythonExecutor instance for testing."""
        yield QPythonExecutor()

    def test_dataset_specific_patterns(
        self, executor: QPythonExecutor
    ) -> None:
        """Test patterns found in the actual q_humaneval dataset."""

        # Pattern from separate_paren_groups task
        test_input = """def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == [
        '(()())', '((()))', '()', '((())()())'
    ]"""

        # AST implementation correctly converts multi-line assertions
        expected = """def check(candidate):
    assert smart_equal(candidate('(()()) ((())) () ((())()())'), ['(()())', '((()))', '()', '((())()())'])"""

        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == expected

        # Pattern from truncate_number task with inequality
        test_input2 = """def check(candidate):
    assert candidate(3.5) == 0.5
    assert abs(candidate(1.33) - 0.33) < 1e-6"""

        expected2 = """def check(candidate):
    assert smart_equal(candidate(3.5), 0.5)
    assert abs(candidate(1.33) - 0.33) < 1e-06"""

        result2 = executor._preprocess_test_for_smart_equality(test_input2)
        assert result2 == expected2

        # Pattern with assert not
        test_input3 = """def check(candidate):
    assert candidate([1, 2, 4, 10], 100)
    assert not candidate([1, 20, 4, 10], 5)"""

        result3 = executor._preprocess_test_for_smart_equality(test_input3)
        # Note: The first line doesn't have == so it should remain unchanged
        assert result3 == test_input3

    def test_mathematical_expressions(self, executor: QPythonExecutor) -> None:
        """Test mathematical expressions in assertions."""
        test_input = """def check(candidate):
    assert candidate([3, 5, 7]) == (3 + 5 + 7, 3 * 5 * 7)
    assert candidate(3 * 19) == [3, 19]
    assert candidate(5 * 17) == False"""

        expected = """def check(candidate):
    assert smart_equal(candidate([3, 5, 7]), (3 + 5 + 7, 3 * 5 * 7))
    assert smart_equal(candidate(3 * 19), [3, 19])
    assert smart_equal(candidate(5 * 17), False)"""

        result = executor._preprocess_test_for_smart_equality(test_input)
        assert result == expected
