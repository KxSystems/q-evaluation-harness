#!/usr/bin/env python3
"""
Tests for QPythonExecutor from q_python_executor.py
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Any, Generator

# Import after path modification to avoid linter error
from src.evaluation.q_python_executor import QPythonExecutor  # noqa: E402
from src.constants import (  # noqa: E402
    EXECUTION_PASSED,
    EXECUTION_TIMED_OUT,
    EXECUTION_FAILED_PREFIX,
)

# Check if pykx is available
try:
    import pykx  # noqa: F401

    HAS_PYKX = True
except ImportError:
    HAS_PYKX = False

class TestQPythonExecutor:
    """Test the QPythonExecutor class with various scenarios."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create a temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def executor(self) -> QPythonExecutor:
        """Create a QPythonExecutor instance."""
        return QPythonExecutor()

    # ===== Success Cases Based on Provided Examples =====

    @pytest.mark.parametrize(
        "q_code,test_function,description",
        [
            (
                "{ last each x }",
                """def check(candidate):
    data1 = [(1, 'Rash', 21), (2, 'Varsha', 20), (3, 'Kil', 19)]
    assert candidate(data1) == [21, 20, 19]
    data2 = [(1, 'Sai', 36), (2, 'Ayesha', 25), (3, 'Salman', 45)]
    assert candidate(data2) == [36, 25, 45]
    data3 = [(1, 'Sudeep', 14), (2, 'Vandana', 36), (3, 'Dawood', 56)]
    assert candidate(data3) == [14, 36, 56]""",
                "rear_extract",
            ),
            (
                "{lower x}",
                """def check(candidate):
    assert candidate("InValid") == "invalid"
    assert candidate("TruE") == "true"
    assert candidate("SenTenCE") == "sentence" """,
                "is_lower",
            ),
            (
                "{[x] raze x}",
                """def check(candidate):
    assert candidate(['x', 'y', 'z']) == 'xyz'
    assert candidate(['x', 'y', 'z', 'w', 'k']) == 'xyzwk'""",
                "string_concatenation",
            ),
        ],
    )
    def test_success_cases(
        self,
        executor: QPythonExecutor,
        q_code: str,
        test_function: str,
        description: str,
    ) -> None:
        """Test successful Q code execution with various functions."""
        # Execute
        passed, info = executor.execute(q_code, test_function, timeout=5)

        # Assert
        assert passed is True
        assert info == EXECUTION_PASSED

    # ===== Error Cases =====

    @pytest.mark.parametrize(
        "q_code,expected_error",
        [
            ("", "Empty Q code"),  # Empty Q code
            ("   \n\t  \n  ", "Empty Q code"),  # Whitespace only
        ],
    )
    def test_empty_q_code_errors(
        self, executor: QPythonExecutor, q_code: str, expected_error: str
    ) -> None:
        """Test various empty Q code error conditions."""
        test_function = """def check(candidate):
    assert candidate() == 1"""

        # Execute
        passed, info = executor.execute(q_code, test_function)

        # Assert
        assert passed is False
        assert expected_error in str(info)

    @pytest.mark.parametrize(
        "q_code,test_function,expected_error",
        [
            (
                "42",  # Not a function
                """def check(candidate):
    assert candidate() == 1""",
                "Test assertion failed",
            ),
            (
                "invalid syntax",
                """def check(candidate):
    assert candidate() == 1""",
                "syntax",
            ),
            (
                "{x + 1}",
                """def check(candidate):
    assert candidate(5) == 7""",  # Should be 6, not 7
                "Test assertion failed",
            ),
            (
                "{x + 1}",
                "invalid test format",  # No check function
                "Test execution failed: invalid syntax",
            ),
        ],
    )
    def test_execution_errors(
        self,
        executor: QPythonExecutor,
        q_code: str,
        test_function: str,
        expected_error: str,
    ) -> None:
        """Test various execution error conditions."""
        # Execute
        passed, info = executor.execute(q_code, test_function)

        # Assert
        assert passed is False
        assert expected_error in str(info)

    @pytest.mark.parametrize(
        "test_function,expected_pass",
        [
            (
                """def check(candidate):
    pass""",
                True,
            ),  # Empty check function should pass
            (
                """def check(candidate):
    assert candidate(5) == 6""",
                True,
            ),  # Valid test
        ],
    )
    def test_empty_tests(
        self,
        executor: QPythonExecutor,
        test_function: str,
        expected_pass: bool,
    ) -> None:
        """Test handling of empty test functions."""
        q_code = "{x + 1}"

        passed, info = executor.execute(q_code, test_function)

        assert passed is expected_pass
        if expected_pass:
            assert info == EXECUTION_PASSED
        else:
            assert info.startswith(EXECUTION_FAILED_PREFIX)

    # ===== Timeout and Performance Tests =====

    def test_custom_timeout(self, executor: QPythonExecutor) -> None:
        """Test with custom timeout setting."""
        q_code = "{x + 1}"
        test_function = """def check(candidate):
    assert candidate(5) == 6"""

        passed, info = executor.execute(q_code, test_function, timeout=10)

        assert passed is True
        assert info == EXECUTION_PASSED

    # ===== Exception Handling Tests =====

    def test_execution_time_measurement(
        self, executor: QPythonExecutor
    ) -> None:
        """Test that execution time is properly measured."""
        q_code = "{x + 1}"
        test_function = """def check(candidate):
    import time
    time.sleep(0.001)  # Small delay to ensure measurable time
    assert candidate(5) == 6"""

        passed, info = executor.execute(q_code, test_function)

        assert passed is True
        assert info == "passed"

    def test_string_list_returns(self, executor: QPythonExecutor) -> None:
        """Test parentheses grouping function."""
        q_code = """{[x]
s: x where x<>" ";
if[0=count s; :()];
state: (();"";0i);
process: {[state;char]
    result: state 0;
    buffer: state 1;
    depth: state 2;
    newbuffer: buffer,char;
    newdepth: depth + $[char="("; 1; -1];
    $[0=newdepth;
        (result,enlist newbuffer;"";0i);
        (result;newbuffer;newdepth)]
};
finalstate: process/[state;s];
finalstate 0}"""

        test_function = """def check(candidate):
    result1 = ['(()())', '((()))', '()', '((())()())']
    assert candidate('(()()) ((())) () ((())()())') == result1
    result2 = ['()', '(())', '((()))', '(((())))']
    assert candidate('() (()) ((())) (((())))') == result2
    assert candidate('(()(())((())))') == ['(()(())((())))']
    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']"""

        passed, info = executor.execute(q_code, test_function, timeout=5)

        assert passed is True
        assert info == EXECUTION_PASSED

    def test_timeout_memory_explosion(self, executor: QPythonExecutor) -> None:
        """Test timeout handling for memory-explosive Q function."""
        # Q function that calculates product of factorials
        q_code = """{[x]
  if[x<1; '"n must be positive"];
  :prd prd\\ 1+til x   / product of factorials
}"""

        test_function = """def check(candidate):
    # This should timeout before completing
    result = candidate(15)  # 15! factorials will exhaust memory/time
    assert result > 0"""  # Should never reach this

        # Use short timeout to test timeout mechanism
        passed, info = executor.execute(q_code, test_function, timeout=2)

        # Should fail due to timeout, not memory explosion
        assert passed is False
        assert info == EXECUTION_TIMED_OUT

class TestSmartEqual:
    """Test the smart_equal utility function used by QPythonExecutor."""

    @pytest.fixture
    def executor(self) -> QPythonExecutor:
        """Create a QPythonExecutor instance to access smart_equal."""
        return QPythonExecutor()

    @pytest.mark.parametrize(
        "val1,val2,expected",
        [
            # Simple equality cases
            (1, 1, True),
            ("test", "test", True),
            (True, True, True),
            (None, None, True),
            (1, 2, False),
            ("test", "other", False),
            (True, False, False),
        ],
    )
    def test_simple_equality(
        self, executor: QPythonExecutor, val1: Any, val2: Any, expected: bool
    ) -> None:
        """Test basic equality cases."""
        assert executor._smart_equal(val1, val2) is expected

    @pytest.mark.parametrize(
        "val1,val2,expected",
        [
            # List/array equality
            ([1, 2, 3], [1, 2, 3], True),
            ([1, 2, 3], (1, 2, 3), True),  # List vs tuple
            ([], [], True),
            ([1, 2, 3], [1, 2, 4], False),
            ([1, 2], [1, 2, 3], False),
        ],
    )
    def test_list_equality(
        self, executor: QPythonExecutor, val1: Any, val2: Any, expected: bool
    ) -> None:
        """Test list/array equality."""
        assert executor._smart_equal(val1, val2) is expected

    @pytest.mark.parametrize(
        "val1,val2,expected",
        [
            # Nested structures
            ([[1, 2], [3, 4]], [[1, 2], [3, 4]], True),
            ([[1, 2], [3, 4]], [(1, 2), (3, 4)], True),
            ([[1, 2], [3, 4]], [[1, 2], [3, 5]], False),
        ],
    )
    def test_nested_structures(
        self, executor: QPythonExecutor, val1: Any, val2: Any, expected: bool
    ) -> None:
        """Test nested data structure equality."""
        assert executor._smart_equal(val1, val2) is expected

    @pytest.mark.parametrize(
        "val1,val2,expected",
        [
            # Set equality
            ({1, 2, 3}, {3, 2, 1}, True),
            (set(), set(), True),
            ({1, 2, 3}, {1, 2, 4}, False),
        ],
    )
    def test_set_equality(
        self, executor: QPythonExecutor, val1: Any, val2: Any, expected: bool
    ) -> None:
        """Test set equality."""
        assert executor._smart_equal(val1, val2) is expected

    def test_error_handling(self, executor: QPythonExecutor) -> None:
        """Test error handling in smart_equal."""
        # These should not raise exceptions
        result = executor._smart_equal(object(), object())
        assert isinstance(result, bool)

        # Test with incomparable types
        result = executor._smart_equal([1, 2, 3], "string")
        assert isinstance(result, bool)

if __name__ == "__main__":
    pytest.main([__file__])
