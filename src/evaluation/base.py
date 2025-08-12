"""Abstract base class for test executors."""

from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseTestExecutor(ABC):
    """Abstract base class for test execution engines."""

    @abstractmethod
    def execute(
        self, code: str, tests: str, setup_code: str = "", timeout: float = 5.0
    ) -> Tuple[bool, Any]:
        """Execute generated code against test cases.

        Args:
            code: Generated code to test
            tests: Test cases/assertions
            setup_code: Code to execute before tests
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (success, info) where success is bool and info
            contains execution time on success or error message on failure
        """
        pass

    @property
    @abstractmethod
    def supported_languages(self) -> Tuple[str, str]:
        """Return tuple of (code_language, test_language) supported.

        Returns:
            Tuple indicating what language combination this executor supports
        """
        pass
