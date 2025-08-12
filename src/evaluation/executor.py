"""Test execution engine and factory."""


from .base import BaseTestExecutor
from .q_python_executor import QPythonExecutor
from .q_q_executor import QQExecutor


def get_executor(code_language: str, test_language: str) -> BaseTestExecutor:
    """Get appropriate executor based on code and test languages.

    Args:
        code_language: Language of generated code
        test_language: Language of test assertions

    Returns:
        Appropriate executor instance

    Raises:
        ValueError: If combination not supported
    """
    if code_language == "q" and test_language == "python":
        return QPythonExecutor()
    elif code_language == "q" and test_language == "q":
        return QQExecutor()
    else:
        raise ValueError(
            f"Unsupported combination: {code_language} -> {test_language}"
        )
