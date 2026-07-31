"""Tests for running API-backed model paths without PyTorch installed."""

import subprocess
import sys
import textwrap

import pytest

from src.models.optional_dependencies import raise_optional_dependency_error


WITHOUT_TORCH = """
import builtins
import importlib.util
import sys
import types

real_import = builtins.__import__
real_find_spec = importlib.util.find_spec

def import_without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")
    return real_import(name, *args, **kwargs)

def find_spec_without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        return None
    return real_find_spec(name, *args, **kwargs)

builtins.__import__ = import_without_torch
importlib.util.find_spec = find_spec_without_torch

# CLI import normally initializes PyKX through the execution backend. That is
# unrelated to model dependencies and can contend with live q test processes.
q_executor = types.ModuleType("src.evaluation.q_python_executor")
q_executor.QPythonExecutor = type("QPythonExecutor", (), {})
sys.modules["src.evaluation.q_python_executor"] = q_executor
"""


def run_without_torch(code: str) -> subprocess.CompletedProcess[str]:
    """Run code in a fresh interpreter that behaves as if torch is absent."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(WITHOUT_TORCH + code)],
        capture_output=True,
        check=False,
        text=True,
    )


def test_cli_and_api_backends_import_without_torch() -> None:
    result = run_without_torch(
        """
import src.cli
from src.models import LiteLLMModel
from src.models.factory import create_model

litellm_model = create_model("gpt-4o", model_type="litellm")
mcp_model = create_model(
    "test-model",
    model_type="mcp",
    mcp_url="https://example.test/mcp",
    mcp_tool_name="generate_code",
)
assert isinstance(litellm_model, LiteLLMModel)
assert mcp_model.__class__.__name__ == "MCPModel"
"""
    )

    assert result.returncode == 0, result.stderr


def test_local_backends_explain_how_to_install_torch() -> None:
    result = run_without_torch(
        """
from src.models.factory import create_model

for backend, extra in (("huggingface", "huggingface"), ("vllm", "vllm")):
    try:
        create_model("test-model", model_type=backend)
    except RuntimeError as error:
        message = str(error)
        assert "'torch' is not installed" in message
        assert f"poetry install -E {extra}" in message
    else:
        raise AssertionError(f"{backend} unexpectedly loaded")
"""
    )

    assert result.returncode == 0, result.stderr


def test_public_huggingface_import_has_actionable_error_without_torch() -> None:
    result = run_without_torch(
        """
try:
    from src.models import HuggingFaceModel
except RuntimeError as error:
    assert "poetry install -E huggingface" in str(error)
else:
    raise AssertionError(HuggingFaceModel)
"""
    )

    assert result.returncode == 0, result.stderr


def test_profiling_fails_cleanly_without_torch() -> None:
    result = run_without_torch(
        """
import logging
from types import SimpleNamespace

from src.cli import profile_command

logging.basicConfig(level=logging.ERROR)
args = SimpleNamespace(model="test-model", output_dir="unused", quick=True,
                       benchmark=False)
try:
    profile_command(args)
except SystemExit as error:
    assert error.code == 1
else:
    raise AssertionError("profiling unexpectedly succeeded")
"""
    )

    assert result.returncode == 0, result.stderr
    assert "hardware profiling requires optional" in result.stderr
    assert "poetry install -E vllm" in result.stderr


def test_unrelated_backend_import_error_is_not_misreported() -> None:
    original = ModuleNotFoundError(
        "No module named 'backend_internal'", name="backend_internal"
    )

    with pytest.raises(ModuleNotFoundError) as raised:
        raise_optional_dependency_error("HuggingFace", original)

    assert raised.value is original
