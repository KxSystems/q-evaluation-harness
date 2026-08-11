"""Tests for the generic MCP model backend."""

import argparse
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli import _add_mcp_arguments
from src.models.factory import create_model
from src.models.generation_strategy import GenerationStrategy
from src.models.mcp_model import MCPModel


def make_model(**kwargs: object) -> MCPModel:
    return MCPModel(
        "remote-code-model",
        mcp_url="https://example.test/mcp",
        mcp_tool_name="generate_code",
        **kwargs,
    )


def text_result(*texts: str, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        isError=False,
        content=[SimpleNamespace(type="text", text=text) for text in texts],
        structuredContent=kwargs.get("structured_content"),
    )


def test_factory_creates_mcp_model() -> None:
    model = create_model(
        "remote-code-model",
        model_type="mcp",
        mcp_url="https://example.test/mcp",
        mcp_tool_name="generate_code",
    )
    assert isinstance(model, MCPModel)
    assert model.get_generation_strategy() == GenerationStrategy.ASYNC_OPTIMIZED


def test_public_model_import() -> None:
    from src.models import MCPModel as PublicMCPModel

    assert PublicMCPModel is MCPModel


@pytest.mark.parametrize("missing", ["mcp_url", "mcp_tool_name"])
def test_required_connection_configuration(missing: str) -> None:
    kwargs = {
        "mcp_url": "https://example.test/mcp",
        "mcp_tool_name": "generate_code",
    }
    kwargs.pop(missing)

    with pytest.raises(ValueError, match=missing):
        MCPModel("remote-code-model", **kwargs)


def test_rejects_prompt_argument_in_static_arguments() -> None:
    with pytest.raises(ValueError, match="must not define"):
        make_model(mcp_tool_arguments={"prompt": "fixed"})


def test_extracts_plain_text_result() -> None:
    model = make_model()
    result = text_result("  function answer() { return 42; }  ")

    assert model._extract_completion(result) == ("function answer() { return 42; }")


def test_extracts_nested_field_from_json_text() -> None:
    model = make_model(mcp_result_field="result.code")
    result = text_result('{"result":{"code":"  answer:{42}  "}}')

    assert model._extract_completion(result) == "answer:{42}"


def test_prefers_structured_result_field() -> None:
    model = make_model(mcp_result_field="output.code")
    result = text_result(
        "fallback",
        structured_content={"output": {"code": "  structured code  "}},
    )

    assert model._extract_completion(result) == "structured code"


def test_configured_result_field_must_be_a_string() -> None:
    model = make_model(mcp_result_field="code")

    with pytest.raises(TypeError, match="not a string"):
        model._extract_completion(text_result('{"code": 42}'))


@pytest.mark.asyncio
async def test_call_tool_uses_only_configured_contract() -> None:
    model = make_model(
        mcp_prompt_argument="instruction",
        mcp_prompt_suffix="\n\nReturn only generated code.",
        mcp_tool_arguments={"candidate_count": 2, "language": "example"},
        mcp_headers={"Authorization": "Bearer token"},
    )
    result = text_result("generated code")

    transport = MagicMock()
    transport.return_value.__aenter__ = AsyncMock(
        return_value=("read", "write", lambda: None)
    )
    transport.return_value.__aexit__ = AsyncMock(return_value=None)

    session = AsyncMock()
    session.call_tool.return_value = result
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=None)

    with patch("src.models.mcp_model.streamablehttp_client", transport), patch(
        "src.models.mcp_model.ClientSession", return_value=session_context
    ):
        completion = await model._call_tool("write a function")

    assert completion == "generated code"
    session.initialize.assert_awaited_once_with()
    session.call_tool.assert_awaited_once_with(
        "generate_code",
        {
            "candidate_count": 2,
            "language": "example",
            "instruction": (
                "write a function\n\n"
                "Return only generated code."
            ),
        },
    )
    transport.assert_called_once_with(
        "https://example.test/mcp",
        headers={"Authorization": "Bearer token"},
        timeout=30,
        sse_read_timeout=300,
    )


@pytest.mark.asyncio
async def test_generate_async_returns_requested_samples() -> None:
    model = make_model(mcp_concurrency=2)
    model._call_tool = AsyncMock(side_effect=["first", "second"])

    completions = await model.generate_async("Implement f", n=2)

    assert completions == ["first", "second"]
    assert model._call_tool.await_count == 2


@pytest.mark.asyncio
async def test_generate_async_bounds_tool_concurrency() -> None:
    model = make_model(mcp_concurrency=2)
    active = 0
    peak = 0

    async def call_tool(prompt: str) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return prompt

    model._call_tool = AsyncMock(side_effect=call_tool)

    assert await model.generate_async("code", n=5) == ["code"] * 5
    assert peak == 2


@pytest.mark.asyncio
async def test_generate_async_turns_tool_failure_into_empty_sample() -> None:
    model = make_model()
    model._call_tool = AsyncMock(side_effect=RuntimeError("unavailable"))

    assert await model.generate_async("Implement f") == [""]


def test_sync_generation_can_use_multiple_event_loops() -> None:
    model = make_model(mcp_concurrency=1)
    model._call_tool = AsyncMock(return_value="completion")

    assert model.generate("first", n=2) == ["completion", "completion"]
    assert model.generate("second", n=2) == ["completion", "completion"]


def test_cli_mcp_options_are_generic() -> None:
    parser = argparse.ArgumentParser()
    _add_mcp_arguments(parser)

    args = parser.parse_args(
        [
            "--mcp-url",
            "https://example.test/mcp",
            "--mcp-tool-name",
            "generate_code",
            "--mcp-prompt-argument",
            "instruction",
            "--mcp-prompt-suffix",
            " Return only code.",
            "--mcp-result-field",
            "result.code",
            "--mcp-tool-arguments",
            '{"language":"example"}',
            "--mcp-headers",
            '{"Authorization":"Bearer token"}',
        ]
    )

    assert args.mcp_url == "https://example.test/mcp"
    assert args.mcp_tool_name == "generate_code"
    assert args.mcp_prompt_argument == "instruction"
    assert args.mcp_prompt_suffix == " Return only code."
    assert args.mcp_result_field == "result.code"
    assert args.mcp_tool_arguments == {"language": "example"}
    assert args.mcp_headers == {"Authorization": "Bearer token"}
