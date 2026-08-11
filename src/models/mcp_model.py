"""Model backend for code generators exposed as MCP tools."""

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any, List

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .base import BaseModel
from .generation_strategy import (
    AsyncConfig,
    AsyncGenerator,
    GenerationStrategy,
)

logger = logging.getLogger(__name__)


class MCPModel(BaseModel, AsyncGenerator):
    """Invoke a configurable MCP tool as a code-generation model.

    MCP does not define a standard code-generation tool contract, so callers
    explicitly configure how the benchmark prompt maps to the tool input and,
    when necessary, which field contains the generated code in the result.
    """

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)

        self.url = self._required_string(kwargs.get("mcp_url"), "mcp_url")
        self.tool_name = self._required_string(
            kwargs.get("mcp_tool_name"), "mcp_tool_name"
        )
        self.prompt_argument = self._required_string(
            kwargs.get("mcp_prompt_argument", "prompt"),
            "mcp_prompt_argument",
        )
        self.prompt_suffix = kwargs.get("mcp_prompt_suffix", "")
        if not isinstance(self.prompt_suffix, str):
            raise TypeError("mcp_prompt_suffix must be a string")
        self.result_field = kwargs.get("mcp_result_field")
        if self.result_field is not None:
            self.result_field = self._required_string(
                self.result_field, "mcp_result_field"
            )

        self.tool_arguments = self._mapping(
            kwargs.get("mcp_tool_arguments"), "mcp_tool_arguments"
        )
        if self.prompt_argument in self.tool_arguments:
            raise ValueError(
                "mcp_tool_arguments must not define the configured prompt "
                f"argument {self.prompt_argument!r}"
            )

        self.headers = self._string_mapping(kwargs.get("mcp_headers"), "mcp_headers")
        self.timeout = kwargs.get("mcp_timeout", 30)
        self.sse_read_timeout = kwargs.get("mcp_sse_read_timeout", 300)
        self.max_concurrent = kwargs.get("mcp_concurrency", 3)
        if not isinstance(self.max_concurrent, int) or self.max_concurrent < 1:
            raise ValueError("mcp_concurrency must be a positive integer")

        self._request_semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    def get_generation_strategy(self) -> GenerationStrategy:
        """MCP tools are remote services and are called asynchronously."""
        return GenerationStrategy.ASYNC_OPTIMIZED

    def get_async_config(self) -> AsyncConfig:
        """Limit concurrent benchmark problems sent to the MCP service."""
        return AsyncConfig(max_concurrent=self.max_concurrent)

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """Synchronously generate ``n`` independent completions."""
        return asyncio.run(self.generate_async(prompt, n, **kwargs))

    async def generate_async(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """Generate independent samples, bounded by MCP concurrency."""
        if n < 1:
            return []

        semaphore = self._get_request_semaphore()

        async def generate_one() -> str:
            async with semaphore:
                try:
                    return await self._call_tool(prompt)
                except Exception as exc:
                    logger.warning(
                        "MCP generation failed for tool %s: %s",
                        self.tool_name,
                        exc,
                    )
                    return ""

        return await asyncio.gather(*(generate_one() for _ in range(n)))

    async def _call_tool(self, prompt: str) -> str:
        arguments = {
            **self.tool_arguments,
            self.prompt_argument: prompt + self.prompt_suffix,
        }

        async with streamablehttp_client(
            self.url,
            headers=self.headers or None,
            timeout=self.timeout,
            sse_read_timeout=self.sse_read_timeout,
        ) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(self.tool_name, arguments)

        if result.isError:
            raise RuntimeError(f"MCP tool {self.tool_name!r} returned an error")

        return self._extract_completion(result)

    def _extract_completion(self, result: Any) -> str:
        structured = getattr(result, "structuredContent", None)
        if structured is None:
            structured = getattr(result, "structured_content", None)

        if self.result_field and isinstance(structured, Mapping):
            try:
                field_value = self._get_field(structured, self.result_field)
                return self._string_result(field_value)
            except KeyError:
                pass

        text_blocks = [block.text for block in result.content if block.type == "text"]
        if not text_blocks:
            raise RuntimeError("MCP tool returned no text content")

        text = "\n".join(text_blocks)
        if not self.result_field:
            return text.strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "MCP tool text result is not JSON, but mcp_result_field "
                f"is configured as {self.result_field!r}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise TypeError("MCP tool JSON result is not an object")

        return self._string_result(self._get_field(payload, self.result_field))

    def _get_request_semaphore(self) -> asyncio.Semaphore:
        """Return a semaphore bound to the current event loop."""
        loop = asyncio.get_running_loop()
        if self._request_semaphore is None or self._semaphore_loop is not loop:
            self._request_semaphore = asyncio.Semaphore(self.max_concurrent)
            self._semaphore_loop = loop
        return self._request_semaphore

    @staticmethod
    def _get_field(payload: Mapping[str, Any], field_path: str) -> Any:
        value: Any = payload
        for part in field_path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise KeyError(f"MCP result does not contain field {field_path!r}")
            value = value[part]
        return value

    @staticmethod
    def _string_result(value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("Configured MCP result field is not a string")
        return value.strip()

    @staticmethod
    def _required_string(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _mapping(value: Any, name: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a JSON object")
        return dict(value)

    @classmethod
    def _string_mapping(cls, value: Any, name: str) -> dict[str, str]:
        mapping = cls._mapping(value, name)
        if not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in mapping.items()
        ):
            raise TypeError(f"{name} keys and values must be strings")
        return mapping
