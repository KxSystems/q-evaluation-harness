"""Translate shared MCP JSON configuration for agent CLI backends."""

import json
import math
import re
from pathlib import Path
from typing import Any, List, Mapping


_CODEX_FIELD_MAP = {
    "url": "url",
    "command": "command",
    "args": "args",
    "env": "env",
    "env_vars": "env_vars",
    "cwd": "cwd",
    "headers": "http_headers",
    "http_headers": "http_headers",
    "env_http_headers": "env_http_headers",
    "bearer_token_env_var": "bearer_token_env_var",
    "startup_timeout_sec": "startup_timeout_sec",
    "tool_timeout_sec": "tool_timeout_sec",
    "enabled": "enabled",
    "required": "required",
    "enabled_tools": "enabled_tools",
    "disabled_tools": "disabled_tools",
}


def load_codex_mcp_overrides(config_path: str) -> List[str]:
    """Translate a Claude-style MCP JSON file into Codex config overrides."""
    path = Path(config_path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"Unable to read MCP config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in MCP config {path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError("MCP config must be a JSON object")
    servers = payload.get("mcpServers")
    if not isinstance(servers, Mapping) or not servers:
        raise ValueError("MCP config must contain a non-empty mcpServers object")

    overrides: List[str] = []
    for server_name, raw_config in servers.items():
        if not isinstance(server_name, str) or not server_name:
            raise ValueError("MCP server names must be non-empty strings")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"MCP server {server_name!r} must be an object")

        transport_type = raw_config.get("type")
        if transport_type not in (None, "http", "stdio"):
            raise ValueError(
                f"Unsupported MCP transport type {transport_type!r} "
                f"for server {server_name!r}"
            )

        configured_fields = set()
        for source_field, value in raw_config.items():
            if source_field == "type":
                continue
            target_field = _CODEX_FIELD_MAP.get(source_field)
            if target_field is None:
                raise ValueError(
                    f"Unsupported MCP config field {source_field!r} "
                    f"for Codex server {server_name!r}"
                )
            configured_fields.add(target_field)
            _append_override(
                overrides,
                ["mcp_servers", server_name, target_field],
                value,
            )

        if "url" not in configured_fields and "command" not in configured_fields:
            raise ValueError(
                f"MCP server {server_name!r} requires either url or command"
            )
        if "required" not in configured_fields:
            _append_override(
                overrides,
                ["mcp_servers", server_name, "required"],
                True,
            )

    return overrides


def _append_override(output: List[str], path: List[str], value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("MCP config map keys must be non-empty strings")
            _append_override(output, [*path, key], item)
        return
    output.append(f"{_toml_key(path)}={_toml_value(value)}")


def _toml_key(parts: List[str]) -> str:
    return ".".join(
        part if re.fullmatch(r"[A-Za-z0-9_-]+", part) else json.dumps(part)
        for part in parts
    )


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return "[" + ",".join(_toml_value(item) for item in value) + "]"
    raise ValueError(f"Unsupported MCP config value: {value!r}")
