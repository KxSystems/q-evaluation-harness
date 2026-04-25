"""Factory for creating agent backends."""

from typing import Any, List, Optional

from .base import AgentBackend
from .claude_code import ClaudeCodeBackend
from .codex import CodexBackend


AGENT_BACKENDS = {
    "claude-code": ClaudeCodeBackend,
    "codex": CodexBackend,
}


def create_agent_backend(
    backend_name: str,
    model: str,
    max_turns: int = 10,
    agent_instructions: Optional[str] = None,
    timeout: float = 300.0,
    extra_args: Optional[List[str]] = None,
    skill_dirs: Optional[List[str]] = None,
    save_events: bool = False,
    **kwargs: Any,
) -> AgentBackend:
    """Create an agent backend instance.

    Args:
        backend_name: One of 'claude-code', 'codex'.
        model: Model name passed to the agent CLI.
        max_turns: Maximum agent iterations.
        agent_instructions: Path to instruction file or raw text.
        timeout: Per-problem timeout in seconds.
        extra_args: Additional CLI arguments.
        skill_dirs: Paths to skill directories to install in workspaces.
        **kwargs: Backend-specific arguments (e.g., reasoning_effort for codex).

    Returns:
        Configured AgentBackend instance.

    Raises:
        ValueError: If backend_name is not recognized.
    """
    if backend_name not in AGENT_BACKENDS:
        available = ", ".join(AGENT_BACKENDS.keys())
        raise ValueError(
            f"Unknown agent backend: '{backend_name}'. "
            f"Available: {available}"
        )

    cls = AGENT_BACKENDS[backend_name]
    return cls(
        model=model,
        max_turns=max_turns,
        agent_instructions=agent_instructions,
        timeout=timeout,
        extra_args=extra_args,
        skill_dirs=skill_dirs,
        save_events=save_events,
        **kwargs,
    )


def list_agent_backends() -> list[str]:
    """Return list of available agent backend names."""
    return list(AGENT_BACKENDS.keys())
