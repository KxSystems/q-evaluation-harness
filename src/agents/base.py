"""Abstract base class for agent backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent solving a single problem."""

    task_id: int
    success: bool
    solution_code: str
    wall_time_seconds: float
    num_turns: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    raw_output: Optional[str] = None
    error: Optional[str] = None
    workspace_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentBackend(ABC):
    """Abstract base class for coding agent backends.

    Each backend wraps a specific agent CLI (Claude Code, Codex, etc.)
    and handles workspace scaffolding, CLI invocation, and output parsing.
    """

    def __init__(
        self,
        model: str,
        max_turns: int = 10,
        agent_instructions: Optional[str] = None,
        timeout: float = 300.0,
        extra_args: Optional[List[str]] = None,
        skill_dirs: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self.agent_instructions = agent_instructions
        self.timeout = timeout
        self.extra_args = extra_args or []
        self.skill_dirs = skill_dirs or []

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'claude-code', 'codex')."""

    @property
    @abstractmethod
    def instruction_filename(self) -> str:
        """Filename for agent instructions in workspace.

        Claude Code reads CLAUDE.md, Codex reads AGENTS.md.
        """

    @abstractmethod
    async def invoke(
        self,
        prompt: str,
        workspace: Path,
    ) -> AgentResult:
        """Invoke the agent CLI on a single problem.

        Args:
            prompt: The full task prompt for the agent.
            workspace: Path to the prepared workspace directory.

        Returns:
            AgentResult with solution and telemetry.
        """

    def get_default_instructions(self) -> str:
        """Return backend-specific default instructions.

        Subclasses override to provide built-in instructions that
        encourage testing and iteration. These are prepended to any
        user-provided agent_instructions.
        """
        return ""

    @staticmethod
    def _merge_instructions(default: str, user: str) -> str:
        """Merge default and user-provided instructions."""
        parts = [p for p in [default.strip(), user.strip()] if p]
        return "\n\n---\n\n".join(parts)

    def prepare_workspace(
        self,
        problem: Dict[str, Any],
        base_dir: Path,
        prompt_text: str,
    ) -> Path:
        """Scaffold a per-problem workspace directory.

        Args:
            problem: Problem dict from the dataset.
            base_dir: Parent directory for all workspaces.
            prompt_text: The formatted prompt to give the agent.

        Returns:
            Path to the created workspace directory.
        """
        task_id = problem.get("task_id", "unknown")
        workspace = base_dir / f"task_{task_id}"
        workspace.mkdir(parents=True, exist_ok=True)

        # Write the prompt/problem description
        (workspace / "problem.md").write_text(prompt_text)

        # Write a starter solution.q with the function stub
        stub = problem.get("prompt", "")
        (workspace / "solution.q").write_text(stub)

        # Build combined instructions: backend defaults + user-provided
        default_instructions = self.get_default_instructions()
        user_instructions = ""
        if self.agent_instructions:
            instructions_path = Path(self.agent_instructions)
            if instructions_path.is_file():
                user_instructions = instructions_path.read_text()
            else:
                user_instructions = self.agent_instructions

        combined = self._merge_instructions(
            default_instructions, user_instructions
        )
        if combined.strip():
            (workspace / self.instruction_filename).write_text(combined)

        # Install skills into workspace for both Claude Code and Codex
        if self.skill_dirs:
            for skill_dir in self.skill_dirs:
                skill_path = Path(skill_dir)
                if not skill_path.is_dir():
                    logger.warning(f"Skill dir not found: {skill_dir}")
                    continue
                skill_name = skill_path.name
                # Claude Code: .claude/skills/<name>/
                claude_dest = workspace / ".claude" / "skills" / skill_name
                claude_dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(skill_path, claude_dest, dirs_exist_ok=True)
                # Codex: .agents/skills/<name>/
                codex_dest = workspace / ".agents" / "skills" / skill_name
                codex_dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(skill_path, codex_dest, dirs_exist_ok=True)

        return workspace

    def extract_solution(self, workspace: Path, entry_point: str) -> str:
        """Extract solution code from the workspace.

        Reads solution.q and extracts the function using the existing
        extraction infrastructure.

        Args:
            workspace: Path to the agent's workspace.
            entry_point: Function name to extract.

        Returns:
            Extracted Q function code, or empty string if not found.
        """
        from ..utils.extraction import extract_function_from_content

        solution_file = workspace / "solution.q"

        if not solution_file.exists():
            # Try alternative filenames the agent might have used
            for alt_name in ["answer.q", "result.q", f"{entry_point}.q"]:
                alt_path = workspace / alt_name
                if alt_path.exists():
                    solution_file = alt_path
                    break

        if not solution_file.exists():
            # Last resort: search for any .q file
            q_files = list(workspace.glob("*.q"))
            if q_files:
                solution_file = q_files[0]
                logger.warning(
                    f"solution.q not found, using {solution_file.name}"
                )
            else:
                return ""

        content = solution_file.read_text()
        if not content.strip():
            return ""

        return extract_function_from_content(content, entry_point)
