"""Tests for the agent evaluation module.

Tests mock subprocess calls so they don't require real agent CLIs.
"""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import AgentBackend, AgentResult
from src.agents.claude_code import ClaudeCodeBackend
from src.agents.codex import CodexBackend
from src.agents.factory import create_agent_backend, list_agent_backends


# --- Factory Tests ---


class TestAgentFactory:
    """Tests for agent backend factory."""

    def test_list_backends(self) -> None:
        backends = list_agent_backends()
        assert "claude-code" in backends
        assert "codex" in backends

    def test_create_claude_code_backend(self) -> None:
        backend = create_agent_backend("claude-code", model="opus")
        assert isinstance(backend, ClaudeCodeBackend)
        assert backend.name == "claude-code"
        assert backend.model == "opus"
        assert backend.instruction_filename == "CLAUDE.md"

    def test_create_codex_backend(self) -> None:
        backend = create_agent_backend(
            "codex", model="gpt-5.3-codex", reasoning_effort="high"
        )
        assert isinstance(backend, CodexBackend)
        assert backend.name == "codex"
        assert backend.model == "gpt-5.3-codex"
        assert backend.instruction_filename == "AGENTS.md"
        assert backend.reasoning_effort == "high"

    def test_create_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown agent backend"):
            create_agent_backend("nonexistent", model="test")

    def test_backend_params_flow(self) -> None:
        backend = create_agent_backend(
            "claude-code",
            model="sonnet",
            max_turns=20,
            agent_instructions="/tmp/instructions.md",
            timeout=600.0,
            extra_args=["--verbose"],
        )
        assert backend.max_turns == 20
        assert backend.agent_instructions == "/tmp/instructions.md"
        assert backend.timeout == 600.0
        assert backend.extra_args == ["--verbose"]


# --- Workspace Scaffolding Tests ---


class TestWorkspaceScaffolding:
    """Tests for workspace preparation."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        temp = Path(tempfile.mkdtemp())
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def sample_problem(self) -> dict:
        return {
            "task_id": 42,
            "prompt": (
                "has_close_elements:{[x;y]\n    / function body\n    }"
            ),
            "entry_point": "has_close_elements",
            "tests": "def check(candidate):\n    assert candidate([1.0, 2.0], 0.5) == True",
            "test_setup_code": "",
        }

    def test_workspace_scaffolding(
        self, temp_dir: Path, sample_problem: dict
    ) -> None:
        backend = create_agent_backend("claude-code", model="opus")
        workspace = backend.prepare_workspace(
            sample_problem, temp_dir, "Write the function"
        )

        assert (workspace / "problem.md").exists()
        assert (workspace / "solution.q").exists()
        assert workspace.name == "task_42"

    def test_solution_stub_written(
        self, temp_dir: Path, sample_problem: dict
    ) -> None:
        backend = create_agent_backend("claude-code", model="opus")
        workspace = backend.prepare_workspace(
            sample_problem, temp_dir, "Write the function"
        )

        stub = (workspace / "solution.q").read_text()
        assert "has_close_elements" in stub

    def test_instructions_file_from_path(
        self, temp_dir: Path, sample_problem: dict
    ) -> None:
        # Create a temp instructions file
        instructions_file = temp_dir / "my_instructions.md"
        instructions_file.write_text("Be thorough with Q idioms.")

        backend = create_agent_backend(
            "claude-code",
            model="opus",
            agent_instructions=str(instructions_file),
        )
        ws_dir = temp_dir / "workspaces"
        ws_dir.mkdir()
        workspace = backend.prepare_workspace(
            sample_problem, ws_dir, "Write the function"
        )

        claude_md = workspace / "CLAUDE.md"
        assert claude_md.exists()
        assert "Be thorough with Q idioms." in claude_md.read_text()

    def test_codex_instructions_filename(
        self, temp_dir: Path, sample_problem: dict
    ) -> None:
        backend = create_agent_backend(
            "codex",
            model="gpt-5.3-codex",
            agent_instructions="Use Q adverbs wherever possible.",
        )
        workspace = backend.prepare_workspace(
            sample_problem, temp_dir, "Write the function"
        )

        agents_md = workspace / "AGENTS.md"
        assert agents_md.exists()
        assert "Use Q adverbs" in agents_md.read_text()


# --- Solution Extraction Tests ---


class TestSolutionExtraction:
    """Tests for extracting solutions from agent workspaces."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        temp = Path(tempfile.mkdtemp())
        yield temp
        shutil.rmtree(temp)

    def test_extract_from_solution_q(self, temp_dir: Path) -> None:
        workspace = temp_dir / "task_0"
        workspace.mkdir()
        (workspace / "solution.q").write_text(
            "has_close_elements:{[x;y]\n    any (abs x -/:\\: x) < y\n    }"
        )

        backend = create_agent_backend("claude-code", model="opus")
        code = backend.extract_solution(workspace, "has_close_elements")
        assert "has_close_elements" in code or "{[x;y]" in code

    def test_extract_fallback_to_alt_filename(self, temp_dir: Path) -> None:
        workspace = temp_dir / "task_1"
        workspace.mkdir()
        # No solution.q, but answer.q exists
        (workspace / "answer.q").write_text(
            "my_func:{[x]\n    x + 1\n    }"
        )

        backend = create_agent_backend("claude-code", model="opus")
        code = backend.extract_solution(workspace, "my_func")
        assert code  # Should find something

    def test_extract_empty_workspace_returns_empty(
        self, temp_dir: Path
    ) -> None:
        workspace = temp_dir / "task_2"
        workspace.mkdir()

        backend = create_agent_backend("claude-code", model="opus")
        code = backend.extract_solution(workspace, "no_func")
        assert code == ""


# --- Claude Code Backend Invoke Tests ---


class TestClaudeCodeBackend:
    """Tests for Claude Code CLI invocation (mocked subprocess)."""

    @pytest.fixture
    def backend(self) -> ClaudeCodeBackend:
        return ClaudeCodeBackend(model="opus", max_turns=5, timeout=30.0)

    @pytest.fixture
    def temp_workspace(self) -> Generator[Path, None, None]:
        temp = Path(tempfile.mkdtemp())
        workspace = temp / "task_0"
        workspace.mkdir(parents=True)
        (workspace / "solution.q").write_text("stub:{x}")
        yield workspace
        shutil.rmtree(temp)

    @pytest.mark.asyncio
    async def test_invoke_success(
        self, backend: ClaudeCodeBackend, temp_workspace: Path
    ) -> None:
        json_output = json.dumps(
            {
                "result": "Done",
                "session_id": "abc123",
                "num_turns": 3,
                "usage": {
                    "input_tokens": 1500,
                    "output_tokens": 400,
                },
                "total_cost_usd": 0.05,
            }
        )

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(json_output.encode(), b"")
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await backend.invoke("Write the Q function", temp_workspace)

        assert result.success is True
        assert result.num_turns == 3
        assert result.cost_usd == 0.05
        assert result.input_tokens == 1500
        assert result.error is None

    @pytest.mark.asyncio
    async def test_invoke_failure(
        self, backend: ClaudeCodeBackend, temp_workspace: Path
    ) -> None:
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Error: model unavailable")
        )
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await backend.invoke("Write the Q function", temp_workspace)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_invoke_timeout(
        self, backend: ClaudeCodeBackend, temp_workspace: Path
    ) -> None:
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await backend.invoke("Write the Q function", temp_workspace)

        assert result.success is False
        assert "Timed out" in result.error

    @pytest.mark.asyncio
    async def test_invoke_builds_correct_command(
        self, backend: ClaudeCodeBackend, temp_workspace: Path
    ) -> None:
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"{}", b""))
        mock_process.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ) as mock_exec:
            await backend.invoke("Write function", temp_workspace)

        call_args = mock_exec.call_args[0]
        assert call_args[0] == "claude"
        assert "-p" in call_args
        assert "--model" in call_args
        assert "opus" in call_args
        assert "--output-format" in call_args
        assert "--dangerously-skip-permissions" in call_args


# --- Codex Backend Invoke Tests ---


class TestCodexBackend:
    """Tests for Codex CLI invocation (mocked subprocess)."""

    @pytest.fixture
    def backend(self) -> CodexBackend:
        return CodexBackend(
            model="gpt-5.3-codex",
            reasoning_effort="high",
            max_turns=5,
            timeout=30.0,
        )

    @pytest.fixture
    def temp_workspace(self) -> Generator[Path, None, None]:
        temp = Path(tempfile.mkdtemp())
        workspace = temp / "task_0"
        workspace.mkdir(parents=True)
        (workspace / "solution.q").write_text("stub:{x}")
        yield workspace
        shutil.rmtree(temp)

    @pytest.mark.asyncio
    async def test_invoke_success(
        self, backend: CodexBackend, temp_workspace: Path
    ) -> None:
        # Codex writes result.json
        (temp_workspace / "result.json").write_text(
            json.dumps({"input_tokens": 2000, "output_tokens": 500})
        )

        jsonl_events = "\n".join(
            [
                json.dumps({"type": "turn.completed", "turn": 1}),
                json.dumps({"type": "turn.completed", "turn": 2}),
            ]
        )

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(jsonl_events.encode(), b"")
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await backend.invoke("Write the Q function", temp_workspace)

        assert result.success is True
        assert result.num_turns == 2
        assert result.input_tokens == 2000

    @pytest.mark.asyncio
    async def test_invoke_builds_correct_command(
        self, backend: CodexBackend, temp_workspace: Path
    ) -> None:
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ) as mock_exec:
            await backend.invoke("Write function", temp_workspace)

        call_args = mock_exec.call_args[0]
        assert call_args[0] == "codex"
        assert "exec" in call_args
        assert "--model" in call_args
        assert "gpt-5.3-codex" in call_args
        assert "--full-auto" in call_args


# --- Agent Metrics Tests ---


class TestAgentMetrics:
    """Tests for agent-specific metric calculations."""

    def test_safe_mean(self) -> None:
        from src.agents.runner import _safe_mean

        assert _safe_mean([1.0, 2.0, 3.0]) == 2.0
        assert _safe_mean([]) is None

    def test_safe_median(self) -> None:
        from src.agents.runner import _safe_median

        assert _safe_median([1.0, 2.0, 3.0]) == 2.0
        assert _safe_median([1.0, 2.0, 3.0, 4.0]) == 2.5
        assert _safe_median([]) is None

    def test_safe_percentile(self) -> None:
        from src.agents.runner import _safe_percentile

        values = list(range(100))
        assert _safe_percentile(values, 90) == 90
        assert _safe_percentile([], 90) is None

    def test_calculate_agent_metrics(self) -> None:
        from src.agents.runner import _calculate_agent_metrics

        execution_results = [
            {"task_id": 0, "passed": True, "agent_wall_time": 10.0},
            {"task_id": 1, "passed": False, "agent_wall_time": 20.0},
            {"task_id": 2, "passed": True, "agent_wall_time": 15.0},
        ]
        agent_results = [
            AgentResult(
                task_id=0,
                success=True,
                solution_code="f:{x}",
                wall_time_seconds=10.0,
                num_turns=2,
                cost_usd=0.03,
            ),
            AgentResult(
                task_id=1,
                success=True,
                solution_code="g:{x}",
                wall_time_seconds=20.0,
                num_turns=5,
                cost_usd=0.08,
            ),
            AgentResult(
                task_id=2,
                success=True,
                solution_code="h:{x}",
                wall_time_seconds=15.0,
                num_turns=3,
                cost_usd=0.05,
            ),
        ]

        backend = create_agent_backend("claude-code", model="opus")
        summary = _calculate_agent_metrics(
            execution_results, agent_results, backend, "q-humaneval"
        )

        assert summary["total_solutions"] == 3
        assert summary["passed_solutions"] == 2
        assert summary["evaluation_type"] == "agent"
        assert summary["agent_backend"] == "claude-code"

        metrics = summary["agent_metrics"]
        assert metrics["total_wall_time_seconds"] == 45.0
        assert metrics["mean_wall_time_seconds"] == 15.0
        assert metrics["total_cost_usd"] == pytest.approx(0.16)
        assert metrics["no_solution_count"] == 0


# --- Build Agent Prompt Tests ---


class TestBuildAgentPrompt:
    """Tests for prompt construction."""

    def test_prompt_references_solution_file(self) -> None:
        from src.agents.runner import _build_agent_prompt

        class MockTemplate:
            def format(self, problem: dict) -> str:
                return f"Write Q function: {problem['prompt']}"

        problem = {
            "prompt": "my_func:{[x]\n    / body\n    }",
            "entry_point": "my_func",
        }

        prompt = _build_agent_prompt(problem, MockTemplate())
        assert "solution.q" in prompt
        assert "my_func" in prompt
