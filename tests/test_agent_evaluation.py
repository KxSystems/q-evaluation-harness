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

from src.agents.base import AgentBackend, AgentResult, parse_cli_version
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

    # Shape of a real codex-cli 0.159.3 `exec --json` stream (task 0, gpt-5.5).
    CODEX_EVENTS = [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
        {"type": "item.completed", "item": {"type": "command_execution"}},
        {"type": "item.completed", "item": {"type": "file_change"}},
        {"type": "item.completed", "item": {"type": "reasoning"}},
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 133147,
                "cached_input_tokens": 112640,
                "cache_write_input_tokens": 0,
                "output_tokens": 3083,
                "reasoning_output_tokens": 1046,
            },
        },
    ]

    @pytest.mark.asyncio
    async def test_invoke_success(self, temp_workspace: Path) -> None:
        backend = CodexBackend(model="gpt-5.5", timeout=30.0)
        # -o writes the last message as plain text; it must not break parsing.
        (temp_workspace / "last_message.txt").write_text("Implemented solution.q")
        jsonl_events = "\n".join(json.dumps(e) for e in self.CODEX_EVENTS)

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(jsonl_events.encode(), b"")
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await backend.invoke("Write the Q function", temp_workspace)

        assert result.success is True
        assert result.num_turns == 3  # reasoning items are not actions
        assert result.input_tokens == 133147
        assert result.cached_input_tokens == 112640
        assert result.output_tokens == 3083
        # (133147-112640)*5 + 112640*0.50 + 3083*30, per 1M tokens
        assert result.cost_usd == pytest.approx(0.251345)
        assert result.metadata["reasoning_output_tokens"] == 1046

    def test_unknown_model_has_no_cost(self, backend: CodexBackend) -> None:
        events = "\n".join(json.dumps(e) for e in self.CODEX_EVENTS)
        metadata = backend._parse_output(events, Path("."))
        assert metadata["input_tokens"] == 133147
        assert metadata["cost_usd"] is None  # gpt-5.3-codex is not priced

    def test_no_usage_event_means_no_tokens(self, backend: CodexBackend) -> None:
        metadata = backend._parse_output(
            json.dumps({"type": "turn.started"}), Path(".")
        )
        assert "input_tokens" not in metadata
        assert "cost_usd" not in metadata

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
        assert "--full-auto" not in call_args  # removed in codex-cli 0.159
        i = call_args.index("--sandbox")
        assert call_args[i + 1] == "workspace-write"


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


# --- Agent CLI Version Tests ---


class TestAgentVersion:
    """The agent CLI version is recorded per task and per run."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("2.1.286 (Claude Code)\n", "2.1.286"),
            ("codex-cli 0.48.0", "0.48.0"),
            ("1.0.0-beta.2", "1.0.0-beta.2"),
            ("dev build\n", "dev build"),
            ("", None),
        ],
    )
    def test_parse_cli_version(self, text: str, expected: str) -> None:
        assert parse_cli_version(text) == expected

    def test_probe_version_runs_once(self) -> None:
        backend = ClaudeCodeBackend(model="opus")
        completed = MagicMock(stdout="2.1.286 (Claude Code)\n", stderr="")
        with patch("src.agents.base.subprocess.run", return_value=completed) as run:
            assert backend.probe_version() == "2.1.286"
            assert backend.probe_version() == "2.1.286"
        run.assert_called_once()
        assert run.call_args.args[0] == ["claude", "--version"]

    def test_probe_version_missing_cli(self) -> None:
        backend = CodexBackend(model="gpt-5.5")
        with patch("src.agents.base.subprocess.run", side_effect=FileNotFoundError):
            assert backend.probe_version() is None

    def test_stream_json_reads_init_version(self) -> None:
        backend = ClaudeCodeBackend(model="opus", save_events=True)
        events = "\n".join(
            json.dumps(e)
            for e in [
                {"type": "system", "subtype": "init", "claude_code_version": "2.1.290"},
                {"type": "result", "num_turns": 2, "usage": {}, "total_cost_usd": 0.1},
            ]
        )
        assert backend._parse_stream_json(events)["agent_version"] == "2.1.290"

    def test_metrics_record_versions(self) -> None:
        from src.agents.runner import _calculate_agent_metrics

        backend = create_agent_backend("claude-code", model="opus")
        backend.cli_version = "2.1.286"
        agent_results = [
            AgentResult(task_id=i, success=True, solution_code="f:{x}",
                        wall_time_seconds=1.0, agent_version=v)
            for i, v in enumerate(["2.1.286", "2.1.290", None])
        ]
        execution_results = [
            {"task_id": i, "passed": True, "agent_wall_time": 1.0} for i in range(3)
        ]
        summary = _calculate_agent_metrics(
            execution_results, agent_results, backend, "q-humaneval"
        )
        assert summary["agent_cli_version"] == "2.1.286"
        assert summary["agent_versions_seen"] == ["2.1.286", "2.1.290"]


class TestClaudeCodeIsolation:
    """Agents must not inherit the operator's Claude Code auto-memory."""

    @pytest.mark.asyncio
    async def test_auto_memory_disabled(self, tmp_path: Path) -> None:
        workspace = tmp_path / "task_0"
        workspace.mkdir()
        backend = ClaudeCodeBackend(model="opus", timeout=30.0)
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"{}", b""))
        mock_process.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as ex:
            await backend.invoke("Write function", workspace)
        assert ex.call_args.kwargs["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
# --- Pricing Tests ---


class TestPricing:
    """List-price cost for backends whose CLI reports only tokens."""

    def test_openai_cost_formula(self) -> None:
        from src.agents.pricing import openai_cost_usd

        # 1M uncached in, 1M cached in, 1M out at gpt-5.5 list prices
        cost = openai_cost_usd("gpt-5.5", 2_000_000, 1_000_000, 1_000_000)
        assert cost == pytest.approx(5.00 + 0.50 + 30.00)

    def test_cached_never_exceeds_input(self) -> None:
        from src.agents.pricing import openai_cost_usd

        cost = openai_cost_usd("gpt-5.4", 100, 500, 0)
        assert cost == pytest.approx(100 * 0.25 / 1_000_000)

    def test_unknown_model_is_none(self) -> None:
        from src.agents.pricing import openai_cost_usd

        assert openai_cost_usd("gpt-9-imaginary", 1000, 0, 1000) is None

    def test_cost_source_per_backend(self) -> None:
        assert ClaudeCodeBackend(model="opus").cost_source == "cli_reported"
        assert CodexBackend(model="gpt-5.5").cost_source == "list_price"

    def test_claude_cost_formula(self) -> None:
        from src.agents.pricing import claude_cost_usd

        # 1M each of input, output, cache read and 1h cache write at Sonnet 5.5
        usage = {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "cache_read_input_tokens": 1_000_000,
            "cache_creation_input_tokens": 1_000_000,
            "cache_creation": {"ephemeral_1h_input_tokens": 1_000_000},
        }
        assert claude_cost_usd("claude-sonnet-5-5", usage) == pytest.approx(
            2.00 + 10.00 + 0.10 + 4.00
        )

    def test_claude_cache_write_defaults_to_1h_tier(self) -> None:
        from src.agents.pricing import claude_cost_usd

        usage = {"cache_creation_input_tokens": 1_000_000}
        assert claude_cost_usd("claude-opus-5-5", usage) == pytest.approx(8.00)

    def test_claude_unknown_model_is_none(self) -> None:
        from src.agents.pricing import claude_cost_usd

        assert claude_cost_usd("claude-imaginary-9", {"input_tokens": 10}) is None

    def test_haiku_55_needs_prompt_length_proof(self) -> None:
        from src.agents.pricing import claude_cost_usd

        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        # Totals cannot show per-request prompt length: refuse rather than guess
        assert claude_cost_usd("claude-haiku-5-5", usage) is None
        assert claude_cost_usd("claude-haiku-5-5", usage, 100_001) is None
        assert claude_cost_usd("claude-haiku-5-5", usage, 100_000) == pytest.approx(
            0.10 + 0.50
        )
