"""Codex CLI agent backend."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import AgentBackend, AgentResult

logger = logging.getLogger(__name__)

CODEX_DEFAULT_INSTRUCTIONS = """\
# Workflow

You are solving a Q/kdb+ task. You MUST follow this exact workflow. Do NOT
skip verification.

## 0. Load the q-kdb skill
Read .agents/skills/q-kdb/SKILL.md before writing any code. It contains
syntax rules, common errors, and idioms you will need. This step is
mandatory for every task.

## 1. Write solution
Read problem.md. Write your Q function in solution.q.

## 2. Check for parse errors
Run: q solution.q -q <<< "exit 0"
If there is any error, fix solution.q and run again.

## 3. Test with examples
Pick 2-3 examples from problem.md. Test by passing solution.q as q's script
argument so the function is loaded before stdin is read:
  q solution.q -q <<< "show FUNC[arg1;arg2]; exit 0"
where FUNC is the function name from solution.q. Do not chain commands
after `\\\\l` on the same line — the system command consumes the rest of the
line and the rest of your statements become part of the file path.
If output is wrong, fix and re-test.

## 4. Iterate
Repeat steps 2-3 until the solution loads cleanly and returns correct values.
Do NOT finish until verification passes.
"""


class CodexBackend(AgentBackend):
    """Backend that invokes Codex CLI in headless mode.

    Uses `codex exec` with --json for structured event output.
    Agent instructions are injected via AGENTS.md in the workspace.
    """

    def __init__(
        self,
        model: str,
        reasoning_effort: str = "high",
        max_turns: int = 10,
        agent_instructions: Optional[str] = None,
        timeout: float = 300.0,
        extra_args: Optional[List[str]] = None,
        skill_dirs: Optional[List[str]] = None,
        save_events: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            max_turns=max_turns,
            agent_instructions=agent_instructions,
            timeout=timeout,
            extra_args=extra_args,
            skill_dirs=skill_dirs,
            save_events=save_events,
        )
        self.reasoning_effort = reasoning_effort

    @property
    def name(self) -> str:
        return "codex"

    @property
    def instruction_filename(self) -> str:
        return "AGENTS.md"

    def get_default_instructions(self) -> str:
        return CODEX_DEFAULT_INSTRUCTIONS

    async def invoke(
        self,
        prompt: str,
        workspace: Path,
    ) -> AgentResult:
        """Invoke codex exec in headless mode."""
        task_id_str = workspace.name
        workspace = workspace.resolve()

        cmd = [
            "codex",
            "exec",
            "--model",
            self.model,
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "--full-auto",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--cd",
            str(workspace),
            "-o",
            str(workspace / "result.json"),
        ]

        cmd.extend(self.extra_args)

        # Pass prompt via stdin (using "-") rather than as a positional
        # argument — avoids issues with multi-line text and special chars.
        cmd.append("-")

        logger.debug(
            f"Invoking Codex for {task_id_str}: {' '.join(cmd[:8])}..."
        )

        events_path = workspace / "events.jsonl"
        stderr_path = workspace / "events.stderr.log"
        events_fh = open(events_path, "wb") if self.save_events else None
        stderr_fh = open(stderr_path, "wb") if self.save_events else None

        start_time = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=events_fh if events_fh else asyncio.subprocess.PIPE,
                stderr=stderr_fh if stderr_fh else asyncio.subprocess.PIPE,
                cwd=str(workspace),
            )

            if events_fh:
                # stdout AND stderr stream directly to files so the OS pipe
                # buffer cannot fill (which would deadlock the child on
                # stderr writes), and partial output survives a timeout
                # cancel. Manage stdin/wait manually.
                if process.stdin:
                    try:
                        process.stdin.write(prompt.encode("utf-8"))
                        await process.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        process.stdin.close()
                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    raise
                events_fh.close()
                events_fh = None
                stderr_fh.close()
                stderr_fh = None
                stdout = (
                    events_path.read_text(errors="replace")
                    if events_path.exists()
                    else ""
                )
                stderr_bytes = (
                    stderr_path.read_bytes() if stderr_path.exists() else b""
                )
            else:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(input=prompt.encode("utf-8")),
                    timeout=self.timeout,
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")

            wall_time = time.monotonic() - start_time
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            if process.returncode != 0:
                logger.warning(
                    f"Codex exited with code {process.returncode} "
                    f"for {task_id_str}: {stderr[:200]}"
                )

            metadata = self._parse_output(stdout, workspace)
            task_id = int(str(task_id_str).split("_")[-1])

            return AgentResult(
                task_id=task_id,
                success=process.returncode == 0,
                solution_code="",  # Filled by extract_solution
                wall_time_seconds=wall_time,
                num_turns=metadata.get("num_turns"),
                input_tokens=metadata.get("input_tokens"),
                output_tokens=metadata.get("output_tokens"),
                cost_usd=metadata.get("cost_usd"),
                raw_output=stdout,
                error=stderr if process.returncode != 0 else None,
                workspace_path=str(workspace),
                metadata=metadata,
            )

        except asyncio.TimeoutError:
            wall_time = time.monotonic() - start_time
            task_id = int(str(task_id_str).split("_")[-1])
            logger.warning(
                f"Codex timed out for {task_id_str} after {wall_time:.1f}s"
            )
            return AgentResult(
                task_id=task_id,
                success=False,
                solution_code="",
                wall_time_seconds=wall_time,
                error=f"Timed out after {self.timeout}s",
                workspace_path=str(workspace),
            )
        finally:
            if events_fh and not events_fh.closed:
                events_fh.close()
            if stderr_fh and not stderr_fh.closed:
                stderr_fh.close()

    def _parse_output(
        self, stdout: str, workspace: Path
    ) -> Dict[str, Any]:
        """Parse Codex output: JSONL events from stdout + result.json."""
        metadata: Dict[str, Any] = {}

        # Try result.json first (most structured)
        result_file = workspace / "result.json"
        if result_file.exists():
            try:
                data = json.loads(result_file.read_text())
                metadata.update(
                    {
                        "input_tokens": data.get("input_tokens"),
                        "output_tokens": data.get("output_tokens"),
                        "cost_usd": data.get("cost_usd"),
                    }
                )
            except (json.JSONDecodeError, TypeError):
                logger.debug("Failed to parse result.json")

        # Parse JSONL events from stdout for turn count and token usage
        num_turns = 0
        total_input_tokens = 0
        total_output_tokens = 0
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                event_type = event.get("type", "")
                if event_type in (
                    "tool_call",
                    "function_call",
                    "action",
                    "turn.completed",
                ):
                    num_turns += 1
                if event_type == "turn.completed":
                    usage = event.get("usage", {})
                    total_input_tokens += usage.get(
                        "input_tokens", 0
                    )
                    total_output_tokens += usage.get(
                        "output_tokens", 0
                    )
            except (json.JSONDecodeError, TypeError):
                continue

        if num_turns > 0:
            metadata["num_turns"] = num_turns
        if total_input_tokens > 0:
            metadata["input_tokens"] = total_input_tokens
        if total_output_tokens > 0:
            metadata["output_tokens"] = total_output_tokens

        return metadata
