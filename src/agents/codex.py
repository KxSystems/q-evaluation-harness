"""Codex CLI agent backend."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import AgentBackend, AgentResult
from .pricing import openai_cost_usd

logger = logging.getLogger(__name__)

CODEX_SKILL_STEP = """\
## 0. Load the q-kdb skill
Read .agents/skills/q-kdb/SKILL.md before writing any code. It contains
syntax rules, common errors, and idioms you will need. This step is
mandatory for every task.

"""

CODEX_DEFAULT_INSTRUCTIONS = """\
# Workflow

You are solving a Q/kdb+ task. You MUST follow this exact workflow. Do NOT
skip verification.

{skill_step}## 1. Write solution
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
        no_skills: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            max_turns=max_turns,
            agent_instructions=agent_instructions,
            timeout=timeout,
            extra_args=extra_args,
            skill_dirs=skill_dirs,
            save_events=save_events,
            no_skills=no_skills,
        )
        self.reasoning_effort = reasoning_effort

    cli_name = "codex"
    cost_source = "list_price"

    @property
    def name(self) -> str:
        return "codex"

    @property
    def instruction_filename(self) -> str:
        return "AGENTS.md"

    def get_default_instructions(self) -> str:
        # Step 0 (load the q-kdb skill) only applies when a skill is installed
        # in the workspace; omit it for no-skill baseline runs.
        skill_step = CODEX_SKILL_STEP if self.skill_dirs else ""
        return CODEX_DEFAULT_INSTRUCTIONS.format(skill_step=skill_step)

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
            # codex-cli 0.159 removed --full-auto; it was shorthand for this
            # sandbox (run anything, write only inside the workspace).
            "--sandbox",
            "workspace-write",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--cd",
            str(workspace),
            # -o is --output-last-message: the agent's final message as plain
            # text. It is not structured output; usage comes from the events.
            "-o",
            str(workspace / "last_message.txt"),
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

            # Loud warning when a task burns through more turns than we
            # asked for — the codex CLI does not have a turn-cap flag,
            # so we surface this here for visibility.
            actual_turns = metadata.get("num_turns")
            if actual_turns is not None and actual_turns > self.max_turns:
                logger.warning(
                    f"Codex task {task_id_str} used {actual_turns} turns, "
                    f"exceeding the configured cap of {self.max_turns}. "
                    f"The CLI does not enforce the cap; only --timeout "
                    f"({self.timeout}s) limits runaway."
                )

            return AgentResult(
                task_id=task_id,
                success=process.returncode == 0,
                solution_code="",  # Filled by extract_solution
                wall_time_seconds=wall_time,
                num_turns=metadata.get("num_turns"),
                input_tokens=metadata.get("input_tokens"),
                output_tokens=metadata.get("output_tokens"),
                cached_input_tokens=metadata.get("cached_input_tokens"),
                cost_usd=metadata.get("cost_usd"),
                raw_output=stdout,
                error=stderr if process.returncode != 0 else None,
                workspace_path=str(workspace),
                agent_version=self.cli_version,
                metadata=metadata,
            )

        except asyncio.TimeoutError:
            wall_time = time.monotonic() - start_time
            task_id = int(str(task_id_str).split("_")[-1])
            logger.warning(
                f"Codex timed out for {task_id_str} after {wall_time:.1f}s"
            )
            # The spend still happened: recover usage from the partial event
            # stream when it was saved to disk.
            metadata = {}
            if events_fh:
                events_fh.close()
            if events_path.exists():
                metadata = self._parse_output(
                    events_path.read_text(errors="replace"), workspace
                )
            return AgentResult(
                task_id=task_id,
                success=False,
                solution_code="",
                wall_time_seconds=wall_time,
                num_turns=metadata.get("num_turns"),
                input_tokens=metadata.get("input_tokens"),
                output_tokens=metadata.get("output_tokens"),
                cached_input_tokens=metadata.get("cached_input_tokens"),
                cost_usd=metadata.get("cost_usd"),
                error=f"Timed out after {self.timeout}s",
                workspace_path=str(workspace),
                agent_version=self.cli_version,
                metadata=metadata,
            )
        finally:
            if events_fh and not events_fh.closed:
                events_fh.close()
            if stderr_fh and not stderr_fh.closed:
                stderr_fh.close()

    def _parse_output(
        self, stdout: str, workspace: Path
    ) -> Dict[str, Any]:
        """Parse Codex JSONL events for turn count, token usage and cost.

        `codex exec --json` reports usage only in `turn.completed` events and
        never reports dollars, so cost is computed from list prices
        (src/agents/pricing.py). One exec call is one turn, so its usage is
        the whole task; summing handles any multi-turn stream.
        """
        metadata: Dict[str, Any] = {}

        # Modern codex emits per-step events as `item.completed` with an
        # `item.type` of `command_execution`, `agent_message`, or
        # `file_change`. Each represents one discrete agent action and
        # counts as a turn. Older schemas used flat `tool_call` /
        # `function_call` / `action` events, which we still match for
        # backwards compatibility.
        num_turns = 0
        usage_totals = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
        }
        saw_usage = False
        modern_action_item_types = {
            "command_execution",
            "agent_message",
            "file_change",
        }
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                event_type = event.get("type", "")
                if event_type == "item.completed":
                    item = event.get("item", {})
                    if (
                        isinstance(item, dict)
                        and item.get("type") in modern_action_item_types
                    ):
                        num_turns += 1
                elif event_type in (
                    "tool_call",
                    "function_call",
                    "action",
                ):
                    num_turns += 1
                if event_type == "turn.completed":
                    usage = event.get("usage") or {}
                    saw_usage = True
                    for key in usage_totals:
                        usage_totals[key] += usage.get(key) or 0
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue

        if num_turns > 0:
            metadata["num_turns"] = num_turns
        if saw_usage:
            metadata.update(usage_totals)
            metadata["cost_usd"] = openai_cost_usd(
                self.model,
                usage_totals["input_tokens"],
                usage_totals["cached_input_tokens"],
                usage_totals["output_tokens"],
            )

        return metadata
