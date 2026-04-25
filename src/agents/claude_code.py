"""Claude Code agent backend."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

from .base import AgentBackend, AgentResult

logger = logging.getLogger(__name__)

CLAUDE_CODE_DEFAULT_INSTRUCTIONS = """\
# Workflow

Always verify your solution before finishing.

## 0. Load relevant skills
Skills live under .claude/skills/. List that directory and read the SKILL.md
for any skill whose front-matter description matches this task (e.g., a Q/kdb
skill is directly relevant — use it for syntax, idioms, and error diagnosis).
Skip this step only if no skill is relevant.

## 1. Write solution
Read problem.md. Write your Q function in solution.q.

## 2. Check for parse errors
Run: q solution.q -q <<< "exit 0"
If there is any error, fix solution.q and run again.

## 3. Test with examples
Pick 2-3 examples from problem.md. Test in Q:
  q -q <<< "\\\\l solution.q; show FUNC[arg1;arg2]; exit 0"
where FUNC is the function name from solution.q.
If output is wrong, fix and re-test.

## 4. Iterate
Repeat steps 2-3 until the solution loads cleanly and returns correct values.
"""


class ClaudeCodeBackend(AgentBackend):
    """Backend that invokes Claude Code CLI in headless mode.

    Uses `claude -p` with --output-format json for structured output.
    Agent instructions are injected via CLAUDE.md in the workspace
    and/or --append-system-prompt-file.
    """

    @property
    def name(self) -> str:
        return "claude-code"

    @property
    def instruction_filename(self) -> str:
        return "CLAUDE.md"

    def get_default_instructions(self) -> str:
        return CLAUDE_CODE_DEFAULT_INSTRUCTIONS

    async def invoke(
        self,
        prompt: str,
        workspace: Path,
    ) -> AgentResult:
        """Invoke claude CLI in headless mode."""
        task_id_str = workspace.name

        cmd = [
            "claude",
            "-p",
            prompt,
            "--model",
            self.model,
            "--output-format",
            "stream-json" if self.save_events else "json",
            "--max-turns",
            str(self.max_turns),
            "--dangerously-skip-permissions",
            "--no-session-persistence",
        ]

        # stream-json requires the CLI's --verbose flag to emit per-event records
        if self.save_events:
            cmd.append("--verbose")

        # Note: agent instructions are written as CLAUDE.md in the workspace
        # by prepare_workspace(). Claude Code auto-reads CLAUDE.md from its
        # cwd, so no --append-system-prompt-file flag is needed.

        cmd.extend(self.extra_args)

        logger.debug(f"Invoking Claude Code for {task_id_str}")

        # Remove CLAUDECODE env var to allow nested invocations
        # (e.g. when the harness is run from within a Claude Code session)
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        events_path = workspace / "events.jsonl"
        events_fh = open(events_path, "wb") if self.save_events else None

        start_time = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=events_fh if events_fh else asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workspace),
                env=env,
            )

            if events_fh:
                # stdout streams directly to events.jsonl so partial output
                # survives a timeout cancel.
                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    raise
                stderr_bytes = (
                    await process.stderr.read() if process.stderr else b""
                )
                events_fh.close()
                events_fh = None
                stdout = (
                    events_path.read_text(errors="replace")
                    if events_path.exists()
                    else ""
                )
            else:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")

            wall_time = time.monotonic() - start_time
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            if process.returncode != 0:
                logger.warning(
                    f"Claude Code exited with code {process.returncode} "
                    f"for {task_id_str}: {stderr[:200]}"
                )

            if self.save_events:
                metadata = self._parse_stream_json(stdout)
            else:
                metadata = self._parse_output(stdout)
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
                f"Claude Code timed out for {task_id_str} "
                f"after {wall_time:.1f}s"
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

    def _parse_output(self, stdout: str) -> Dict[str, Any]:
        """Parse Claude Code JSON output for telemetry."""
        try:
            data = json.loads(stdout)
            usage = data.get("usage", {})
            return {
                "session_id": data.get("session_id"),
                "num_turns": data.get("num_turns"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cost_usd": data.get("total_cost_usd"),
                "result_text": data.get("result", ""),
            }
        except (json.JSONDecodeError, TypeError):
            logger.debug(
                "Failed to parse Claude Code JSON output, "
                "treating as plain text"
            )
            return {"result_text": stdout}

    def _parse_stream_json(self, stdout: str) -> Dict[str, Any]:
        """Parse Claude Code stream-json (JSONL) output for telemetry.

        The final event of type 'result' carries the same aggregate fields
        as the single-shot json format.
        """
        result_event: Dict[str, Any] = {}
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if event.get("type") == "result":
                result_event = event

        if not result_event:
            logger.debug("No 'result' event in claude stream-json output")
            return {"result_text": stdout}

        usage = result_event.get("usage", {})
        return {
            "session_id": result_event.get("session_id"),
            "num_turns": result_event.get("num_turns"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cost_usd": result_event.get("total_cost_usd"),
            "result_text": result_event.get("result", ""),
        }
