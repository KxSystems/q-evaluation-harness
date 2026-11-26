"""Antigravity CLI (`agy`) agent backend, for Gemini models.

Antigravity CLI is Google's coding agent and the official harness for the
Gemini 4 models. Gemini CLI was shut down in its favour.

Interface notes. Flags are checked against `agy --help` on 1.5.13. The output
schema comes from the CLI's GitHub repo (CHANGELOG, release notes and issue
reports), because the official headless docs were not readable when this was
written.

- Headless mode is `agy -p <prompt>`. It runs in the current directory, and
  `--output-format json|stream-json` gives structured output. The backend
  always uses stream-json (checked on 1.5.17): `init`, then `step_update`
  events (one `agent_response` step per model call, each with its own
  `usage`), then a `result` event, each nested as
  `{"event": "<name>", "<name>": {...}}`. agy's `num_turns` is 1 per `-p`
  prompt, so the harness counts `agent_response` steps as turns instead.
- The json result (and the final `result` event of stream-json) carries
  `status` ("SUCCESS" / "ERROR"), `response`, `error`, `num_turns`,
  `duration_seconds`, `finished_reason` (omitted on a normal finish),
  `conversation_id` and `usage`. Before 1.5.13, `status` stayed "ERROR" for runs
  that succeeded after a retry, so use 1.5.13 or later.
- `usage`, as observed from agy 1.5.13 `-p --output-format json`:
  `{"input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens",
  "total_tokens"}`, where total = input + output. So `cache_read_tokens` is NOT
  part of `input_tokens` (unlike Codex and the Gemini API). `_extract_usage`
  adds it back, so the harness's `input_tokens` is all input and
  `cached_input_tokens` is the cached part of it, as for Codex. Gemini API
  key names (`promptTokenCount`, ...) are also accepted, with their own subset
  semantics, in case another auth mode reports them.
- Print mode stops itself after `--print-timeout` (1.5.13 default: no limit;
  older builds: 5m). The harness passes a value past its own --timeout so agy
  never stops a task first. Issue #318 reports
  `-p` runs that hang with no output; the harness timeout kills those.
- Auth: per a maintainer (issue #1411), running `agy -p` inside third-party
  automation is only supported with a Gemini API key (`GEMINI_API_KEY`) or
  Vertex AI / ADC (`AGY_ADC_AUTH`), not a signed-in Google subscription.
  API-key mode also needs `modelProvider: "gemini"` (docs/install); with only
  the key set, agy ignores it and, if nobody is signed in, `-p` hangs. When
  GEMINI_API_KEY is set the backend passes it per session via `--settings`,
  which agy never saves, so the operator's settings.json is left alone.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import AgentBackend, AgentResult
from .pricing import gemini_cost_usd

logger = logging.getLogger(__name__)

ANTIGRAVITY_SKILL_STEP = """\
## 0. Load the q-kdb skill
Read .agents/skills/q-kdb/SKILL.md before writing any code. It contains
syntax rules, common errors, and idioms you will need. This step is
mandatory for every task.

"""

ANTIGRAVITY_DEFAULT_INSTRUCTIONS = """\
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

# Environment variables that select an auth mode agy supports for automation.
AUTOMATION_AUTH_ENV_VARS = ("GEMINI_API_KEY", "AGY_ADC_AUTH")

# Effort levels agy accepts for --effort and as a model-name suffix.
AGY_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Candidate key names inside agy's `usage` object, most specific first.
_INPUT_KEYS = ("input_tokens", "prompt_tokens", "promptTokenCount")
_OUTPUT_KEYS = ("output_tokens", "candidates_tokens", "candidatesTokenCount")
# Cache reads reported on top of input_tokens (agy's own usage object).
_SEPARATE_CACHED_KEYS = ("cache_read_tokens", "cache_read_input_tokens")
# Cache reads already counted in input (Gemini API / OpenAI style).
_SUBSET_CACHED_KEYS = (
    "cached_input_tokens",
    "cached_tokens",
    "cachedContentTokenCount",
)
_THOUGHTS_KEYS = (
    "thoughts_tokens",
    "thinking_tokens",
    "reasoning_tokens",
    "thoughtsTokenCount",
)


def _first_int(usage: Dict[str, Any], keys: tuple) -> Optional[int]:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _extract_usage(usage: Any) -> Dict[str, Optional[int]]:
    """Normalize agy's `usage` object to the harness's token fields.

    Output: `input_tokens` is all input (uncached + cached) and
    `cached_input_tokens` the cached subset of it, the Codex convention that
    pricing expects. Thought tokens are kept separate and billed as output.
    """
    if not isinstance(usage, dict):
        return {}
    input_tokens = _first_int(usage, _INPUT_KEYS)
    cached = _first_int(usage, _SUBSET_CACHED_KEYS)
    separate = _first_int(usage, _SEPARATE_CACHED_KEYS)
    if separate is not None:
        cached = separate
        if input_tokens is not None:
            input_tokens += separate
    out = {
        "input_tokens": input_tokens,
        "output_tokens": _first_int(usage, _OUTPUT_KEYS),
        "cached_input_tokens": cached,
        "thoughts_tokens": _first_int(usage, _THOUGHTS_KEYS),
    }
    return {k: v for k, v in out.items() if v is not None}


class AntigravityBackend(AgentBackend):
    """Backend that invokes Antigravity CLI (`agy -p`) in headless mode.

    Agent instructions are written to AGENTS.md, one of the rule files agy
    reads. Skills go to .agents/skills/ (done by prepare_workspace), and the
    default instructions name the SKILL.md path explicitly, so the skill is
    found whether or not agy discovers that directory on its own.
    """

    cli_name = "agy"
    cost_source = "list_price"

    def __init__(
        self,
        model: str,
        reasoning_effort: Optional[str] = None,
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
        # Passed as --effort (low|medium|high|xhigh|max); None leaves agy's
        # default for the model. agy also encodes effort in the model name
        # (gemini-4-hydrogen-high). A suffixed name plus a different --effort
        # makes every `agy -p` exit with "conflicts with --effort", which
        # would score as a run of failures, so refuse it here. A bare alias
        # (gemini-4-hydrogen) resolves to a default that depends on the
        # operator's agy settings, so prefer suffixed names for recorded runs.
        suffix = model.rsplit("-", 1)[-1]
        if (
            reasoning_effort
            and suffix in AGY_EFFORT_LEVELS
            and suffix != reasoning_effort
        ):
            raise ValueError(
                f"--model {model} already selects effort '{suffix}'; drop "
                f"--reasoning-effort {reasoning_effort} or use a model name "
                f"without the effort suffix"
            )
        self.reasoning_effort = reasoning_effort
        if not any(os.environ.get(v) for v in AUTOMATION_AUTH_ENV_VARS):
            logger.warning(
                "Neither GEMINI_API_KEY nor AGY_ADC_AUTH is set, so agy will "
                "use its signed-in account. Antigravity's maintainers say "
                "running `agy -p` from third-party automation is only "
                "supported with a Gemini API key or Vertex AI, and that "
                "subscription quota is shared with interactive use."
            )
        if no_skills:
            # --disable-slash-commands (see build_command) blocks skills, but
            # agy has no flag to ignore user-level MCP servers or plugins.
            logger.warning(
                "--no-skills: skills are disabled, but agy still loads "
                "user-level MCP servers and plugins. Check "
                "~/.gemini/antigravity-cli/ is clean for a true clean room."
            )

    @property
    def name(self) -> str:
        return "antigravity"

    @property
    def instruction_filename(self) -> str:
        return "AGENTS.md"

    def get_default_instructions(self) -> str:
        skill_step = ANTIGRAVITY_SKILL_STEP if self.skill_dirs else ""
        return ANTIGRAVITY_DEFAULT_INSTRUCTIONS.format(skill_step=skill_step)

    def build_command(self, prompt: str) -> List[str]:
        # Leave agy 30s beyond the harness timeout so the harness, not agy's
        # print timeout, decides when a task has run too long.
        print_timeout = int(self.timeout) + 30
        cmd = [
            "agy",
            "-p",
            prompt,
            "--model",
            self.model,
            # Always stream: the step events are the only source of a real
            # turn count, and they let usage survive a timeout.
            "--output-format",
            "stream-json",
            # Headless runs must not stop to ask for approval of shell
            # commands or file edits.
            "--dangerously-skip-permissions",
            "--print-timeout",
            f"{print_timeout}s",
        ]
        # API-key mode: agy reads GEMINI_API_KEY only when modelProvider is
        # "gemini". Skip if the operator passes their own --settings, since a
        # second --settings flag would conflict with theirs.
        if os.environ.get("GEMINI_API_KEY") and "--settings" not in self.extra_args:
            cmd += ["--settings", json.dumps({"modelProvider": "gemini"})]
        if self.reasoning_effort:
            cmd += ["--effort", self.reasoning_effort]
        # Clean-room baseline: "Disable slash command and skill expansion in
        # print mode" (agy --help), so no global skill reaches the agent.
        if self.no_skills:
            cmd.append("--disable-slash-commands")
        cmd.extend(self.extra_args)
        return cmd

    async def invoke(
        self,
        prompt: str,
        workspace: Path,
    ) -> AgentResult:
        """Invoke agy in headless print mode."""
        task_id_str = workspace.name
        workspace = workspace.resolve()
        task_id = int(str(task_id_str).split("_")[-1])
        cmd = self.build_command(prompt)

        logger.debug(f"Invoking Antigravity for {task_id_str}")

        events_path = workspace / "events.jsonl"
        stderr_path = workspace / "events.stderr.log"
        events_fh = open(events_path, "wb") if self.save_events else None
        stderr_fh = open(stderr_path, "wb") if self.save_events else None

        start_time = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                # Issue #318: some agy builds wait on an open stdin in print
                # mode. The prompt is an argument, so give it none.
                stdin=asyncio.subprocess.DEVNULL,
                stdout=events_fh if events_fh else asyncio.subprocess.PIPE,
                stderr=stderr_fh if stderr_fh else asyncio.subprocess.PIPE,
                cwd=str(workspace),
            )

            if events_fh:
                # Stream straight to files so a full pipe cannot deadlock the
                # child and partial output survives a timeout.
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
                stdout = events_path.read_text(errors="replace")
                stderr = stderr_path.read_text(errors="replace")
            else:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")

            wall_time = time.monotonic() - start_time
            metadata = self._parse_output(stdout)

            status = metadata.get("status")
            success = process.returncode == 0 and status in (None, "SUCCESS")
            if not success:
                logger.warning(
                    f"Antigravity failed for {task_id_str} "
                    f"(exit {process.returncode}, status {status}): "
                    f"{(metadata.get('error') or stderr)[:200]}"
                )

            actual_turns = metadata.get("num_turns")
            if actual_turns is not None and actual_turns > self.max_turns:
                logger.warning(
                    f"Antigravity task {task_id_str} used {actual_turns} "
                    f"turns, exceeding the configured cap of "
                    f"{self.max_turns}. The CLI does not enforce the cap; "
                    f"only --timeout ({self.timeout}s) limits runaway."
                )

            error = None
            if not success:
                error = metadata.get("error") or stderr or (
                    f"agy exited with code {process.returncode}"
                )

            return self._result(
                task_id, success, wall_time, metadata, workspace,
                raw_output=stdout, error=error,
            )

        except asyncio.TimeoutError:
            wall_time = time.monotonic() - start_time
            logger.warning(
                f"Antigravity timed out for {task_id_str} "
                f"after {wall_time:.1f}s"
            )
            # The spend still happened: recover usage from any partial
            # event stream saved to disk.
            metadata: Dict[str, Any] = {}
            if events_fh:
                events_fh.close()
            if self.save_events and events_path.exists():
                metadata = self._parse_output(
                    events_path.read_text(errors="replace")
                )
            return self._result(
                task_id, False, wall_time, metadata, workspace,
                error=f"Timed out after {self.timeout}s",
            )
        finally:
            if events_fh and not events_fh.closed:
                events_fh.close()
            if stderr_fh and not stderr_fh.closed:
                stderr_fh.close()

    def _result(
        self,
        task_id: int,
        success: bool,
        wall_time: float,
        metadata: Dict[str, Any],
        workspace: Path,
        raw_output: Optional[str] = None,
        error: Optional[str] = None,
    ) -> AgentResult:
        return AgentResult(
            task_id=task_id,
            success=success,
            solution_code="",  # Filled by extract_solution
            wall_time_seconds=wall_time,
            num_turns=metadata.get("num_turns"),
            input_tokens=metadata.get("input_tokens"),
            output_tokens=metadata.get("output_tokens"),
            cached_input_tokens=metadata.get("cached_input_tokens"),
            cost_usd=metadata.get("cost_usd"),
            raw_output=raw_output,
            error=error,
            workspace_path=str(workspace),
            agent_version=metadata.get("agent_version") or self.cli_version,
            metadata=metadata,
        )

    def _parse_output(self, stdout: str) -> Dict[str, Any]:
        """Parse `--output-format json` or `stream-json` output.

        json is one object; stream-json is JSONL whose last `result` event
        holds the same fields. Either way the result object is used.

        agy 1.5.17 stream-json names the event under `event` and nests its
        payload under a key of the same name:
        `{"event": "result", "result": {...}}`,
        `{"event": "step_update", "step_update": {...}}`. A flat
        `{"type": "result", ...}` shape is also accepted.

        Turns: agy's `num_turns` counts prompts (always 1 for `-p`), so it is
        kept as `agy_num_turns` and `num_turns` is the number of finished
        `agent_response` steps, i.e. model calls, which is what other backends
        report. Each such step carries its own `usage`; they sum to the result
        usage, so they stand in for it when no result event arrived (timeout).
        """
        result: Dict[str, Any] = {}
        agent_version: Optional[str] = None
        response_steps: set = set()
        step_usage: Dict[str, int] = {}

        text = stdout.strip()
        try:
            parsed = json.loads(text) if text else None
            candidates = [parsed] if isinstance(parsed, dict) else []
        except json.JSONDecodeError:
            candidates = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    candidates.append(event)

        for event in candidates:
            event_type = event.get("event") or event.get("type")
            # 1.5.17 nests the payload under the event name; older shapes
            # are flat.
            payload = event.get(event_type) if event_type else None
            if not isinstance(payload, dict):
                payload = event
            if event_type == "init":
                agent_version = (
                    payload.get("version")
                    or payload.get("cli_version")
                    or event.get("version")
                    or agent_version
                )
            elif event_type == "step_update":
                if (
                    payload.get("state") == "DONE"
                    and payload.get("step_type") == "agent_response"
                ):
                    response_steps.add(payload.get("step_index"))
                    usage = payload.get("usage")
                    if isinstance(usage, dict):
                        for key, value in usage.items():
                            if isinstance(value, int):
                                step_usage[key] = step_usage.get(key, 0) + value
            elif event_type == "result" or (
                event_type is None and ("status" in event or "usage" in event)
            ):
                result = payload

        metadata: Dict[str, Any] = {}
        if agent_version:
            metadata["agent_version"] = str(agent_version)
        step_turns = len(response_steps) if response_steps else None
        if not result:
            if not step_usage:
                logger.debug("No result object in agy output")
                metadata["result_text"] = stdout
                return metadata
            # Cut off before the result event: rebuild usage from the steps.
            logger.debug("No result event; using per-step usage")
            result = {"usage": step_usage}

        metadata.update(
            {
                "status": result.get("status"),
                "finished_reason": result.get("finished_reason"),
                "error": result.get("error"),
                "num_turns": (
                    step_turns if step_turns is not None
                    else result.get("num_turns")
                ),
                "agy_num_turns": result.get("num_turns"),
                "conversation_id": result.get("conversation_id"),
                "duration_seconds": result.get("duration_seconds"),
                "result_text": result.get("response", ""),
            }
        )
        raw_usage = result.get("usage")
        if raw_usage is not None:
            # Keep the raw object so field names can be checked against real
            # runs and _extract_usage corrected if needed.
            metadata["raw_usage"] = raw_usage
        tokens = _extract_usage(raw_usage)
        metadata.update(tokens)
        if "input_tokens" in tokens and "output_tokens" in tokens:
            metadata["cost_usd"] = gemini_cost_usd(
                self.model,
                tokens["input_tokens"],
                tokens.get("cached_input_tokens"),
                tokens["output_tokens"] + tokens.get("thoughts_tokens", 0),
            )
        return metadata
