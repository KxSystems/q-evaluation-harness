# OpenAI GPT-5.5

**Provider**: OpenAI
**Version**: gpt-5.5
**Parameters**: Unknown (proprietary)
**License**: Proprietary

## Evaluation Details
- **Dataset**: q-humaneval (164 problems)
- **Grader**: new (post-[PR #7](https://github.com/KxSystems/q-evaluation-harness/pull/7))
- **Date**: 2026-06-25
- **Hardware**: macOS (agent runs via Codex CLI, `--reasoning-effort high`)

## Results Summary

### Agent mode (1 task attempt, iterative tool use, `--timeout 300`, high reasoning)
- **With q-kdb skill (✓):** Pass@1 **94.5%** (155/164) · mean 18.8 turns
- **No skills (✗, clean-room):** Pass@1 **88.4%** (145/164) · mean 20.3 turns
- Skill effect: +6.1 pts — the largest skill benefit observed in this pass

### One-shot mode
- *Run pending* (not yet evaluated)

## Model Description
GPT-5.5 is OpenAI's flagship reasoning model, evaluated in agent mode via the Codex CLI.
It tops the new-grader agent leaderboard at 94.5% with the q-kdb skill installed.

## Notes
- **Cost is not captured** for the Codex backend — the CLI records token usage but does
  not surface a dollar cost, so cost is omitted from the leaderboard.
- **Cross-check:** an independent prior agentic GPT-5.5 run re-grades to 93.3% (153/164)
  under the new grader, consistent with the fresh 94.5% within run-to-run variance.
- The skill-arm results file is a re-grade of the run's solutions; turns are computed
  from the run's per-task `agent_num_turns`.
- Result files: `outputs/results_agent_gpt-5.5_{skill,noskill}.json`;
  reference cross-check: `outputs/results_agent_gpt-5.5_reference_newgrader.json`.
