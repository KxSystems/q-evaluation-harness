# Anthropic Claude Opus 4.8

**Provider**: Anthropic
**Version**: claude-opus-4-8
**Parameters**: Unknown (proprietary)
**License**: Proprietary

## Evaluation Details
- **Dataset**: q-humaneval (164 problems)
- **Grader**: new (post-[PR #7](https://github.com/KxSystems/q-evaluation-harness/pull/7))
- **Date**: 2026-06-25
- **Hardware**: macOS (agent runs via Claude Code CLI; one-shot via OpenRouter/litellm)

## Results Summary

### Agent mode (1 task attempt, iterative tool use, `--timeout 300`)
- **With q-kdb skill (✓):** Pass@1 **88.4%** (145/164) · cost $52.40/run · mean 8.7 turns
- **No skills (✗, clean-room):** Pass@1 **87.2%** (143/164) · cost $24.86/run · mean 5.8 turns
- Skill effect: +1.2 pts (Opus is near its ceiling; the skill mostly swaps which problems it solves)

### One-shot mode (50 samples, temp 0.8, seed 1234, no extended thinking)
- **Pass@1**: 53.2%
- **Pass@5**: 74.4%
- **Pass@10**: 80.7%
- **Pass@20**: 85.7%
- **Pass@50**: 89.6%

## Model Description
Claude Opus 4.8 is Anthropic's flagship model. In agent mode it iteratively writes,
runs, and fixes Q solutions against a live interpreter; in one-shot mode it generates
code in a single pass. The ~34-point gap between one-shot Pass@1 (53.2%) and agent
Pass@1 (87–88%) illustrates how much iteration and tool use contribute on this benchmark.

## Notes
- Agent cost is the Claude Code CLI's API-equivalent figure (subscription-covered in practice).
- The skill-arm results file is a re-grade of the run's solutions; cost/turns are
  computed from the run's per-task `agent_cost_usd` / `agent_num_turns`.
- Result files: `outputs/results_agent_claude-opus-4-8_{skill,noskill}.json`,
  `outputs/results_oneshot_claude-opus-4-8.json`.
