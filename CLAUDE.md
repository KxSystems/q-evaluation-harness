# CLAUDE.md — Development Guidance

## Project Overview

Q Evaluation Harness — an open-source Python framework by KX for evaluating LLMs on Q/kdb+ code generation tasks. Includes the Q-HumanEval dataset (164 problems) and supports both standard multi-sample evaluation (Pass@k) and agent-mode evaluation (single-attempt with tool use).

## Build and Run

```bash
# Install
poetry install
eval $(poetry env activate)

# Run tests
poetry run pytest

# Standard evaluation (50 samples, Pass@k)
qeval run q-humaneval <model_name> --num-samples 50

# Agent evaluation (single attempt with tool use)
qeval agent-run q-humaneval --backend claude-code --model claude-opus-4-6
qeval agent-run q-humaneval --backend codex --model gpt-5.3-codex
```

## Architecture

- **CLI** (`src/cli.py`) — `qeval` entrypoint with subcommands: `run`, `generate`, `execute`, `agent-run`, `profile`, `list`
- **Generation** (`src/generation/`) — sample generation for standard eval (litellm, huggingface, vllm backends)
- **Evaluation** (`src/evaluation/`) — Q executor runs code via PyKX IPC against Python test assertions; metrics compute Pass@k
- **Agents** (`src/agents/`) — agent runner + backends (Claude Code, Codex) for iterative tool-use evaluation
- **Datasets** (`datasets/`) — Q-HumanEval problems in JSONL format
- **Outputs** (`outputs/`) — result files per model/dataset combination

## File Layout

```
├── datasets/          # Q-HumanEval JSONL problems
├── docs/              # Leaderboard, style guide, submission guides
├── model_cards/       # Per-model configuration
├── outputs/           # Evaluation results (per dataset/model)
├── scripts/           # Utility scripts
├── src/
│   ├── agents/        # Agent runner, backends (claude-code, codex)
│   ├── evaluation/    # Q executor, metrics (Pass@k)
│   ├── generation/    # Sample generation for standard eval
│   └── cli.py         # qeval CLI entrypoint
└── tests/             # Test suite
```

## Writing Q Code

If you are writing or modifying Q solutions (e.g. for agent evaluation workspaces), read `docs/q-style-guide.md` first. It covers idiomatic Q patterns and critical pitfalls — especially right-to-left evaluation, `%` being division (not modulo), reserved word conflicts, and atom/vector type mismatches.
