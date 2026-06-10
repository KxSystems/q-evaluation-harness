#!/bin/bash
# launchd wrapper for the overnight rotating benchmark. launchd starts jobs with
# a minimal PATH, so we set it explicitly to find poetry, claude, codex and q.
set -euo pipefail

export HOME="/Users/scotwork"
export PATH="$HOME/.local/bin:$HOME/.homebrew/bin:$HOME/.kx/bin:/usr/bin:/bin:/usr/sbin:/sbin"

REPO="/Users/scotwork/Documents/coding/github-repos/q-evaluation-harness"
cd "$REPO"

LOGDIR="$REPO/outputs/overnight"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d_%H%M%S)"

# Run the driver; tee a per-night log.
poetry run python scripts/overnight_eval.py >>"$LOGDIR/run_$TS.log" 2>&1
