#!/usr/bin/env python3
"""Collapse each sweep model's run fragments into one canonical results file.

Sweep counterpart of merge_ab_results.py: reads the per-model ledgers written
by scripts/model_sweep.py (outputs/sweep/<model>/state.json) and writes
outputs/results_agent_<model>_<label>.json in the committed schema. Keeps each
task's LATEST attempt; cost is the sum of those final attempts only.

Cost is recomputed from the usage in each final attempt's event stream at list
price (src/agents/pricing.py), not taken from the CLI, which mis-prices some
models; the CLI's own figure is kept as cli_reported_cost_usd. Tasks without an
event stream (run without --save-events) fall back to the CLI figure.

Cross-checks merged passes against the ledger and refuses to write on mismatch.

Usage: python3 scripts/merge_sweep_results.py
       SWEEP_BASE=outputs/sweep_haiku_skilled SWEEP_LABEL=skill \
           python3 scripts/merge_sweep_results.py
       (SWEEP_EFFORT=medium records an explicit --effort level in the file)
"""
import glob
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
from src.agents.pricing import CLAUDE_PRICES, claude_cost_usd  # noqa: E402

BASE = os.environ.get("SWEEP_BASE", "outputs/sweep")
LABEL = os.environ.get("SWEEP_LABEL", "noskill")
EFFORT = os.environ.get("SWEEP_EFFORT") or None


def stream_usage(events_path):
    """(task usage totals from the last result event, largest request prompt).

    The prompt side of an assistant message's usage is exact even though its
    output count is a start-of-message snapshot, so it gives the longest
    prompt any single request sent, which decides length-tiered pricing.
    """
    usage, max_prompt = None, 0
    for line in open(events_path, errors="ignore"):
        if '"type":"result"' in line or '"type": "result"' in line:
            try:
                usage = json.loads(line).get("usage") or usage
            except ValueError:
                pass
        elif '"type":"assistant"' in line or '"type": "assistant"' in line:
            try:
                u = (json.loads(line).get("message") or {}).get("usage") or {}
            except ValueError:
                continue
            max_prompt = max(max_prompt, (u.get("input_tokens") or 0)
                             + (u.get("cache_read_input_tokens") or 0)
                             + (u.get("cache_creation_input_tokens") or 0))
    return usage, max_prompt


def build(model):
    ledger = json.load(open(os.path.join(BASE, model, "state.json")))
    sols, res, final_dir = {}, {}, {}
    for d in sorted(glob.glob(os.path.join(BASE, model, "agent_*/")), key=os.path.getmtime):
        for line in open(os.path.join(d, "solutions.jsonl")):
            s = json.loads(line)
            sols[s["task_id"]] = s
            final_dir[s["task_id"]] = d
        for r in json.load(open(os.path.join(d, "results.json")))["results"]:
            res[r["task_id"]] = r

    tasks = sorted(res)
    results = [{"task_id": t, "sample_index": 0, "passed": bool(res[t]["passed"]),
                "info": res[t].get("info", "passed" if res[t]["passed"] else "failed"),
                "errored": bool(res[t].get("errored", False))} for t in tasks]
    got = {r["task_id"] for r in results if r["passed"]}
    if got != set(ledger["passed"]) or ledger["pending"]:
        sys.exit(f"REFUSING to write {model}: merged passes {len(got)} vs ledger "
                 f"{len(ledger['passed'])}, pending {len(ledger['pending'])}")

    wall = [s.get("agent_wall_time") or 0 for s in sols.values()]
    turns = [s["agent_num_turns"] for s in sols.values() if s.get("agent_num_turns")]
    cli_cost = sum(s.get("agent_cost_usd") or 0 for s in sols.values())
    cost, repriced = 0.0, 0
    for t, s in sols.items():
        ev = os.path.join(final_dir[t], "workspaces", f"task_{t}", "events.jsonl")
        usage, max_prompt = stream_usage(ev) if os.path.exists(ev) else (None, 0)
        listed = claude_cost_usd(model, usage, max_prompt) if usage else None
        if listed is None and usage and model in CLAUDE_PRICES:
            sys.exit(f"REFUSING to write {model}: task {t} has a request over the "
                     f"length-tier threshold ({max_prompt} tokens); needs per-request pricing")
        if listed is None:
            cost += s.get("agent_cost_usd") or 0
        else:
            cost += listed
            repriced += 1
    if repriced not in (0, len(sols)):
        sys.exit(f"REFUSING to write {model}: only {repriced}/{len(sols)} tasks "
                 f"could be priced from events; cost would mix methods")
    passed_wall = [sols[t].get("agent_wall_time") or 0 for t in tasks if res[t]["passed"]]
    failed_wall = [sols[t].get("agent_wall_time") or 0 for t in tasks if not res[t]["passed"]]

    return {
        "total_solutions": len(results),
        "passed_solutions": len(got),
        "errored_solutions": sum(r["errored"] for r in results),
        "scored_solutions": len(results),
        "pass_rate": len(got) / len(results),
        "total_problems": len(results),
        "pass_at_1": len(got) / len(results),
        "per_problem_results": [{"task_id": t, "num_samples": 1,
                                 "num_correct": int(res[t]["passed"])} for t in tasks],
        "evaluation_type": "agent",
        "agent_backend": "claude-code",
        "agent_model": model,
        "agent_skill": LABEL.startswith("skill"),
        "agent_effort": EFFORT,  # explicit --effort; None = Claude Code default
        "agent_max_turns": 20,
        "dataset": ledger.get("dataset", "q-humaneval"),
        "agent_metrics": {
            "total_wall_time_seconds": sum(wall),
            "mean_wall_time_seconds": statistics.mean(wall),
            "median_wall_time_seconds": statistics.median(wall),
            "mean_wall_time_passed": statistics.mean(passed_wall) if passed_wall else None,
            "mean_wall_time_failed": statistics.mean(failed_wall) if failed_wall else None,
            "total_cost_usd": round(cost, 4),
            "mean_cost_usd": cost / len(results),
            "cost_basis": "list price from token usage" if repriced else "CLI reported",
            "cli_reported_cost_usd": round(cli_cost, 4),
            "mean_turns": statistics.mean(turns) if turns else None,
            "median_turns": statistics.median(turns) if turns else None,
            "no_solution_count": sum(
                1 for s in sols.values()
                if not (s.get("completion") or "").strip()
                or "/ function body" in (s.get("completion") or "")),
            "errored_count": sum(r["errored"] for r in results),
        },
        "results": results,
    }


def main():
    for state in sorted(glob.glob(os.path.join(BASE, "*", "state.json"))):
        model = os.path.basename(os.path.dirname(state))
        blob = build(model)
        out = f"outputs/results_agent_{model}_{LABEL}.json"
        json.dump(blob, open(out, "w"), indent=2)
        m = blob["agent_metrics"]
        print(f"{out}: {blob['passed_solutions']}/{blob['total_solutions']} "
              f"${m['total_cost_usd']:.2f} (${m['mean_cost_usd']:.4f}/task; "
              f"CLI said ${m['cli_reported_cost_usd']:.2f}) "
              f"turns {m['mean_turns']:.1f}")


if __name__ == "__main__":
    main()
