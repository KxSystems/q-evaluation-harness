#!/usr/bin/env python3
"""Collapse each sweep model's run fragments into one canonical results file.

Sweep counterpart of merge_ab_results.py: reads the per-model ledgers written
by scripts/model_sweep.py (outputs/sweep/<model>/state.json) and writes
outputs/results_agent_<model>_noskill.json in the committed schema. Keeps each
task's LATEST attempt; cost is the sum of those final attempts only.

Cross-checks merged passes against the ledger and refuses to write on mismatch.

Usage: python3 scripts/merge_sweep_results.py
"""
import glob
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
BASE = os.environ.get("SWEEP_BASE", "outputs/sweep")


def build(model):
    ledger = json.load(open(os.path.join(BASE, model, "state.json")))
    sols, res = {}, {}
    for d in sorted(glob.glob(os.path.join(BASE, model, "agent_*/")), key=os.path.getmtime):
        for line in open(os.path.join(d, "solutions.jsonl")):
            s = json.loads(line)
            sols[s["task_id"]] = s
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
    cost = sum(s.get("agent_cost_usd") or 0 for s in sols.values())
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
        out = f"outputs/results_agent_{model}_noskill.json"
        json.dump(blob, open(out, "w"), indent=2)
        m = blob["agent_metrics"]
        print(f"{out}: {blob['passed_solutions']}/{blob['total_solutions']} "
              f"${m['total_cost_usd']:.2f} (${m['mean_cost_usd']:.3f}/task) "
              f"turns {m['mean_turns']:.1f}")


if __name__ == "__main__":
    main()
