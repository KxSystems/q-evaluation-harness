#!/usr/bin/env python3
"""Overnight rotating benchmark driver.

Each invocation runs ONE model's full q-humaneval agent eval, rotating
Fable 5 -> Opus 4.8 -> Codex 5.5 across nights. Survives the 5-hour token cap:
if a run gets rate-limited, the un-run tasks are deferred and re-run after the
window resets (a sleeping process costs no tokens — only `claude`/`codex` calls
do). Designed to be launched by launchd; see com.qeval.overnight.plist.

State:  outputs/overnight/rotation.json   (which model is next)
Log:    outputs/overnight/leaderboard.log (one line per completed model run)
"""
import json, os, sys, time, subprocess, glob, datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
OUT = os.path.join(REPO, "outputs", "overnight")
os.makedirs(OUT, exist_ok=True)
SKILL = os.path.expanduser("~/.q-eval/skills/q-kdb")
ROTATION = os.path.join(OUT, "rotation.json")
LOG = os.path.join(OUT, "leaderboard.log")
DETECT = os.path.join(REPO, "outputs", "improvement_loop", "detect_cap.py")

MODELS = [
    {"name": "claude-fable-5", "backend": "claude-code"},
    {"name": "claude-opus-4-8", "backend": "claude-code"},
    {"name": "gpt-5.5", "backend": "codex", "extra": ["--reasoning-effort", "high"]},
]
MAX_RESUME_CYCLES = 4   # safety: don't loop forever across cap windows
ALL_IDS = list(range(164))


def pick_model():
    idx = 0
    if os.path.exists(ROTATION):
        idx = json.load(open(ROTATION)).get("next_index", 0) % len(MODELS)
    return idx, MODELS[idx]


def advance(idx):
    json.dump({"next_index": (idx + 1) % len(MODELS)}, open(ROTATION, "w"))


def run_round(model, ids):
    """Run agent-run on the given ids; return the new output dir."""
    before = set(glob.glob("outputs/agent_*"))
    cmd = [
        "poetry", "run", "qeval", "agent-run", "q-humaneval",
        "--backend", model["backend"], "--model", model["name"],
        "--skill-dirs", SKILL, "--problem-ids", *map(str, ids),
        "--timeout", "600", "--concurrency", "4", "--save-events",
    ] + model.get("extra", [])
    subprocess.run(cmd, check=False)
    after = set(glob.glob("outputs/agent_*")) - before
    return max(after, key=os.path.getmtime) if after else None


def detect_cap(run_dir):
    r = subprocess.run(["poetry", "run", "python", DETECT, run_dir],
                       capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"capped_ids": [], "resets_at_epoch": 0}


def passed_ids(run_dir):
    rj = os.path.join(run_dir, "results.json")
    if not os.path.exists(rj):
        return set()
    return {r["task_id"] for r in json.load(open(rj))["results"] if r["passed"]}


def main():
    idx, model = pick_model()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[overnight {stamp}] model={model['name']} backend={model['backend']}")

    remaining = list(ALL_IDS)
    all_passed = set()
    for cycle in range(MAX_RESUME_CYCLES):
        if not remaining:
            break
        run_dir = run_round(model, remaining)
        if not run_dir:
            print("no run dir produced; aborting"); break
        cap = detect_cap(run_dir)
        passed = passed_ids(run_dir)
        all_passed |= passed
        capped = set(cap["capped_ids"]) - passed     # capped & not salvaged
        # tasks attempted this round and not capped are final (pass or fail)
        attempted_final = set(remaining) - capped
        remaining = sorted(capped)
        print(f"  cycle {cycle}: +{len(passed)} passed, {len(capped)} capped/deferred")
        if not remaining:
            break
        reset = cap.get("resets_at_epoch", 0)
        wait = max(60, reset + 60 - int(time.time())) if reset else 300
        wait = min(wait, 6 * 3600)   # cap the sleep at 6h
        print(f"  hit 5h cap; sleeping {wait}s until window resets "
              f"(no token cost while sleeping)")
        time.sleep(wait)

    line = (f"{stamp}\t{model['name']}\t{model['backend']}\t"
            f"{len(all_passed)}/164\tpass_rate={len(all_passed)/164:.4f}\n")
    with open(LOG, "a") as f:
        f.write(line)
    print("DONE:", line.strip())
    advance(idx)


if __name__ == "__main__":
    main()
