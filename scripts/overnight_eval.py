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
    # No-skill baseline: same Fable 5 model, but zero q/kdb help. The harness's
    # --no-skills flag installs no skill (so the "load q-kdb skill" step is
    # dropped) AND blocks global skills/plugins at the claude CLI, so nothing
    # leaks in. The stream-json init event proves it (skills:[] plugins:[]).
    # Measures raw model ability vs the skill-assisted Fable run above.
    {
        "name": "claude-fable-5",
        "backend": "claude-code",
        "label": "claude-fable-5-noskill",
        "no_skill": True,
    },
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
        "--problem-ids", *map(str, ids),
        "--timeout", "600", "--concurrency", "4", "--save-events",
        # keep workspaces so detect_cap.py can read events.jsonl (resetsAt);
        # without this the runner deletes them and cap detection silently fails.
        "--keep-workspaces",
    ]
    # Skill-assisted models install the bundled q-kdb skill; the no-skill
    # baseline passes --no-skills, which omits it and blocks global skills/
    # plugins at the claude CLI.
    if model.get("no_skill"):
        cmd += ["--no-skills"]
    else:
        cmd += ["--skill-dirs", SKILL]
    cmd += model.get("extra", [])
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


def stub_casualties(run_dir):
    """Events-independent cap detection: a task whose agent returned almost
    instantly (<=3s, <=1 turn) leaving the stub untouched is a rate-limit
    casualty, not a real failure. Survives even if events.jsonl is gone."""
    sj = os.path.join(run_dir, "solutions.jsonl")
    out = set()
    if not os.path.exists(sj):
        return out
    for line in open(sj):
        r = json.loads(line)
        c = r.get("completion", "")
        is_stub = ("/ function body" in c) or (not c.strip())
        instant = (r.get("agent_wall_time", 99) <= 3) and (r.get("agent_num_turns", 9) <= 1)
        if is_stub and instant:
            out.add(r["task_id"])
    return out


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sleep_through_cap(wait, n_capped):
    """Suspend until the token window resets, logging loudly the whole time.

    The 5h-cap 429s are otherwise silent (the agent subprocess just returns an
    instant stub), and a multi-hour bare time.sleep() under launchd looks like
    a hung/dead process. We print a clear banner with the absolute resume time,
    then a heartbeat every few minutes so it's obvious the run is alive and
    merely waiting — at no token cost.
    """
    resume_at = datetime.datetime.now() + datetime.timedelta(seconds=wait)
    resume_str = resume_at.strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 64, flush=True)
    print(f"  [{_now()}] 5-HOUR TOKEN CAP HIT — {n_capped} task(s) deferred.", flush=True)
    print(f"  SUSPENDING for {wait // 60}m{wait % 60:02d}s; will resume at {resume_str}.", flush=True)
    print(f"  The process is ALIVE and sleeping — no tokens are spent while waiting.", flush=True)
    print("=" * 64, flush=True)
    HEARTBEAT = 300  # seconds between "still alive" pings
    slept = 0
    while slept < wait:
        chunk = min(HEARTBEAT, wait - slept)
        time.sleep(chunk)
        slept += chunk
        remaining = wait - slept
        if remaining > 0:
            print(f"  [{_now()}] still sleeping through cap — "
                  f"{remaining // 60}m left, resume at {resume_str}", flush=True)
    print(f"  [{_now()}] window reset reached — resuming run.", flush=True)


def main():
    idx, model = pick_model()
    label = model.get("label", model["name"])
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[overnight {stamp}] model={label} backend={model['backend']}", flush=True)

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
        # Union event-based detection with the events-independent stub signal,
        # then drop any that actually passed.
        capped = (set(cap["capped_ids"]) | stub_casualties(run_dir)) - passed
        remaining = sorted(capped)
        print(f"  cycle {cycle}: +{len(passed)} passed, {len(capped)} capped/deferred",
              flush=True)
        if not remaining:
            break
        reset = cap.get("resets_at_epoch", 0)
        # If events gave no resetsAt (e.g. stub-only detection), wait a full
        # window rather than retrying immediately into the same cap.
        wait = max(60, reset + 60 - int(time.time())) if reset else 3600
        wait = min(wait, 6 * 3600)   # cap the sleep at 6h
        sleep_through_cap(wait, len(remaining))

    line = (f"{stamp}\t{label}\t{model['backend']}\t"
            f"{len(all_passed)}/164\tpass_rate={len(all_passed)/164:.4f}\n")
    with open(LOG, "a") as f:
        f.write(line)
    print("DONE:", line.strip(), flush=True)
    advance(idx)


if __name__ == "__main__":
    main()
