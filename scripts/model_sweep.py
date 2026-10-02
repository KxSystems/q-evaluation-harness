#!/usr/bin/env python3
"""Clean-room (--no-skills) q-humaneval sweep over several models of one backend.

Generalisation of scripts/opus5_ab.py (branch harness-abort-detection-tooling):
same abort detection, pidfile lock and cap-aware resume, but one ledger per
model and models run strictly in order.

Survives the 5-hour token cap across sessions. State of record is
outputs/sweep/<model>/state.json — re-running this script picks up exactly
where it left off, so a fresh Claude session (or a human) can resume with no
context:

    python3 scripts/model_sweep.py
    SWEEP_BACKEND=codex SWEEP_MODELS=gpt-5.5,gpt-6-sol python3 scripts/model_sweep.py

Also runs scripts/reap_runaway_q.py for its lifetime (runaway agent self-tests
starve and kill other agents — see that script's docstring).
"""
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-sonnet-5"]
if os.environ.get("SWEEP_MODELS"):
    MODELS = os.environ["SWEEP_MODELS"].split(",")
BACKEND = os.environ.get("SWEEP_BACKEND", "claude-code")
ARM_ARGS = ["--no-skills"]
BASE = os.environ.get("SWEEP_BASE", "outputs/sweep")
ALL_IDS = [json.loads(l)["task_id"] for l in open("datasets/q_humaneval.jsonl")]
if os.environ.get("SWEEP_IDS"):  # smoke-test subset
    ALL_IDS = [int(x) for x in os.environ["SWEEP_IDS"].split(",")]

# `claude --model <older id>` can be silently remapped to the latest model of
# that family (seen with opus-4-1 -> Opus 4.8). claude-sonnet-5 is no longer the
# latest Sonnet, so this is load-bearing for the Sonnet 5 row.
CHILD_ENV = {**os.environ, "CLAUDE_CODE_DISABLE_LEGACY_MODEL_REMAP": "1"}


def state_path(model):
    return os.path.join(BASE, model, "state.json")


def load_state(model):
    p = state_path(model)
    if os.path.exists(p):
        return json.load(open(p))
    return {"model": model, "dataset": "q-humaneval", "arm": "noskill",
            "total": len(ALL_IDS), "pending": list(ALL_IDS), "passed": [],
            "failed": [], "runs": [], "cost": 0.0, "next_eligible_epoch": 0,
            "cycles": 0}


def save_state(st):
    p = state_path(st["model"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(st, open(tmp, "w"), indent=2)
    os.replace(tmp, p)


CODEX_CAP_MARKERS = ("usage limit", "usage_limit", "rate limit", "rate_limit", "429")
CODEX_CAP_FALLBACK_S = 3600


def scan_aborted_codex(run_dir):
    """Codex flavour of scan_aborted.

    A fair attempt ends in a turn.completed event. An error / turn.failed event
    mentioning a usage or rate limit means the account is capped; Codex does not
    reliably report when the window reopens, so pause CODEX_CAP_FALLBACK_S
    unless the event carries resets_in_seconds.
    """
    capped, reset = set(), 0
    for ev in glob.glob(os.path.join(run_dir, "workspaces", "task_*", "events.jsonl")):
        tid = int(ev.split("task_")[-1].split("/")[0])
        completed = False
        for line in open(ev, errors="ignore"):
            try:
                e = json.loads(line)
            except Exception:
                continue
            t = e.get("type")
            if t == "turn.completed":
                completed = True
            elif t in ("error", "turn.failed"):
                blob = json.dumps(e).lower()
                if any(m in blob for m in CODEX_CAP_MARKERS):
                    capped.add(tid)
                    secs = (e.get("error") or {}).get("resets_in_seconds") \
                        or e.get("resets_in_seconds") or CODEX_CAP_FALLBACK_S
                    reset = max(reset, int(time.time() + secs))
        if not completed:
            capped.add(tid)
    return capped, reset


def scan_aborted(run_dir):
    """Task ids that never got a fair attempt, + max resetsAt.

    Retryable rather than scoreable: five_hour 429 / rejected rate-limit event,
    error_during_execution or user-interrupt (agent CLI killed externally),
    empty stream, or a stream with no terminal result event.
    """
    capped, reset = set(), 0
    for ev in glob.glob(os.path.join(run_dir, "workspaces", "task_*", "events.jsonl")):
        tid = int(ev.split("task_")[-1].split("/")[0])
        if os.path.getsize(ev) == 0:
            capped.add(tid)
            continue
        if not any('"type":"result"' in l or '"type": "result"' in l
                   for l in open(ev, errors="ignore")):
            capped.add(tid)
            continue
        for line in open(ev, errors="ignore"):
            if '"error_during_execution"' in line or "Request interrupted by user" in line:
                capped.add(tid)
            if "rate_limit" not in line and "429" not in line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") == "result" and e.get("is_error") and e.get("api_error_status") == 429:
                capped.add(tid)
            rli = e.get("rate_limit_info") or {}
            if rli.get("rateLimitType") == "five_hour" and rli.get("status") == "rejected":
                capped.add(tid)
                if rli.get("resetsAt"):
                    reset = max(reset, int(rli["resetsAt"]))
    return capped, reset


def served_models(run_dir):
    """Every model id seen in init events and assistant messages of a run.

    Claude Code only; codex --json events carry no model id, and Codex errors
    on an unknown id instead of silently remapping it.
    """
    seen = set()
    for ev in glob.glob(os.path.join(run_dir, "workspaces", "task_*", "events.jsonl")):
        for line in open(ev, errors="ignore"):
            if '"model"' not in line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") == "system" and e.get("subtype") == "init" and e.get("model"):
                seen.add(e["model"])
            elif e.get("type") == "assistant":
                m = (e.get("message") or {}).get("model")
                if m and m != "<synthetic>":
                    seen.add(m)
    return seen


def is_stub(sol):
    c = sol.get("completion", "")
    return ("/ function body" in c) or (not c.strip())


def run_batch(model, ids):
    """Returns (passed, failed, still_pending, reset_epoch, run_dir, cost)."""
    out = os.path.join(BASE, model)
    before = set(glob.glob(os.path.join(out, "agent_*")))
    cmd = ["poetry", "run", "qeval", "agent-run", "q-humaneval",
           "--backend", BACKEND, "--model", model, *ARM_ARGS,
           "--problem-ids", *map(str, ids),
           "--timeout", "600", "--concurrency", "4",
           "--save-events", "--keep-workspaces", "-o", out]
    print(f"[{time.strftime('%H:%M:%S')}] {model}: launching {len(ids)} tasks", flush=True)
    # own process group: isolates the agent CLIs from stray group signals
    # (lost 13 tasks to simultaneous interrupts on 2026-07-25)
    subprocess.run(cmd, check=False, start_new_session=True, env=CHILD_ENV)
    new = set(glob.glob(os.path.join(out, "agent_*"))) - before
    if not new:
        print(f"{model}: no run dir produced — aborting cycle", flush=True)
        return set(), set(), set(ids), 0, "", 0.0
    run_dir = max(new, key=os.path.getmtime)

    wrong = served_models(run_dir) - {model} if BACKEND == "claude-code" else set()
    if wrong:
        sys.exit(f"!! {model}: run {run_dir} was served by {sorted(wrong)} — "
                 f"model remap/fallback. Stopping; nothing from this run is scored.")

    sols = {json.loads(l)["task_id"]: json.loads(l)
            for l in open(os.path.join(run_dir, "solutions.jsonl"))}
    res_blob = json.load(open(os.path.join(run_dir, "results.json")))
    res = {r["task_id"]: r["passed"] for r in res_blob["results"]}

    scan = scan_aborted_codex if BACKEND == "codex" else scan_aborted
    capped, reset = scan(run_dir)
    passed, failed, pending = set(), set(), set()
    for t in ids:
        s = sols.get(t, {})
        stubbed = is_stub(s) and s.get("agent_wall_time", 99) <= 5 \
            and (s.get("agent_num_turns") or 0) <= 1
        if t in capped or stubbed or t not in res:
            pending.add(t)
        elif res[t]:
            passed.add(t)
        else:
            failed.add(t)
    cost = (res_blob.get("agent_metrics") or {}).get("total_cost_usd") or 0.0
    print(f"[{time.strftime('%H:%M:%S')}] {model}: +{len(passed)} pass, "
          f"{len(failed)} fail, {len(pending)} retry  (run {os.path.basename(run_dir)})",
          flush=True)
    return passed, failed, pending, reset, run_dir, cost


def acquire_lock():
    """Two concurrent drivers race on the ledgers and silently corrupt them."""
    os.makedirs(BASE, exist_ok=True)
    lock = os.path.join(BASE, "driver.lock")
    if os.path.exists(lock):
        pid = open(lock).read().strip()
        try:
            os.kill(int(pid), 0)
            alive = True
        except (OSError, ValueError):
            alive = False
        if alive:
            sys.exit(f"another driver is running (pid {pid}); refusing to race. "
                     f"Kill it or remove {lock} if that pid is stale.")
        print(f"clearing stale lock from dead pid {pid}", flush=True)
    open(lock, "w").write(str(os.getpid()))
    return lock


def wait_for_window(st):
    wait = st.get("next_eligible_epoch", 0) - time.time()
    if wait <= 0:
        return
    print(f"[{time.strftime('%H:%M:%S')}] token window closed — sleeping {wait/60:.0f} min "
          f"(resume {time.strftime('%H:%M', time.localtime(st['next_eligible_epoch']))})",
          flush=True)
    while time.time() < st["next_eligible_epoch"]:
        time.sleep(min(300, st["next_eligible_epoch"] - time.time() + 1))


def run_model(model, max_cycles):
    st = load_state(model)
    cycles = 0
    while st["pending"] and cycles < max_cycles:
        wait_for_window(st)
        cycles += 1
        st["cycles"] += 1
        passed, failed, still, reset, run_dir, cost = run_batch(model, st["pending"])
        st["passed"] = sorted(set(st["passed"]) | passed)
        st["failed"] = sorted(set(st["failed"]) | failed)
        st["pending"] = sorted(still)
        if run_dir:
            st["runs"].append(run_dir)
        st["cost"] = round(st["cost"] + cost, 2)
        if st["pending"]:
            if reset:                      # genuine 5-hour cap
                st["next_eligible_epoch"] = reset + 60
            elif passed or failed:         # transient abort; retry now
                st["next_eligible_epoch"] = time.time() + 30
            else:                          # nothing progressed, no resetsAt
                st["next_eligible_epoch"] = time.time() + 3600
        save_state(st)
        print(f"  {model}: {len(st['passed'])} passed / {len(st['failed'])} failed "
              f"/ {len(st['pending'])} pending  (${st['cost']})", flush=True)
    st["done"] = not st["pending"]
    save_state(st)
    return st


def main():
    lock = acquire_lock()
    reaper = subprocess.Popen([sys.executable, "scripts/reap_runaway_q.py", "120"])
    max_cycles = int(os.environ.get("SWEEP_MAX_CYCLES", "24"))
    try:
        for model in MODELS:
            st = run_model(model, max_cycles)
            if not st["done"]:
                print(f"!! {model} still has {len(st['pending'])} pending — "
                      f"stopping before the next model", flush=True)
                break
            print(f"== {model} COMPLETE: {len(st['passed'])}/{len(ALL_IDS)} "
                  f"(${st['cost']}) ==", flush=True)
    finally:
        reaper.kill()
        try:
            os.remove(lock)
        except OSError:
            pass

    print("\n=== clean-room sweep ===")
    all_done = True
    for model in MODELS:
        st = load_state(model)
        all_done &= bool(st.get("done"))
        print(f"{model:20s} {len(st['passed'])}/{len(ALL_IDS)} = "
              f"{len(st['passed']) / len(ALL_IDS) * 100:.1f}%  "
              f"(pending {len(st['pending'])}, ${st['cost']})")
    return 0 if all_done else 1


if __name__ == "__main__":
    sys.exit(main())
