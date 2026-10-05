#!/usr/bin/env python3
"""Kill agent self-test q processes that have gone runaway.

Root cause of the 2026-07-25 silent agent terminations: the agent probes its own
solution with `q solution.q`, occasionally on a pathological input (factorize of
the 10-digit prime 1000000007), and that q process pins a core indefinitely.
Enough of them and other in-flight agent CLIs get starved and die with no
terminal result event — silently scored as model failures.

A legitimate self-test finishes in seconds. Anything past MAX_AGE is stuck, so
SIGKILL it (SIGTERM is not enough — observed ignored by a spinning q).

Usage: python3 scripts/reap_runaway_q.py [max_age_seconds]
Runs until interrupted; prints each kill.
"""
import os
import re
import signal
import subprocess
import sys
import time

MAX_AGE = int(sys.argv[1]) if len(sys.argv) > 1 else 120
POLL = 10


def etime_to_seconds(et):
    """ps etime: [[dd-]hh:]mm:ss"""
    days = 0
    if "-" in et:
        d, et = et.split("-", 1)
        days = int(d)
    parts = [int(p) for p in et.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def scan():
    out = subprocess.run(["ps", "-eo", "pid,etime,pcpu,command"],
                         capture_output=True, text=True).stdout
    victims = []
    for line in out.splitlines()[1:]:
        if "q solution.q" not in line or "grep" in line:
            continue
        m = re.match(r"\s*(\d+)\s+(\S+)\s+([\d.]+)\s+(.*)", line)
        if not m:
            continue
        pid, et, cpu, cmd = int(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
        # only the bare q process, not the wrapping shell that spawned it
        if not cmd.startswith("q solution.q"):
            continue
        age = etime_to_seconds(et)
        if age >= MAX_AGE:
            victims.append((pid, age, cpu))
    return victims


def main():
    print(f"reaper: killing `q solution.q` older than {MAX_AGE}s, polling every {POLL}s",
          flush=True)
    killed = 0
    while True:
        for pid, age, cpu in scan():
            try:
                os.kill(pid, signal.SIGKILL)   # SIGTERM is ignored by a spinning q
                killed += 1
                print(f"[{time.strftime('%H:%M:%S')}] reaped pid {pid} "
                      f"(age {age}s, {cpu}% cpu) — total {killed}", flush=True)
            except OSError as e:
                print(f"  could not kill {pid}: {e}", flush=True)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
