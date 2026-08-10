# Opus 5 run notes — skilled vs clean room (2026-07-25)

Run notes behind the two Opus 5 rows on the [agent leaderboard](leaderboard.md),
including the harness issue that made manual re-runs necessary.

Results files: `outputs/results_agent_claude-opus-5_skill.json` and
`outputs/results_agent_claude-opus-5_noskill.json`.

Both arms: 164/164 tasks fairly attempted, verified by
`python3 scripts/reopen_aborted.py` (reports "clean" for both).

| arm | pass | rate | cost | $/task |
|---|---|---|---|---|
| `noskill` (clean room) | **160/164** | **97.6%** | $39.70 | $0.24 |
| `skilled` (q-kdb + kx plugins) | 157/164 | 95.7% | $61.19 | $0.37 |

**The clean room won, and the skill never once rescued a task the clean room
lost**: discordant pairs are 0 skilled-only vs 3 noskill-only (tasks 32, 92,
112). Exact McNemar two-sided p = 0.250 — not significant at n=164, so treat
the 3-task gap as "no demonstrated benefit" rather than "the skill hurts".

**Cost is the unambiguous result: the skilled arm cost 1.54x more** ($61.19 vs
$39.70) for no accuracy gain, and ran slower (63.9s vs ~12s per task median
pace, 9.3 turns/task). Reading the skill burns turns and tokens.

Failed in **both** arms — the genuinely hard four: 12, 81, 116, 162.
Skilled-only failures: 32, 92, 112.

This replicates the direction of the Fable 5 result (noskill 158/164 beat
skilled 153/164), now on a run where every task is verified attempted.

## Caveat on what "skilled" means

The skilled arm loads the staged q-kdb skill **plus** the globally installed kx
plugins (q-knowledge, kdbx-knowledge, pykx-knowledge) — its init event lists all
of them. That matches how earlier skilled leaderboard rows were produced, but
the honest label is "q-kdb + installed kx plugins", not "q-kdb alone".
The clean room is verified genuine: init event shows `skills: []`, `plugins: []`.

Skill provenance: `~/.q-eval/skills/q-kdb` byte-identical (`diff -r`) to
`KxSystems/kx-skills` HEAD `f8deb26`, `plugins/q-knowledge/skills/q`.

## Getting to a trustworthy number took three harness fixes

The first attempt produced 152/164 for skilled, which was wrong. Three separate
defects, all of which silently deflate scores:

1. **Two drivers raced** on one ledger (7 run dirs where ~4 belonged), so the
   tally was last-writer-wins garbage. Fixed with a pidfile lock.
2. **Three abort variants scored as model failures** — rate-cap 429,
   `error_during_execution` kills, and streams that simply stop with no terminal
   result event. The third cost 10 tasks in the contiguous band 24-36 inside one
   44-second window. All three are now detected and retried, not scored.
3. **Root cause of the aborts**: the agent self-tests with `q solution.q` at
   codegen time and sometimes picks a pathological input — `factorize 1000000007`
   (a 10-digit prime) spins q at 100% CPU indefinitely, starving other in-flight
   agents until they die silently. SIGTERM does not kill it; SIGKILL does.
   `scripts/reap_runaway_q.py` reaps them (8 kills during these runs).

The real fix for #3 belongs in the harness, not this driver: wrap agent-issued
Bash in `timeout -k`, or reap orphaned `q solution.q` between tasks.

## Why Opus 5 tripped this and earlier models did not

`scripts/profile_selftests.py` — q self-tests per task, and how many carry a
>=6-digit numeric literal:

| model | q_tests/task | heavy probes |
|---|---|---|
| sonnet-4-7, haiku-4-7 | 0.00 | 0 |
| sonnet-4-6 | 0.88 | 0 |
| opus-4-7 | 0.93 / 1.21 | 0 / 2 |
| fable-5 | 1.26 / 1.90 | 5 / 3 |
| **opus-5** | **1.90** | **27** |

Opus 5 does not self-test *more often* (1.90 ties fable-5's June run). It picks
much more extreme inputs: 27 heavy probes against a prior ceiling of 5. Models
that never self-test (sonnet-4-7, haiku-4-7 at 0.00) never hit the failure mode.
`1000000007` and `600851475143` appear in fable-5 and opus-4-7 runs too, just
far less often.
