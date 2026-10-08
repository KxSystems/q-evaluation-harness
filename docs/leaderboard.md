# 🏆 Q Programming Language Leaderboard

Welcome to the official leaderboard for Q programming language model evaluation! This leaderboard tracks the performance of various language models on Q/kdb+ programming tasks.

> ⚠️ **Grader change (June 2026):** A grader fix ([PR #7](https://github.com/KxSystems/q-evaluation-harness/pull/7)) removed a class of false-negative test failures, lifting every benchmarked model by **+3 to +13 points** with no regressions. Results scored with the **new grader** are **not comparable** to the historical (pre-#7) numbers. The sections below are grouped by grader version — do not compare across them.

---

## 🤖 Agent Leaderboard (new grader)

Agent mode gives the model a Q interpreter and lets it iteratively write, run, and fix its solution (one task attempt, multiple turns). Run with `--timeout 300` (Opus 5 and later: `--timeout 600`); Codex at `--reasoning-effort high`. Ranked by Pass@1. **Cost/task** is total API-equivalent spend ÷ 164 tasks attempted (four decimals for Haiku 5.5, whose cost per task is under a cent), the same per-task measure Databricks uses in its [coding-agent benchmark](https://www.databricks.com/blog/benchmarking-coding-agents-databricks-multi-million-line-codebase). See [Cost per task on q-humaneval](blog/2026-10-01-cost-per-task.md) for the write-up and chart.

| Rank | Model | Backend | Skill | Pass@1 | Cost/task | Cost/run | Mean turns |
|------|-------|---------|:-----:|--------|-----------|----------|-----------|
| 🥇 | **Claude Fable 5.1** § | Claude Code | ✗ | **98.2%** (161/164) | $0.252 | $41.35 | 3.8 |
| 🥈 | **Claude Opus 5.5** § | Claude Code | ✗ | **97.6%** (160/164) | $0.085 | $13.97 | 2.7 |
| 🥈 | **Claude Opus 5** ‡ | Claude Code | ✗ | **97.6%** (160/164) | $0.242 | $39.70 | 7.6 |
| 4 | **Claude Haiku 5.5, medium effort** ‖ | Claude Code | ✓ | 97.0% (159/164) | $0.0081 | $1.33 | 7.5 |
| 5 | **Claude Sonnet 5.5** § | Claude Code | ✗ | 96.3% (158/164) | $0.041 | $6.78 | 3.5 |
| 5 | **Claude Haiku 5.5, high effort** ‖ | Claude Code | ✓ | 96.3% (158/164) | $0.0091 | $1.49 | 7.6 |
| 7 | **Claude Opus 5** ‡ | Claude Code | ✓ | 95.7% (157/164) | $0.371 | $60.82 | 9.3 |
| 7 | **Claude Haiku 5.5** ‖ | Claude Code | ✓ | 95.7% (157/164) | $0.0092 | $1.51 | 7.9 |
| 7 | **Claude Sonnet 5.5** § | Claude Code | ✓ | 95.7% (157/164) | $0.109 | $17.91 | 4.2 |
| 9 | **GPT-6 Sol** ¶ | Codex | ✓ | 95.1% (156/164) | $0.088 | $14.45 | 14.7 |
| 9 | **Fable 5** \* | Claude Code | ✓ | 95.1% (156/164) | — | — | — |
| 11 | **GPT-5.5** | Codex | ✓ | 94.5% (155/164) | — † | — † | 18.8 |
| 12 | **Claude Haiku 5.5** ‖ | Claude Code | ✗ | 93.9% (154/164) | **$0.0076** | $1.25 | 7.7 |
| 13 | **GPT-6 Sol** ¶ | Codex | ✗ | 91.5% (150/164) | $0.080 | $13.05 | 14.5 |
| 13 | **Claude Sonnet 5** § | Claude Code | ✗ | 91.5% (150/164) | $0.121 | $19.87 | 7.4 |
| 15 | **GPT-5.5** ¶ | Codex | ✗ | 90.9% (149/164) | $0.225 | $36.98 | 20.9 |
| 16 | **GPT-5.6 Sol** ¶ | Codex | ✗ | 89.6% (147/164) | $0.150 | $24.59 | 10.7 |
| 17 | **Claude Opus 4.8** | Claude Code | ✓ | 88.4% (145/164) | $0.320 | $52.40 | 8.7 |
| 18 | **Claude Opus 4.8** | Claude Code | ✗ | 87.2% (143/164) | $0.152 | $24.86 | 5.8 |
| 18 | **Claude Opus 4.7** \* | Claude Code | ✓ | 87.2% (143/164) | — | — | — |
| 20 | **Claude Sonnet 4.6** \* | Claude Code | ✓ | 70.1% (115/164) | — | — | — |
| 21 | **Claude Haiku 4.5** \* | Claude Code | ✓ | 37.2% (61/164) | — | — | — |

**Skill column** (`--skill-dirs` q-kdb skill, ✓ = installed, ✗ = clean-room `--no-skills`): the skill still helps the GPT models. **GPT-6 Sol gains +3.7 pts** (156 vs 150) at only **1.11x** the cost per task ($0.088 vs $0.080); it rescued 8 tasks and broke 2 (exact McNemar p = 0.11, suggestive but not significant at n=164). Three of the rescued tasks (22, 88, 109) were failed by every GPT model in the clean room. **GPT-5.5 gains +3.7 pts** against its October clean-room run (155 vs 149; +6.1 against the April run's 145). The skill helps **Opus 4.8 by +1.2 pts** (at **2.1x** the cost per task), but **costs Opus 5 1.9 pts** at **1.53x** the cost per task ($0.371 vs $0.242). On Opus 5 the skill never rescued a task the clean room missed (0 skilled-only wins vs 3 clean-room-only); at n=164 that gap is not statistically significant (exact McNemar p = 0.25), so read it as *no demonstrated benefit* rather than active harm. **Sonnet 5.5 gains nothing from the skill:** 157 skilled vs 158 clean room (3 rescued, 4 broken, exact McNemar p = 1.0) at **2.6x** the cost per task ($0.109 vs $0.041). Output barely changed (about 1,000 tokens per task either way) and turns rose only from 3.5 to 4.2; the extra spend is cache writes, which went from about 6.6k to 21.4k tokens per task as the skill's content was written to the 1-hour cache at twice the input price. **Haiku 5.5** gains in all three skilled arms against its clean-room run: **+3 tasks** at default effort (157 vs 154, 5 rescued and 2 broken, exact McNemar p = 0.45), **+5 at medium** (159 vs 154, 7 rescued and 2 broken, p = 0.18) and **+4 at high** (158 vs 154, 7 and 3, p = 0.34), for 21%, 6% and 19% more cost per task. The sign is consistent but no arm is significant at n=164, and Sonnet 5.5 was run both ways, so that comparison is like-for-like (see below). Opus 5.5 has no skilled run. Explicit effort made no detectable difference between the three skilled arms (157 to 159; exact McNemar p = 0.62 to 1.0). Tasks 92 and 116 fail in all four Haiku 5.5 arms; 162 of 164 tasks are solved by at least one. Final results for all arms carry zero infrastructure errors and zero missing solutions (for Opus 5 this is after the re-runs described in ‡).

§ **Fall 2026 runs** (Opus 5.5, Sonnet 5.5 and Sonnet 5 on Sept 30; Fable 5.1 on Oct 1): clean room, `--timeout 600`, Claude Code 2.1.286, except the skilled Sonnet 5.5 row (Oct 8, Claude Code 2.1.287, otherwise the same setup; two driver batches, every task attempted). No infrastructure errors. Fable 5.1 hit the 5-hour usage cap mid-run; the 31 affected tasks were deferred, not scored, and run after the reset, and cost counts final attempts only. Every agent event stream was audited. One finding: because task workspaces sit inside this repository, Claude Code loaded the operator's project memory notes into every Claude run listed here, including the July Opus 5 rows. No agent read the dataset or tests, and only one task in any run opened a memory note (Fable 5.1, task 118); re-run with memory disabled, it still passed. The harness now disables auto-memory for all agent runs. Claude Code applies per-turn effort to the two 5.5 models (not to Sonnet 5), so these rows measure each model together with Claude Code's default effort handling. **Sonnet 5.5's cost was corrected** from $7.82 ($0.048 per task) to $6.78 ($0.041): Claude Code 2.1.286 billed its cache reads at $0.20 per million, twice the published $0.10, and recomputing from the run's token counts gives the lower figure. The Opus 5.5 and Sonnet 5 figures reproduce to the cent.

\* **Reference rows** — prior overnight runs re-graded with the new grader, included for breadth. Caveats: run on a different machine, mostly `--timeout 600` with 5-hour-cap resume (so wall-time/cost are unreliable and omitted), and three dataset prompts changed after these runs (≤2 pt effect). These pre-date the `--no-skills` flag, so they used the default skill workflow and are reported as skill ✓.

‡ **Opus 5 rows required manual re-runs.** A harness issue silently terminated some agent processes mid-task, and those tasks were initially scored as model failures; the affected tasks were re-run to completion and every task in both arms is verified attempted. Run at `--timeout 600`. Cost counts each task's final attempt only, so the skilled arm is $60.82 rather than the $61.19 ledger total that included the aborted attempts. Details in [OPUS-5-NOTES.md](OPUS-5-NOTES.md).

‖ **Haiku 5.5 runs, Oct 8 2026:** four arms (clean room, and the q-kdb skill at default, medium and high effort), Claude Code 2.1.287, `--timeout 600`, concurrency 4, every task attempted and every event stream captured. Rows without an effort level use Claude Code's default, which applies per-turn effort to the 5.5 models; "medium" and "high" pass an explicit `--effort`. The CLI does not recognize `claude-haiku-5-5` and reported about 32x the real cost, so cost is computed from token counts at published prices (`src/agents/pricing.py`): $0.10 input, $0.50 output and $0.01 cache read per million tokens for prompts up to 100k tokens (longer prompts are priced about 5x higher; one request in a superseded first attempt reached 101,927 tokens, and no final-attempt request exceeded the threshold). Cache writes are the 1-hour tier at $0.20. The driver re-ran 5 clean-room tasks (123, 134, 135, 137, 160) whose first attempt ended without a result event, one of them twice; the three skilled arms needed no re-runs. Each task's final attempt is scored and costed. The sweep's process reaper cleaned up leftover q processes during all four runs (14 to 40 per arm). No agent opened the dataset or hidden tests. The skill is the same q-kdb skill used for the other skilled rows.

¶ **OpenAI runs, Oct 2 2026:** Codex CLI 0.160.0, `--reasoning-effort high`, `--timeout 600`, ChatGPT Business subscription. Clean room for all three models, plus a skilled GPT-6 Sol arm. Every task attempted, with no timeouts, missing solutions or rate-limit deferrals. Cost is computed from Codex's token counts at OpenAI's list prices as of Oct 2 (GPT-5.6 Sol at its post-July-30 price), including cached-input discounts. The October GPT-5.5 clean-room run replaces the April one (145/164, no cost captured); the 4-task difference is run-to-run variance, not a model change. Every Codex command was audited: no agent read the dataset or hidden tests. Two GPT-5.6 Sol agents looked outside their own workspace, because workspaces sit inside this repository. Task 121 searched the repo for its problem text and found nothing, and task 62 read a neighbouring task's problem and solution (task 32, a different problem). Task 62 is solved by every model on the board, so neither changes a score.

**Mean turns are not comparable across backends.** Codex counts every action (each shell command, file edit and message), while Claude Code counts model round trips, which can bundle several tool calls. Counted the same way, as tool actions per task, the gap is even wider: Opus 5.5 takes 1.7 and Sonnet 5.5 2.5, against 12.8 for GPT-5.5, 7.4 for GPT-5.6 Sol and 11.3 for GPT-6 Sol. Two thirds of GPT-6 Sol's commands are q test runs. The extra actions cost little because each round trip mostly re-reads cached context.

† The April GPT-5.5 skilled run pre-dates Codex cost capture: tokens were recorded but no dollar cost. Claude Code reports the API-equivalent cost itself (subscription-covered, so ~$0 cash in practice).

> **Cross-check:** an independent prior agentic GPT-5.5 run re-grades to **93.3%** (153/164), matching the skilled 94.5% within run-to-run variance.

> **What's next:** a skilled GPT-5.6 Sol run, to complete the skill A/B across the OpenAI line. GPT-5.5 and GPT-6 Sol both gain about 4 points from the skill; the open question is whether GPT-5.6 Sol, whose prompting guidance favors leaner context, follows the same pattern.

> Agent results are **not** comparable to the one-shot board below — agents get a single task attempt but iterate with tool use, while one-shot evaluation draws 50 independent samples per problem.

---

## 📝 One-shot Leaderboard (new grader)

One-shot ("classic") evaluation generates code in a single pass with **no tool use and no extended thinking**, 50 samples per problem (temp 0.8, seed 1234).

| Model | Mode | Pass@1 | Pass@5 | Pass@10 | Pass@20 | Pass@50 |
|-------|------|--------|--------|---------|---------|---------|
| **Claude Opus 4.8** | No extended thinking | **53.2%** | 74.4% | 80.7% | 85.7% | 89.6% |
| **GPT-5.5** | Reasoning | *run pending* | — | — | — | — |

> **The agent gap:** Opus 4.8 scores **53.2%** one-shot (Pass@1) vs **87–88%** in agent mode — a **+34-point** lift from iteration and tool use. Even Pass@10 (80.7%, ten independent tries) doesn't reach the single-attempt agent score.

---

## 📚 Historical Leaderboard (original / pre-#7 grader)

> ⚠️ These one-shot results were scored with the **pre-#7 grader** and are **not comparable** to the sections above. Under the new grader they would recover roughly +3 to +13 points. Preserved here for continuity.

The following table shows model performance on the Q-HumanEval dataset, ranked by Pass@1 score:

| Rank | Model | Type | Size | Pass@1 | Pass@5 | Pass@10 |
|------|-------|------|------|--------|--------|---------|
| 🥇 | **qqWen**<br/>*Morgan Stanley* | 🔓 Open Source | **72B** | 45.10% | 59.24% | 62.63% |
| 🥈 | **Grok**<br/>*xAI* | 🧠 Reasoning (medium) | *Unknown* | 43.37% | 68.45% | 74.32% |
| 🥉 | **Claude 4 Sonnet**<br/>*Anthropic* | 🧠 Reasoning (medium) | *Unknown* | 37.70% | 53.47% | 59.13% |
| 4 | **Gemini 2.5 pro**<br/>*Google* | 🧠 Reasoning (medium) | *Unknown* | 27.75% | 51.41% | 59.68% |
| 5 | **GPT-5**<br/>*OpenAI* | 🧠 Reasoning (medium) | *Unknown* | 27.36% | 54.96% | 65.05% |
| 6 | **o3**<br/>*OpenAI* | 🧠 Reasoning (medium) | *Unknown* | 18.42% | 40.93% | 52.15% |
| 7 | **GPT-4o**<br/>*OpenAI* | 🔒 Proprietary | *Unknown* | 14.42% | 24.49% | 29.44% |
| 8 | **Llama 3.3 70B**<br/>*Meta* | 🔓 Open Source | **70B** | 10.12% | 16.69% | 20.14% |
| 9 | **DeepSeek-R1-Distill-Qwen-32B**<br/>*DeepSeek* | 🧠 Reasoning (medium) | **32B** | 9.32% | 17.59% | 22.10% |
| 10 | **Qwen3 Coder 30B A3B**<br/>*Alibaba* | 🔓 Open Source | **30B** | 8.29% | 13.51% | 16.45% |
| 11 | **Gemma 3 12B**<br/>*Google* | 🔓 Open Source | **12B** | 4.15% | 6.22% | 6.66% |
| 12 | **Gemma 3 4B**<br/>*Google* | 🔓 Open Source | **4B** | 3.02% | 4.26% | 4.60% |

---

## 📊 Statistics

- **Agent leaderboard (new grader):** 14 models, 22 runs
- **One-shot leaderboard (new grader):** 1 model (Claude Opus 4.8); GPT-5.5 run pending
- **Historical one-shot (pre-#7 grader):** 12 models
  - 🧠 Reasoning: 6 · 🔒 Proprietary: 1 · 🔓 Open Source: 5

### 🏆 Best Scores
- **Highest agent Pass@1 (new grader):** Claude Fable 5.1, 98.2% (clean room), at $0.252 per task; Opus 5.5 is one problem behind at about a third of the cost, and Haiku 5.5 (medium, skilled) is two behind at 3% of it
- **Lowest agent cost per task (new grader):** Claude Haiku 5.5, $0.0076 at 93.9% (clean room); at medium effort with the skill it scores 97.0% for $0.0081, about a fifth of Sonnet 5.5's cost per task
- **Best OpenAI agent (new grader):** GPT-6 Sol, 95.1% with the q-kdb skill at $0.088 per task (91.5% clean room at $0.080)
- **Highest one-shot Pass@1 (new grader):** Claude Opus 4.8 53.2% (Pass@10 80.7%)
- **Highest historical Pass@1 (pre-#7 grader):** qqWen 45.10% — best Pass@5/Pass@10: Grok (68.45% / 74.32%)

## 🔬 Methodology

### Grader versions
- **New grader (current):** [PR #7](https://github.com/KxSystems/q-evaluation-harness/pull/7) fixed false-negative test failures (char-vector/atom handling, q long-null mapping, char-atom dict keys) and added an `errored` status that excludes q startup/license infrastructure errors from pass/fail. Net effect: **+3 to +13 points** across models, **zero regressions**.
- **Original grader (historical):** everything in the Historical section was scored before #7 and is not comparable to the new-grader sections.

### Evaluation modes
- **Agent mode:** one task attempt with iterative tool use against a live Q interpreter; `--timeout 300` per problem (600 for Opus 5 and later), Codex at `--reasoning-effort high`. Optionally install the q-kdb skill via `--skill-dirs` (the **Skill** column); `--no-skills` is the clean-room baseline.
- **One-shot ("classic") mode:** 50 independent samples per problem, temperature 0.8, seed 1234, no tool use. Claude models run with **no extended thinking** (for comparability with the historical board); reasoning models keep their reasoning config.

### Evaluation Metrics
- **Pass@k:** the percentage of problems solved when generating k samples per problem.
- **Pass@1:** single-attempt success rate (most restrictive). For agent mode this is the single task attempt.

### Model Categories
- 🧠 **Reasoning Models:** Advanced models with enhanced reasoning capabilities
- 🔒 **Proprietary Models:** Closed-source commercial models
- 🔓 **Open Source Models:** Publicly available models

### Evaluation Process
All models are evaluated using the same standardized process:
1. Solutions generated per the mode (one-shot: 50 samples per problem; agent: a single iterative attempt per problem)
2. Consistent prompting and evaluation criteria
3. Automated scoring using the Q evaluation harness
4. Results verified for accuracy and reproducibility

### Cost capture
- **Agent / Claude Code:** API-equivalent cost is computed per run from the token counts in each task's final result event, at the provider's published list prices (`src/agents/pricing.py`, source URL and as-of date), including prompt caching (discounted cache reads, 1-hour cache writes at 2x input). The CLI's own `total_cost_usd` is kept alongside for audit (`cli_reported_cost_usd` in the results files). It matches to the cent for Opus 5.5 and Sonnet 5, but is wrong for two models: it priced Sonnet 5.5 cache reads at twice the published rate (+15%), and does not recognize Haiku 5.5 (about 32x too high). Haiku 5.5 is also priced by prompt length, so the harness refuses to write a result if any final-attempt request exceeds 100k tokens. Subscription-covered in practice, so ~$0 cash.
- **Cost per task:** total cost ÷ tasks attempted (164), counting each task's final attempt only. This follows the per-task cost Databricks reports in [Benchmarking Coding Agents on Databricks' Multi-Million Line Codebase](https://www.databricks.com/blog/benchmarking-coding-agents-databricks-multi-million-line-codebase). Failed attempts are included in the spend, so a model that fails expensively pays for it.
- **Agent / Codex:** the CLI records tokens but not dollars, so the harness computes cost from token counts at OpenAI's list prices (`src/agents/pricing.py`, with source URL and as-of date). Cached input is billed at the cached rate and reasoning tokens are billed as output. Runs before this was added (the April GPT-5.5 skilled run) show "—".
- **One-shot / classic:** no token or cost data is recorded.

### Statistical Rigor
For reliable Pass@k evaluation, Q-HumanEval (164 problems) requires at least 50 samples per problem to achieve statistically significant results with ±3 percentage-point confidence intervals at 95% confidence using Wilson confidence intervals. (This applies to one-shot Pass@k; agent mode is a single task attempt and is reported as Pass@1.)

---

**Last Updated:** October 8, 2026 | **Version:** 2.0.0

*Want to submit your model? Check out our [submission guide](submission_guide.md) for details.*
