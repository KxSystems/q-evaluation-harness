# 🏆 Q Programming Language Leaderboard

Welcome to the official leaderboard for Q programming language model evaluation! This leaderboard tracks the performance of various language models on Q/kdb+ programming tasks.

> ⚠️ **Grader change (June 2026):** A grader fix ([PR #7](https://github.com/KxSystems/q-evaluation-harness/pull/7)) removed a class of false-negative test failures, lifting every benchmarked model by **+3 to +13 points** with no regressions. Results scored with the **new grader** are **not comparable** to the historical (pre-#7) numbers. The sections below are grouped by grader version — do not compare across them.

---

## 🤖 Agent Leaderboard (new grader)

Agent mode gives the model a Q interpreter and lets it iteratively write, run, and fix its solution (one task attempt, multiple turns). Run with `--timeout 300`; Codex at `--reasoning-effort high`.

| Rank | Model | Backend | Skill | Pass@1 | Cost/run | Mean turns |
|------|-------|---------|:-----:|--------|----------|-----------|
| 🥇 | **Fable 5** \* | Claude Code | ✓ | **95.1%** (156/164) | — | — |
| 🥈 | **GPT-5.5** | Codex | ✓ | **94.5%** (155/164) | — † | 18.8 |
| 🥉 | **Claude Opus 4.8** | Claude Code | ✓ | 88.4% (145/164) | $52.40 | 8.7 |
| 4 | **GPT-5.5** | Codex | ✗ | 88.4% (145/164) | — † | 20.3 |
| 5 | **Claude Opus 4.8** | Claude Code | ✗ | 87.2% (143/164) | $24.86 | 5.8 |
| 5 | **Claude Opus 4.7** \* | Claude Code | ✓ | 87.2% (143/164) | — | — |
| 7 | **Claude Sonnet 4.6** \* | Claude Code | ✓ | 70.1% (115/164) | — | — |
| 8 | **Claude Haiku 4.5** \* | Claude Code | ✓ | 37.2% (61/164) | — | — |

**Skill column** (`--skill-dirs` q-kdb skill, ✓ = installed, ✗ = clean-room `--no-skills`): the skill helps **GPT-5.5 by +6.1 pts** and **Opus 4.8 by +1.2 pts** (a near-ceiling model). All arms had zero infrastructure errors.

\* **Reference rows** — prior overnight runs re-graded with the new grader, included for breadth. Caveats: run on a different machine, mostly `--timeout 600` with 5-hour-cap resume (so wall-time/cost are unreliable and omitted), and three dataset prompts changed after these runs (≤2 pt effect). These pre-date the `--no-skills` flag, so they used the default skill workflow and are reported as skill ✓.

† Codex (GPT-5.5) cost is **not captured** by the CLI — tokens are recorded but dollar cost is not surfaced. Claude Code reports the API-equivalent cost (subscription-covered, so ~$0 cash in practice).

> **Cross-check:** an independent prior agentic GPT-5.5 run re-grades to **93.3%** (153/164), matching the fresh 94.5% within run-to-run variance.

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

- **Agent leaderboard (new grader):** 6 models, 8 runs
- **One-shot leaderboard (new grader):** 1 model (Claude Opus 4.8); GPT-5.5 run pending
- **Historical one-shot (pre-#7 grader):** 12 models
  - 🧠 Reasoning: 6 · 🔒 Proprietary: 1 · 🔓 Open Source: 5

### 🏆 Best Scores
- **Highest agent Pass@1 (new grader):** GPT-5.5 94.5% (skill ✓) — Fable 5 95.1% (reference)
- **Highest one-shot Pass@1 (new grader):** Claude Opus 4.8 53.2% (Pass@10 80.7%)
- **Highest historical Pass@1 (pre-#7 grader):** qqWen 45.10% — best Pass@5/Pass@10: Grok (68.45% / 74.32%)

## 🔬 Methodology

### Grader versions
- **New grader (current):** [PR #7](https://github.com/KxSystems/q-evaluation-harness/pull/7) fixed false-negative test failures (char-vector/atom handling, q long-null mapping, char-atom dict keys) and added an `errored` status that excludes q startup/license infrastructure errors from pass/fail. Net effect: **+3 to +13 points** across models, **zero regressions**.
- **Original grader (historical):** everything in the Historical section was scored before #7 and is not comparable to the new-grader sections.

### Evaluation modes
- **Agent mode:** one task attempt with iterative tool use against a live Q interpreter; `--timeout 300` per problem, Codex at `--reasoning-effort high`. Optionally install the q-kdb skill via `--skill-dirs` (the **Skill** column); `--no-skills` is the clean-room baseline.
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
- **Agent / Claude Code:** API-equivalent cost is recorded per run (subscription-covered in practice, so ~$0 cash).
- **Agent / Codex:** the CLI records tokens but **not** dollar cost, so GPT-5.5 agent cost is shown as "—".
- **One-shot / classic:** no token or cost data is recorded.

### Statistical Rigor
For reliable Pass@k evaluation, Q-HumanEval (164 problems) requires at least 50 samples per problem to achieve statistically significant results with ±3 percentage-point confidence intervals at 95% confidence using Wilson confidence intervals. (This applies to one-shot Pass@k; agent mode is a single task attempt and is reported as Pass@1.)

---

**Last Updated:** June 26, 2026 | **Version:** 2.0.0

*Want to submit your model? Check out our [submission guide](submission_guide.md) for details.*
