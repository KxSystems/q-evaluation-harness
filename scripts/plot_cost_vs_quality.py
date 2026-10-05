#!/usr/bin/env python3
"""Cost vs. pass rate scatter for the agent leaderboard (Databricks-style).

Cost per task = total API-equivalent spend / tasks attempted (164), i.e. the
per-attempt metric Databricks reports in "Benchmarking Coding Agents on
Databricks' Multi-Million Line Codebase" (July 2026). List prices including
cache reads/writes, as reported by the Claude Code CLI (Codex: computed from
tokens, src/agents/pricing.py); only each task's final attempt counts
(harness-abort re-runs excluded).

Styled after kx.com: dark panel, yellow marks, red frontier, heavy
uppercase headline with one phrase in the link blue. One image serves both
light and dark pages (it is a self-contained dark card).

Writes docs/img/cost_vs_quality.png.

Usage: poetry run python scripts/plot_cost_vs_quality.py
"""
import os

import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 164

# (label, skill, passed, total_cost_usd); NEW marks the fall 2026 releases,
# which are left out of the previous frontier.
# Sources: outputs/results_agent_<model>_{noskill,skill}.json (Fable 5.1:
# final attempts only, $41.35). Opus 5 skill arm
# counts final attempts only ($60.82). Opus 4.8 as published in #9 (run data
# not in this checkout; --timeout 300). GPT rows: outputs/sweep_gpt{,_skilled}
# (Codex 0.160.0, 2026-10-02).
ROWS = [
    ("Fable 5.1",  False, 161, 41.35),
    ("Opus 5.5",   False, 160, 13.97),
    ("Sonnet 5.5", False, 158, 7.82),
    ("Sonnet 5",   False, 150, 19.87),
    ("Opus 5",     False, 160, 39.70),
    ("Opus 5",     True,  157, 60.82),
    ("Opus 4.8",   False, 143, 24.86),
    ("Opus 4.8",   True,  145, 52.40),
    ("GPT-5.5",     False, 149, 36.98),
    ("GPT-5.6 Sol", False, 147, 24.59),
    ("GPT-6 Sol",   False, 150, 13.05),
    ("GPT-6 Sol",   True,  156, 14.45),
]

NEW = {"Fable 5.1", "Opus 5.5", "Sonnet 5.5", "GPT-5.6 Sol", "GPT-6 Sol"}
# model whose clean-room -> skilled jump gets an arrow
SKILL_ARROW = "GPT-6 Sol"

# label offsets in points, hand-placed to avoid collisions
OFFSETS = {
    ("Opus 5.5", False): (0, -16), ("Sonnet 5.5", False): (0, -17),
    ("Sonnet 5", False): (11, -4), ("Opus 5", False): (8, -15),
    ("Opus 5", True): (11, -4), ("Opus 4.8", False): (11, -4),
    ("Opus 4.8", True): (11, -4), ("Fable 5.1", False): (11, 0),
    ("GPT-5.5", False): (11, -4), ("GPT-5.6 Sol", False): (11, -4),
    ("GPT-6 Sol", False): (-11, -4), ("GPT-6 Sol", True): (11, -4),
}

# kx.com palette (from the site's CSS custom properties)
BG = "#101820"         # dark section background
GRID = "#243040"
INK = "#ffffff"
INK2 = "#a5abb8"
MUTED = "#8891a2"
YELLOW = "#ffcb22"     # --color-primary
RED = "#de2848"
BLUE = "#3272d9"       # --color-link
# matplotlib only sees one face per macOS .ttc: Avenir (book) and Avenir Next (bold)
FONT = ["Avenir", "Helvetica Neue", "Arial", "sans-serif"]
BOLD = dict(fontfamily=["Avenir Next", "Helvetica Neue", "Arial", "sans-serif"],
            fontweight="bold")


def pareto(points):
    """Points not dominated on (lower cost, higher pass rate), cheapest first."""
    front, best = [], -1
    for c, p in sorted(points):
        if p > best:
            front.append((c, p))
            best = p
    return front


def headline(fig, x, y, parts, **kw):
    """Draw text runs side by side, each in its own colour."""
    r = fig.canvas.get_renderer()
    for text, color in parts:
        t = fig.text(x, y, text, color=color, **kw)
        x += t.get_window_extent(renderer=r).width / fig.bbox.width


def draw():
    plt.rcParams.update({"font.family": FONT, "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    pts = [(cost / N, 100 * p / N) for _, _, p, cost in ROWS]
    new = [name in NEW for name, *_ in ROWS]

    # Frontiers end at their last point; extending them would run through
    # dominated ties.
    prior = pareto([pt for pt, n in zip(pts, new) if not n])
    front = pareto(pts)
    px, py = zip(*prior)
    fx, fy = zip(*front)
    ax.plot(px, py, color=RED, lw=1.2, ls=(0, (1, 2.2)), alpha=0.7, zorder=1)
    ax.annotate("PREVIOUS FRONTIER", ((px[0] + px[-1]) / 2, (py[0] + py[-1]) / 2),
                xytext=(10, -6), textcoords="offset points", fontsize=7.5,
                color=MUTED, **BOLD)
    ax.plot(fx, fy, color=RED, lw=1.3, alpha=0.85, solid_capstyle="round", zorder=2)
    # label the longest frontier segment, where there is room under the line
    k = max(range(len(fx) - 1), key=lambda i: fx[i + 1] - fx[i])
    ax.annotate("CURRENT FRONTIER", ((fx[k] + fx[k + 1]) / 2, (fy[k] + fy[k + 1]) / 2),
                xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7.5,
                color=RED, **BOLD)

    # the skill's effect on one model: clean-room point -> skilled point
    (c0, y0), (c1, y1) = [(x, y) for (name, _, _, _), (x, y) in zip(ROWS, pts)
                          if name == SKILL_ARROW]
    ax.annotate("", xy=(c1, y1), xytext=(c0, y0),
                arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.2",
                                color=MUTED, lw=1, shrinkA=6, shrinkB=7))
    ax.annotate(f"+{y1 - y0:.1f} pts with skill", ((c0 + c1) / 2, (y0 + y1) / 2),
                xytext=(8, 0), textcoords="offset points", va="center",
                fontsize=8, color=INK2)

    for (name, skill, p, cost), (x, y) in zip(ROWS, pts):
        ax.scatter(x, y, s=80, marker="s" if skill else "o",
                   color=YELLOW,
                   edgecolors=BG, linewidths=2, zorder=4)
        dx, dy = OFFSETS[(name, skill)]
        ax.annotate(name + (" + skill" if skill else ""), (x, y), xytext=(dx, dy),
                    textcoords="offset points",
                    ha="right" if dx < 0 else "center" if dx == 0 else "left",
                    va="center", fontsize=9, color=INK)

    ax.set_xlim(0, 0.42)
    ax.set_ylim(85, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:.2f}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("COST PER TASK  (USD, LIST PRICE INCL. CACHING)", color=INK2,
                  fontsize=8, labelpad=8, **BOLD)
    ax.set_ylabel("PASS@1", color=INK2, fontsize=8, labelpad=8, **BOLD)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, length=0, labelsize=8.5)

    fig.text(0.035, 0.955, "Q-HUMANEVAL  ·  AGENT MODE", fontsize=8.5,
             color=YELLOW, va="top", **BOLD)
    headline(fig, 0.035, 0.915, [("SAME SCORE, ", INK), ("A THIRD OF THE PRICE", BLUE)],
             fontsize=17, va="top", **BOLD)
    fig.text(0.035, 0.845, "Claude and GPT models on 164 q tasks. Cost per task = "
             "total spend ÷ tasks attempted, final attempts only.",
             fontsize=8.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.12)

    out = os.path.join(REPO, "docs", "img", "cost_vs_quality.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print("wrote", os.path.relpath(out, REPO))


if __name__ == "__main__":
    draw()
