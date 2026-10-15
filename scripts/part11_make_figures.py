"""Part 11: turn the saved results into figures.

Run:  python scripts/part11_make_figures.py
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COLORS = {"mlp": "#2a78d6", "rgcn": "#eb6834"}
NAMES = {"mlp": "MLP monitor", "rgcn": "RGCN monitor"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def team_curve():
    history = load("results/team_history.json")
    distance = np.array([h["final_distance"] for h in history])
    smooth = np.convolve(distance, np.ones(25) / 25, mode="valid")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(distance, color=COLORS["mlp"], alpha=0.25, linewidth=0.8)
    ax.plot(np.arange(len(smooth)) + 12, smooth, color=COLORS["mlp"], linewidth=2)
    ax.axhline(2.30, color=MUTED, linestyle="--", linewidth=1.2)
    ax.annotate("random play", (len(distance), 2.30), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=9, color=MUTED)
    ax.set_xlabel("PPO update (64 episodes each)")
    ax.set_ylabel("Final distance (lower is better)")
    ax.set_title("Honest team learning", loc="left")
    fig.savefig("figures/part5_team_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def pressure():
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for kind in NAMES:
        rows = load(f"results/pressure_{kind}.json")
        if rows is None:
            continue
        lam = [r["lambda"] for r in rows]
        left.plot(lam, [r["damage"] for r in rows], "o-", color=COLORS[kind], linewidth=2)
        right.plot(lam, [r[f"frozen_{kind}"]["auroc"] for r in rows], "o-",
                   color=COLORS[kind], linewidth=2)
        right.plot(lam, [r[f"retrained_{kind}"]["auroc"] for r in rows], "o--",
                   color=COLORS[kind], linewidth=1.5, alpha=0.7)
    left.set_xlabel("Penalty for being flagged, λ")
    left.set_ylabel("Damage (extra final distance)")
    left.set_title("Harm the saboteur still does", loc="left")
    right.axhline(0.5, color=MUTED, linestyle=":", linewidth=1)
    right.annotate("chance", (right.get_xlim()[1], 0.5), xytext=(0, 4),
                   textcoords="offset points", ha="right", fontsize=9, color=MUTED)
    right.set_xlabel("Penalty for being flagged, λ")
    right.set_ylabel("AUROC")
    right.set_ylim(0.4, 1.02)
    right.set_title("Can the monitor still tell?", loc="left")
    handles = [Line2D([], [], color=COLORS[k], marker="o", linewidth=2) for k in NAMES]
    handles += [Line2D([], [], color=MUTED, linewidth=2),
                Line2D([], [], color=MUTED, linestyle="--", linewidth=1.5)]
    labels = [f"{NAMES[k]} in the loop" for k in NAMES]
    labels += ["frozen monitor it trained against", "monitor retrained on the new saboteur"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/part9_pressure.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def arms_race():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for kind in NAMES:
        log = load(f"results/arms_race_{kind}.json")
        if log is None:
            continue
        rounds = [e["round"] for e in log]
        ax.plot(rounds, [e["auroc_before_retrain"] for e in log], "o-", color=COLORS[kind],
                linewidth=2, label=f"{NAMES[kind]}, before retraining")
        ax.plot(rounds, [e["auroc_after_retrain"] for e in log], "o--", color=COLORS[kind],
                linewidth=1.5, alpha=0.7, label=f"{NAMES[kind]}, after retraining")
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Round")
    ax.set_ylabel("AUROC on that round's saboteur")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Arms race: attack, then retrain", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    fig.savefig("figures/part10_arms_race.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


os.makedirs("figures", exist_ok=True)
team_curve()
pressure()
arms_race()
print("wrote figures/part5_team_curve.png, part9_pressure.png, part10_arms_race.png")
