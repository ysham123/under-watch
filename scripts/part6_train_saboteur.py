"""Part 6: turn one teammate into a saboteur and train it to hurt the team.

Run:  python scripts/part6_train_saboteur.py    (about 1 minute)
"""

import os
import sys

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underwatch import env, ppo, rollout, store  # noqa: E402

team = store.load("checkpoints/team.pkl")
cfg = ppo.Config()

# The saboteur starts as an exact copy of an honest agent, then learns its own goal.
update = ppo.make_saboteur_update(cfg, team)
saboteur, history = ppo.run(update, team, jax.random.PRNGKey(1), 1_000_000, cfg)
store.save("checkpoints/saboteur_naive.pkl", saboteur)

harm = rollout.damage(team, saboteur, jax.random.PRNGKey(99))
print(f"\nDamage: the saboteur adds {harm:.2f} to the team's final distance")

# Draw one episode with the saboteur in red.
traj, final, landmarks, mask = jax.jit(rollout.episode, static_argnums=(2, 4))(
    team, jax.random.PRNGKey(7), 1, saboteur, False)
fig, ax = plt.subplots(figsize=(4.6, 4.6))
for i in range(env.N_AGENTS):
    bad = bool(mask[0, i])
    path = traj.pos[:, 0, i]
    color = "#d6312a" if bad else "#2a78d6"
    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.8)
    ax.scatter(path[-1, 0], path[-1, 1], color=color, s=60, zorder=3)
    ax.annotate("saboteur" if bad else f"honest {i}", path[-1], xytext=(6, 6),
                textcoords="offset points", fontsize=8)
ax.scatter(landmarks[0, :, 0], landmarks[0, :, 1], marker="x", s=80, color="black")
edge = max(1.6, float(abs(traj.pos[:, 0]).max()) + 0.3)
ax.set_xlim(-edge, edge); ax.set_ylim(-edge, edge); ax.set_aspect("equal")
ax.set_title("One episode with a hidden saboteur", fontsize=10)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/part6_saboteur_episode.png", dpi=140, bbox_inches="tight")
print("saved figures/part6_saboteur_episode.png")
