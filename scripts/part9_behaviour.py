"""Part 9 (follow-up): what does a saboteur actually do differently from honest agents?

Run after part9_pressure.py for both monitors:  python scripts/part9_behaviour.py
"""

import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underwatch import env, rollout, store  # noqa: E402

team = store.load("checkpoints/team.pkl")
play = jax.jit(rollout.episode, static_argnums=(2, 4))

print("Each pair is saboteur / honest agents, averaged over 2,048 episodes")
print(f"{'saboteur':22s} {'gap to landmark':>16s} {'left the square':>16s} "
      f"{'speed':>12s} {'collisions':>12s}")
for name in ["saboteur_naive", "saboteur_mlp_lam20", "saboteur_rgcn_lam20"]:
    saboteur = store.load(f"checkpoints/{name}.pkl")
    traj, final, landmarks, mask = play(team, jax.random.PRNGKey(5), 2048, saboteur, False)
    measures = {
        "gap": env.landmark_distances(final.pos, landmarks).min(axis=-1),
        "left": (jnp.abs(final.pos) > 1.0).any(axis=-1),                   # outside start area
        "speed": jnp.linalg.norm(traj.vel, axis=-1).mean(axis=0),
        "collisions": traj.collisions.sum(axis=0),
    }
    cells = []
    for values in measures.values():
        sab = float((values * mask).sum() / mask.sum())
        honest = float((values * ~mask).sum() / (~mask).sum())
        cells.append(f"{sab:.2f} / {honest:.2f}")
    print(f"{name:22s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>12s} {cells[3]:>12s}")
