"""Replicate the headline pressure result across independent seeds.

For every seed this retrains everything from scratch: the honest team, the unpressured
saboteur, the MLP monitor, and a saboteur trained against that frozen monitor at
lambda = 20. It then measures damage and AUROC for the frozen and a retrained monitor.

Run:  python scripts/replicate_seeds.py 1 2 3 4 5     (about 3 minutes per seed)
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underwatch import env, monitors, ppo, rollout  # noqa: E402

LAM = 20.0
SEEDS = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4, 5]
OUT = os.environ.get("OUT", "results/seeds_mlp.json")
cfg = ppo.Config()
apply_fn = monitors.MONITORS["mlp"][1]


def behaviour(team, saboteur, key):
    """Saboteur / honest ratios of mean speed and collision steps."""
    play = jax.jit(rollout.episode, static_argnums=(2, 4))
    traj, final, landmarks, mask = play(team, key, 2048, saboteur, False)
    speed = jnp.linalg.norm(traj.vel, axis=-1).mean(axis=0)
    coll = traj.collisions.sum(axis=0)
    gap = env.landmark_distances(final.pos, landmarks).min(axis=-1)
    out = {}
    for name, x in (("speed", speed), ("collisions", coll), ("gap", gap)):
        sab = float((x * mask).sum() / mask.sum())
        honest = float((x * ~mask).sum() / (~mask).sum())
        out[name] = [sab, honest]
    return out


rows = []
for seed in SEEDS:
    k = lambda i: jax.random.PRNGKey(1000 * seed + i)  # noqa: E731
    team, _ = ppo.run(ppo.make_team_update(cfg), ppo.init_team(k(0)), k(1), 2_000_000, cfg, log=None)
    naive, _ = ppo.run(ppo.make_saboteur_update(cfg, team), team, k(2), 1_000_000, cfg, log=None)
    watcher = monitors.train_monitor("mlp", rollout.collect(team, naive, k(3), 4096), k(4), log=None)
    test_naive = rollout.collect(team, naive, k(5), 2048)

    def frozen(traj, landmarks, w=watcher):
        return apply_fn(w, traj, landmarks)

    evader, _ = ppo.run(ppo.make_saboteur_update(cfg, team, frozen, LAM), team, k(6), 1_000_000,
                        cfg, log=None)
    test_evader = rollout.collect(team, evader, k(7), 2048)
    retrained = monitors.train_monitor("mlp", rollout.collect(team, evader, k(8), 4096), k(9),
                                       log=None)
    honest_play = jax.jit(rollout.episode, static_argnums=(2, 4))(team, k(10), 2048, None, False)
    row = {
        "seed": seed,
        "team_final_distance": float(rollout.summary(*honest_play[:3])["final_distance"]),
        "damage_naive": rollout.damage(team, naive, k(10)),
        "damage_evader": rollout.damage(team, evader, k(10)),
        "auroc_naive": monitors.evaluate_monitor("mlp", watcher, test_naive)["auroc"],
        "auroc_frozen_on_evader": monitors.evaluate_monitor("mlp", watcher, test_evader)["auroc"],
        "auroc_retrained_on_evader": monitors.evaluate_monitor("mlp", retrained, test_evader)["auroc"],
        "evader_behaviour": behaviour(team, evader, k(11)),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=1)

print("saved", OUT)
for name in ("damage_naive", "damage_evader", "auroc_naive", "auroc_frozen_on_evader",
             "auroc_retrained_on_evader"):
    values = np.array([r[name] for r in rows])
    print(f"{name:28s} mean {values.mean():.3f}  min {values.min():.3f}  max {values.max():.3f}")
