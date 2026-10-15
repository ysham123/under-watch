"""Part 10: take turns. The saboteur learns to evade the monitor, the monitor retrains on
every saboteur seen so far, and we watch who keeps the upper hand.

Run:  python scripts/part10_arms_race.py mlp     (about 2 minutes)
      python scripts/part10_arms_race.py rgcn    (about 7 minutes)
"""

import json
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underwatch import monitors, ppo, rollout, store  # noqa: E402

KIND = sys.argv[1] if len(sys.argv) > 1 else "mlp"
ROUNDS = 3
LAM = 20.0
POOL_PER_SABOTEUR = 2048

team = store.load("checkpoints/team.pkl")
apply_fn = monitors.MONITORS[KIND][1]
cfg = ppo.Config()


def merge(datasets):
    """Stack several (traj, landmarks, labels) datasets along the worlds axis."""
    traj = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=1),
                                  *[d[0] for d in datasets])
    landmarks = jnp.concatenate([d[1] for d in datasets])
    labels = jnp.concatenate([d[2] for d in datasets])
    return traj, landmarks, labels


saboteurs = [store.load("checkpoints/saboteur_naive.pkl")]
tests = [rollout.collect(team, saboteurs[0], jax.random.PRNGKey(11), 2048)]
pool = [rollout.collect(team, saboteurs[0], jax.random.PRNGKey(10), POOL_PER_SABOTEUR)]
watcher = store.load(f"checkpoints/monitor_{KIND}.pkl")
log = []

for rnd in range(1, ROUNDS + 1):
    print(f"\n=== round {rnd}: saboteur attacks the current {KIND} monitor ===")
    frozen = watcher                                   # bind this round's monitor
    update = ppo.make_saboteur_update(cfg, team, lambda tr, lm: apply_fn(frozen, tr, lm),
                                      LAM)
    saboteur, _ = ppo.run(update, team, jax.random.PRNGKey(rnd), 1_000_000, cfg,
                          every=10_000)
    saboteurs.append(saboteur)
    tests.append(rollout.collect(team, saboteur, jax.random.PRNGKey(11 + rnd), 2048))
    pool.append(rollout.collect(team, saboteur, jax.random.PRNGKey(100 + rnd),
                                POOL_PER_SABOTEUR))
    before = monitors.evaluate_monitor(KIND, watcher, tests[-1])

    print(f"=== round {rnd}: monitor retrains on all {len(pool)} saboteurs so far ===")
    watcher = monitors.train_monitor(KIND, merge(pool), jax.random.PRNGKey(rnd), log=None)
    after = [monitors.evaluate_monitor(KIND, watcher, t)["auroc"] for t in tests]
    entry = {"round": rnd, "damage": rollout.damage(team, saboteur, jax.random.PRNGKey(99)),
             "auroc_before_retrain": before["auroc"], "auroc_after_retrain": after[-1],
             "auroc_on_all_saboteurs": after}
    log.append(entry)
    print(json.dumps(entry))

os.makedirs("results", exist_ok=True)
with open(f"results/arms_race_{KIND}.json", "w") as f:
    json.dump(log, f, indent=1)
print(f"\nsaved results/arms_race_{KIND}.json\n")

print(f"Arms race against the {KIND} monitor (AUROC on that round's saboteur)")
print(" round  damage  before retraining  after retraining")
for e in log:
    print(f"{e['round']:6d}  {e['damage']:6.2f}  {e['auroc_before_retrain']:17.3f}  "
          f"{e['auroc_after_retrain']:16.3f}")
