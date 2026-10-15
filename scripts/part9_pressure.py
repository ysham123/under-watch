"""Part 9: train saboteurs that are punished whenever a frozen monitor flags them.

Run:  python scripts/part9_pressure.py mlp     (about 3 minutes)
      python scripts/part9_pressure.py rgcn    (about 10 minutes)
"""

import json
import os
import sys

import jax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underwatch import monitors, ppo, rollout, store  # noqa: E402

KIND = sys.argv[1] if len(sys.argv) > 1 else "mlp"
LAMBDAS = [0.0, 5.0, 10.0, 20.0, 40.0]

team = store.load("checkpoints/team.pkl")
frozen = {k: store.load(f"checkpoints/monitor_{k}.pkl") for k in monitors.MONITORS}
apply_fn = monitors.MONITORS[KIND][1]
watcher = frozen[KIND]


def monitor(traj, landmarks):
    return apply_fn(watcher, traj, landmarks)


cfg = ppo.Config()
rows = []
for lam in LAMBDAS:
    print(f"\n=== lambda = {lam} (saboteur trained against the {KIND} monitor) ===")
    update = ppo.make_saboteur_update(cfg, team, monitor, lam)
    saboteur, _ = ppo.run(update, team, jax.random.PRNGKey(1), 1_000_000, cfg, every=10_000)
    store.save(f"checkpoints/saboteur_{KIND}_lam{lam:g}.pkl", saboteur)

    test = rollout.collect(team, saboteur, jax.random.PRNGKey(11), 2048)
    row = {"lambda": lam, "damage": rollout.damage(team, saboteur, jax.random.PRNGKey(99))}
    for k in monitors.MONITORS:          # both frozen monitors, including the one it evaded
        row[f"frozen_{k}"] = monitors.evaluate_monitor(k, frozen[k], test)
    fresh_train = rollout.collect(team, saboteur, jax.random.PRNGKey(10), 4096)
    fresh = monitors.train_monitor(KIND, fresh_train, jax.random.PRNGKey(3), log=None)
    row[f"retrained_{KIND}"] = monitors.evaluate_monitor(KIND, fresh, test)
    rows.append(row)
    print(json.dumps(row))

os.makedirs("results", exist_ok=True)
with open(f"results/pressure_{KIND}.json", "w") as f:
    json.dump(rows, f, indent=1)
print(f"\nsaved results/pressure_{KIND}.json\n")

print(f"Saboteurs trained against the {KIND} monitor (AUROC; 0.5 is chance)")
print(" lambda  damage  frozen mlp  frozen rgcn  retrained " + KIND)
for r in rows:
    print(f"{r['lambda']:7.0f}  {r['damage']:6.2f}  {r['frozen_mlp']['auroc']:10.3f}  "
          f"{r['frozen_rgcn']['auroc']:11.3f}  {r[f'retrained_{KIND}']['auroc']:12.3f}")
