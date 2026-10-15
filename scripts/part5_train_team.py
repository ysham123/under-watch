"""Part 5: train the honest team with PPO and save it.

Run:  python scripts/part5_train_team.py        (about 2 minutes on a laptop)
"""

import json
import os
import sys
import time

import jax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underwatch import env, ppo, store  # noqa: E402

TOTAL_STEPS = 2_000_000
cfg = ppo.Config()
num_updates = TOTAL_STEPS // (cfg.num_envs * env.MAX_CYCLES)

key = jax.random.PRNGKey(0)
key, k_init = jax.random.split(key)
params = ppo.init_team(k_init)
opt_state = ppo.make_optimizer(cfg).init(params)
update = ppo.make_team_update(cfg)

history = []
start = time.time()
for i in range(num_updates):
    key, k = jax.random.split(key)
    params, opt_state, stats = update(params, opt_state, k)
    history.append({name: float(v) for name, v in stats.items()})
    if i % 100 == 0 or i == num_updates - 1:
        h = history[-1]
        print(f"update {i:5d}  return {h['return']:7.2f}  "
              f"final distance {h['final_distance']:5.2f}  ({time.time() - start:4.0f}s)")

store.save("checkpoints/team.pkl", params)
os.makedirs("results", exist_ok=True)
with open("results/team_history.json", "w") as f:
    json.dump(history, f)
print("saved checkpoints/team.pkl and results/team_history.json")
