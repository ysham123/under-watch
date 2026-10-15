"""Part 7: train a simple monitor that reads one agent's episode and flags saboteurs.

Run:  python scripts/part7_mlp_monitor.py    (under a minute)
"""

import os
import sys

import jax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underwatch import monitors, rollout, store  # noqa: E402

team = store.load("checkpoints/team.pkl")
saboteur = store.load("checkpoints/saboteur_naive.pkl")

train = rollout.collect(team, saboteur, jax.random.PRNGKey(10), 4096)   # for learning
test = rollout.collect(team, saboteur, jax.random.PRNGKey(11), 2048)    # never trained on

params = monitors.train_monitor("mlp", train, jax.random.PRNGKey(3))
print("held-out:", monitors.evaluate_monitor("mlp", params, test))
store.save("checkpoints/monitor_mlp.pkl", params)
