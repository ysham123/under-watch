"""Part 8: a graph monitor (RGCN) that reads the whole scene as typed relations.

Run:  python scripts/part8_rgcn_monitor.py    (one to two minutes)
"""

import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underwatch import monitors, rollout, store  # noqa: E402

team = store.load("checkpoints/team.pkl")
saboteur = store.load("checkpoints/saboteur_naive.pkl")

train = rollout.collect(team, saboteur, jax.random.PRNGKey(10), 4096)
test = rollout.collect(team, saboteur, jax.random.PRNGKey(11), 2048)

# Before training, look at one scene as a graph: the world with the most "near" edges
# at the last step.
nodes, adj = monitors.graph_inputs(*test[:2])
near_edges = adj[:, -1, 1::2].sum(axis=(1, 2, 3))      # relations 1, 3, 5 are "near"
w = int(jnp.argmax(near_edges))
names = ["agent0", "agent1", "agent2", "lm0", "lm1", "lm2"]
relations = ["a->a far", "a->a near", "lm->a far", "lm->a near", "a->lm far", "a->lm near"]
print(f"Edges in world {w} at the last step (saboteur is agent{int(test[2][w].argmax())}):")
for r, rel in enumerate(relations):
    edges = [f"{names[j]}->{names[i]}" for i, j in zip(*jnp.nonzero(adj[w, -1, r]))]
    print(f"  {rel:10s} {', '.join(edges) if edges else '(none)'}")

params = monitors.train_monitor("rgcn", train, jax.random.PRNGKey(3))
print("held-out:", monitors.evaluate_monitor("rgcn", params, test))
store.save("checkpoints/monitor_rgcn.pkl", params)
