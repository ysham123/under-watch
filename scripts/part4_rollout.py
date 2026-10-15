"""Part 4: play a whole episode in 1,024 worlds at once, and see what jit buys you.

Run:  python scripts/part4_rollout.py
"""

import os
import sys
import time

import jax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underwatch import env, rollout  # noqa: E402
from underwatch.nets import actor_critic_init  # noqa: E402

team = actor_critic_init(jax.random.PRNGKey(0), env.OBS_DIM, env.N_ACTIONS)
play = jax.jit(rollout.episode, static_argnums=(2,))

for attempt in range(3):
    start = time.time()
    traj, final, landmarks, mask = play(team, jax.random.PRNGKey(attempt), 1024)
    traj.reward.block_until_ready()
    print(f"call {attempt}: {time.time() - start:.3f}s")

print("\nShapes (time, worlds, agents, ...):")
for name, value in traj._asdict().items():
    print(f"  {name:10s} {value.shape}")

stats = rollout.summary(traj, final, landmarks)
print("\nUntrained team:", {k: round(float(v), 2) for k, v in stats.items()})
