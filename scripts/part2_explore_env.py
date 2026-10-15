"""Part 2: what the world looks like, and what random behaviour scores.

Run:  python scripts/part2_explore_env.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underwatch import env  # noqa: E402

key = jax.random.PRNGKey(0)
key, k_reset = jax.random.split(key)

# 1. Start 1,000 worlds at once (JAX runs them in parallel).
state = env.reset(k_reset, batch=1000)
print("Agent positions, world 0:\n", state.pos[0])
print("Landmark positions, world 0:\n", state.landmarks[0])

# 2. What one agent observes: 18 numbers.
obs = env.observe(state)
print("\nObservation shape (worlds, agents, numbers):", obs.shape)
print("Agent 0's observation in world 0:\n", jnp.round(obs[0, 0], 2))

# 3. Play one full episode with random actions, and keep the path of world 0.
path = [state.pos[0]]
total_reward = jnp.zeros((1000, env.N_AGENTS))
total_collisions = jnp.zeros((1000, env.N_AGENTS))
for t in range(env.MAX_CYCLES):
    key, k_act = jax.random.split(key)
    actions = jax.random.randint(k_act, (1000, env.N_AGENTS), 0, env.N_ACTIONS)
    state, reward, info = env.step(state, actions)
    total_reward += reward
    total_collisions += info["collisions"]
    path.append(state.pos[0])

final_distance = -env.global_reward(state.pos, state.landmarks)
print("\nRandom agents over 1,000 episodes:")
print(f"  return per agent      {float(total_reward.mean()):7.2f}  (higher is better)")
print(f"  final distance        {float(final_distance.mean()):7.2f}  (lower is better)")
print(f"  collisions per agent  {float(total_collisions.mean()):7.2f}  (lower is better)")

# 4. Draw world 0's episode.
path = jnp.stack(path)                                   # (26, 3, 2)
fig, ax = plt.subplots(figsize=(4.6, 4.6))
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
for i in range(env.N_AGENTS):
    ax.plot(path[:, i, 0], path[:, i, 1], color=colors[i], linewidth=1.5, alpha=0.8)
    ax.scatter(path[0, i, 0], path[0, i, 1], color=colors[i], s=30, marker="o")
    ax.add_patch(plt.Circle(path[-1, i], env.AGENT_SIZE, color=colors[i], alpha=0.35))
    ax.annotate(f"agent {i}", path[-1, i], xytext=(6, 6), textcoords="offset points",
                fontsize=8)
ax.scatter(state.landmarks[0, :, 0], state.landmarks[0, :, 1], marker="x", s=80,
           color="black", label="landmarks")
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
ax.set_title("One episode, random actions", fontsize=10)
ax.legend(loc="lower right", fontsize=8)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/part2_random_episode.png", dpi=140, bbox_inches="tight")
print("\nSaved figures/part2_random_episode.png")
