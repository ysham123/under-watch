"""Part 3: a policy network that turns an observation into action probabilities.

Run:  python scripts/part3_policy.py
"""

import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from underwatch import env  # noqa: E402
from underwatch.nets import actor_critic_init, logits_fn, value_fn  # noqa: E402

key = jax.random.PRNGKey(0)
k_net, k_reset, k_act = jax.random.split(key, 3)

# 1. Build the network and look at its shapes.
params = actor_critic_init(k_net, env.OBS_DIM, env.N_ACTIONS)
print("Actor layer shapes:", [layer["w"].shape for layer in params["actor"]])
n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print("Total numbers to learn:", n_params)

# 2. Feed it observations from 4 worlds (3 agents each).
state = env.reset(k_reset, batch=4)
obs = env.observe(state)                       # (4, 3, 18)
logits = logits_fn(params, obs)                # (4, 3, 5)
probs = jax.nn.softmax(logits, axis=-1)
print("\nlogits shape", logits.shape, " probs shape", probs.shape)
print("Agent 0, world 0, action probabilities:", jnp.round(probs[0, 0], 3))
print("Probabilities add up to:", float(probs[0, 0].sum()))

# 3. Sample actions and work out how likely each choice was.
actions = jax.random.categorical(k_act, logits)          # (4, 3)
logp = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], axis=-1)[..., 0]
print("\nSampled actions:\n", actions)
print("log-probability of each chosen action:\n", jnp.round(logp, 3))
print("\nCritic's value guesses:\n", jnp.round(value_fn(params, obs), 3))
