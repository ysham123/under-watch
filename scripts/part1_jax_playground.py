"""Part 1: the handful of JAX ideas this whole project uses.

Run:  python scripts/part1_jax_playground.py
"""

import time

import jax
import jax.numpy as jnp

# 1. Arrays have a shape. Read it left to right: worlds, agents, (x, y).
pos = jnp.zeros((4, 3, 2))
print("1. shape", pos.shape, " number of entries", pos.size)

# 2. Indexing and axes.
x = jnp.arange(24.0).reshape(2, 3, 4)
print("2. x[0].shape", x[0].shape, " x[:, 1].shape", x[:, 1].shape,
      " x[..., 0].shape", x[..., 0].shape)
print("   x.sum(axis=-1).shape", x.sum(axis=-1).shape, " (the summed axis disappears)")

# 3. Broadcasting: None inserts an axis of size 1, which then stretches to match.
agents = jnp.array([[0.0, 0.0], [1.0, 0.0]])            # (2 agents, 2)
landmarks = jnp.array([[0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])  # (3 landmarks, 2)
gaps = agents[:, None, :] - landmarks[None, :, :]       # (2, 1, 2) - (1, 3, 2) -> (2, 3, 2)
dist = jnp.sqrt(jnp.sum(gaps ** 2, axis=-1))            # (2, 3): agent to landmark
print("3. distance table\n", dist)

# 4. Arrays never change in place. .at[...] returns an updated copy.
v = jnp.zeros(3)
w = v.at[0].set(5.0).at[2].add(1.0)
print("4. v", v, " w", w)

# 5. Randomness needs an explicit key. Split it, use each piece once.
key = jax.random.PRNGKey(0)
key, k1, k2 = jax.random.split(key, 3)
print("5. same key, same numbers:", jax.random.uniform(k1, (2,)),
      jax.random.uniform(k1, (2,)))
print("   new key, new numbers:  ", jax.random.uniform(k2, (2,)))

# 6. jit compiles a function the first time it is called with a given shape.
def slow_fn(x):
    for _ in range(50):
        x = jnp.tanh(x @ x.T @ x)
    return x

fast_fn = jax.jit(slow_fn)
m = jnp.ones((64, 64)) * 0.01
for name, fn in [("plain", slow_fn), ("jit 1st call", fast_fn), ("jit 2nd call", fast_fn)]:
    start = time.time()
    fn(m).block_until_ready()
    print(f"6. {name:13s} {1000 * (time.time() - start):7.1f} ms")

# Inside jit, choose with jnp.where, not with a Python if on array values.
@jax.jit
def relu(x):
    return jnp.where(x > 0, x, 0.0)

print("   relu", relu(jnp.array([-1.0, 2.0])))

# 7. lax.scan: a loop that carries a value forward and stacks one output per step.
def step(total, x):
    total = total + x
    return total, total            # (new carry, output for this step)

final, running = jax.lax.scan(step, 0.0, jnp.array([1.0, 2.0, 3.0, 4.0]))
print("7. final", final, " running totals", running)

# 8. Pytrees: nested dicts and lists of arrays are treated as one object.
params = {"w": jnp.ones((2, 3)), "b": jnp.zeros(3)}
print("8. leaves", [p.shape for p in jax.tree_util.tree_leaves(params)])
print("   doubled", jax.tree_util.tree_map(lambda p: 2 * p, params)["w"][0])

# 9. jax.grad gives gradients with the same structure as the parameters.
def loss(params, x):
    return jnp.sum((x @ params["w"] + params["b"]) ** 2)

grads = jax.grad(loss)(params, jnp.array([[1.0, 2.0]]))
print("9. grad shapes", {k: g.shape for k, g in grads.items()}, " grad of b", grads["b"])
