"""Simple Spread (3 agents, 3 landmarks) as pure, batched JAX functions.

A port of PettingZoo MPE ``simple_spread_v3`` with its defaults: ``N=3``,
``local_ratio=0.5``, ``max_cycles=25``, discrete actions. Physics, rewards and the
observation layout match the reference implementation (version 1.24.3) step by step.

Every function takes a whole batch of worlds, so episodes can be compiled with ``jit``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

N_AGENTS = 3
N_LANDMARKS = 3
OBS_DIM = 18            # vel(2) + pos(2) + landmarks(3x2) + other agents(2x2) + comm(2x2)
N_ACTIONS = 5           # no-op, -x, +x, -y, +y
MAX_CYCLES = 25

DT = 0.1
DAMPING = 0.25
CONTACT_FORCE = 1e2
CONTACT_MARGIN = 1e-3
SENSITIVITY = 5.0
AGENT_SIZE = 0.15
LOCAL_RATIO = 0.5
OCCUPIED_RADIUS = 0.1

ACTION_TABLE = jnp.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])


class State(NamedTuple):
    pos: jax.Array        # (B, N, 2) agent positions
    vel: jax.Array        # (B, N, 2) agent velocities
    landmarks: jax.Array  # (B, L, 2) landmark positions
    t: jax.Array          # (B,) steps taken this episode


def reset(key: jax.Array, batch: int) -> State:
    """Agents and landmarks are drawn uniformly from [-1, 1]^2, as in the reference."""
    k_agents, k_landmarks = jax.random.split(key)
    pos = jax.random.uniform(k_agents, (batch, N_AGENTS, 2), minval=-1.0, maxval=1.0)
    landmarks = jax.random.uniform(k_landmarks, (batch, N_LANDMARKS, 2),
                                   minval=-1.0, maxval=1.0)
    return State(pos=pos, vel=jnp.zeros_like(pos), landmarks=landmarks,
                 t=jnp.zeros((batch,), dtype=jnp.int32))


def _pairwise(pos: jax.Array):
    delta = pos[:, :, None, :] - pos[:, None, :, :]     # delta[b, i, j] = pos_i - pos_j
    dist = jnp.sqrt(jnp.sum(delta ** 2, axis=-1))             # (B, N, N)
    off_diag = ~jnp.eye(N_AGENTS, dtype=bool)
    return delta, dist, off_diag


def collision_force(pos: jax.Array) -> jax.Array:
    """Soft contact force between agents (landmarks do not collide in this task)."""
    delta, dist, off_diag = _pairwise(pos)
    safe = jnp.where(off_diag, dist, 1.0)
    overlap = 2 * AGENT_SIZE - safe
    penetration = jnp.logaddexp(0.0, overlap / CONTACT_MARGIN) * CONTACT_MARGIN
    force = CONTACT_FORCE * delta / safe[..., None] * penetration[..., None]
    return jnp.sum(jnp.where(off_diag[None, :, :, None], force, 0.0), axis=2)


def collisions(pos: jax.Array) -> jax.Array:
    """Number of other agents each agent overlaps with, shape (B, N)."""
    _, dist, off_diag = _pairwise(pos)
    return jnp.sum((dist < 2 * AGENT_SIZE) & off_diag, axis=-1)


def landmark_distances(pos: jax.Array, landmarks: jax.Array) -> jax.Array:
    """Distance from every agent to every landmark, shape (B, N, L)."""
    return jnp.sqrt(jnp.sum((pos[:, :, None, :] - landmarks[:, None, :, :]) ** 2, axis=-1))


def global_reward(pos: jax.Array, landmarks: jax.Array) -> jax.Array:
    """Negative sum over landmarks of the distance to the nearest agent, shape (B,)."""
    return -jnp.sum(jnp.min(landmark_distances(pos, landmarks), axis=1), axis=-1)


def observe(state: State) -> jax.Array:
    """Per-agent observations in the reference layout, shape (B, N, 18)."""
    # rel_landmarks[b, i, l] = landmark_l - pos_i, shape (B, N, L, 2)
    rel_landmarks = state.landmarks[:, None, :, :] - state.pos[:, :, None, :]
    # rel_agents[b, i, j] = pos_j - pos_i
    rel_agents = state.pos[:, None, :, :] - state.pos[:, :, None, :]
    others = jnp.array([[j for j in range(N_AGENTS) if j != i] for i in range(N_AGENTS)])
    # keep only the other agents: (B, N, N-1, 2)
    rel_others = jnp.take_along_axis(rel_agents, others[None, :, :, None], axis=2)
    batch = state.pos.shape[0]
    comm = jnp.zeros((batch, N_AGENTS, 2 * (N_AGENTS - 1)), dtype=state.pos.dtype)
    return jnp.concatenate([
        state.vel, state.pos,
        rel_landmarks.reshape(batch, N_AGENTS, -1),
        rel_others.reshape(batch, N_AGENTS, -1),
        comm,
    ], axis=-1)


def step(state: State, actions: jax.Array):
    """Advance one step. ``actions`` has shape (B, N) with values in [0, 5).

    Returns ``(next_state, rewards (B, N), info)``.
    The episode ends when ``t == MAX_CYCLES``.
    """
    force = ACTION_TABLE[actions] * SENSITIVITY + collision_force(state.pos)
    pos = state.pos + state.vel * DT
    vel = state.vel * (1.0 - DAMPING) + force * DT
    nxt = State(pos=pos, vel=vel, landmarks=state.landmarks, t=state.t + 1)

    g = global_reward(pos, state.landmarks)
    n_coll = collisions(pos)
    rewards = g[:, None] * (1.0 - LOCAL_RATIO) - n_coll.astype(pos.dtype) * LOCAL_RATIO
    gaps = jnp.min(landmark_distances(pos, state.landmarks), axis=1)
    occupied = jnp.sum(gaps < OCCUPIED_RADIUS, axis=-1)
    info = {"global_reward": g, "collisions": n_coll, "occupied": occupied}
    return nxt, rewards, info
