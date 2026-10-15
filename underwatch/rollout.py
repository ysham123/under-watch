"""Play whole episodes in many worlds at once.

The honest team shares one set of parameters. If a saboteur is given, one agent per
world (chosen at random) is controlled by the saboteur's network instead, and nobody
is told which.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from underwatch import env
from underwatch.nets import logits_fn, value_fn


class Trajectory(NamedTuple):
    obs: jax.Array        # (T, B, N, 18) what each agent saw
    pos: jax.Array        # (T, B, N, 2)  where each agent was
    vel: jax.Array        # (T, B, N, 2)
    action: jax.Array     # (T, B, N)
    logp: jax.Array       # (T, B, N)     log-probability of the chosen action
    value: jax.Array      # (T, B, N)     critic's guess of future reward
    reward: jax.Array     # (T, B, N)     the team reward from the environment
    distance: jax.Array   # (T, B)        total landmark gap after the step
    collisions: jax.Array  # (T, B, N)


def episode(team, key, num_envs, saboteur=None, greedy=False):
    """Returns (trajectory, final_state, landmarks, saboteur_mask).

    saboteur_mask has shape (B, N): True for the agent the saboteur controls.
    With no saboteur it is all False.
    """
    k_reset, k_who, k_act = jax.random.split(key, 3)
    state = env.reset(k_reset, num_envs)
    if saboteur is None:
        mask = jnp.zeros((num_envs, env.N_AGENTS), dtype=bool)
    else:
        who = jax.random.randint(k_who, (num_envs,), 0, env.N_AGENTS)
        mask = jnp.arange(env.N_AGENTS)[None, :] == who[:, None]

    def step_fn(state, k):
        obs = env.observe(state)
        logits = logits_fn(team, obs)
        value = value_fn(team, obs)
        if saboteur is not None:
            logits = jnp.where(mask[..., None], logits_fn(saboteur, obs), logits)
            value = jnp.where(mask, value_fn(saboteur, obs), value)
        if greedy:
            action = jnp.argmax(logits, axis=-1)
        else:
            action = jax.random.categorical(k, logits)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, action[..., None], axis=-1)[..., 0]
        nxt, reward, info = env.step(state, action)
        out = Trajectory(obs, state.pos, state.vel, action, logp, value, reward,
                         -info["global_reward"], info["collisions"])
        return nxt, out

    keys = jax.random.split(k_act, env.MAX_CYCLES)
    final, traj = jax.lax.scan(step_fn, state, keys)
    return traj, final, state.landmarks, mask


def summary(traj, final, landmarks):
    """A few numbers that describe how the team did, averaged over worlds."""
    gaps = env.landmark_distances(final.pos, landmarks).min(axis=1)        # (B, L)
    return {
        "return": jnp.mean(jnp.sum(traj.reward, axis=0)),
        "final_distance": jnp.mean(gaps.sum(axis=-1)),
        "covered": jnp.mean(jnp.sum(gaps < env.OCCUPIED_RADIUS, axis=-1)),
        "collisions": jnp.mean(jnp.sum(traj.collisions, axis=0)),
    }


def collect(team, saboteur, key, num_worlds, greedy=False):
    """Episodes with one hidden saboteur per world: (traj, landmarks, labels (B, N))."""
    traj, final, landmarks, mask = jax.jit(episode, static_argnums=(2, 4))(
        team, key, num_worlds, saboteur, greedy)
    return traj, landmarks, mask.astype(jnp.float32)


def damage(team, saboteur, key, num_worlds=2048):
    """How much worse the team ends up with the saboteur on it, on the very same worlds."""
    play = jax.jit(episode, static_argnums=(2, 4))
    honest = summary(*play(team, key, num_worlds, None, False)[:3])
    sabotaged = summary(*play(team, key, num_worlds, saboteur, False)[:3])
    return float(sabotaged["final_distance"] - honest["final_distance"])
