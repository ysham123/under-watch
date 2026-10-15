"""PPO for the honest team and for the saboteur.

The learning rule is the same for both. The only difference is which samples go in and
what reward they carry:
  * team:      all three agents, team reward from the environment
  * saboteur:  only the saboteur's own column, reward = how badly the team is doing,
               minus a penalty if a monitor flags it at the end of the episode
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from underwatch import env
from underwatch.nets import actor_critic_init, logits_fn, value_fn
from underwatch.rollout import episode, summary


class Config(NamedTuple):
    num_envs: int = 64
    lr: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 10
    minibatches: int = 4
    max_grad_norm: float = 0.5


def gae(rewards, values, gamma, lam):
    """Generalized advantage estimation. Time is axis 0; the episode ends after
    the last step."""
    next_values = jnp.concatenate([values[1:], jnp.zeros_like(values[:1])], axis=0)
    deltas = rewards + gamma * next_values - values

    def back(carry, delta):
        adv = delta + gamma * lam * carry
        return adv, adv

    _, adv = jax.lax.scan(back, jnp.zeros_like(deltas[0]), deltas, reverse=True)
    return adv, adv + values


def ppo_loss(params, batch, cfg):
    obs, action, old_logp, adv, ret = batch
    logp_all = jax.nn.log_softmax(logits_fn(params, obs))
    logp = jnp.take_along_axis(logp_all, action[..., None], axis=-1)[..., 0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    ratio = jnp.exp(logp - old_logp)
    clipped = jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)
    policy_loss = -jnp.mean(jnp.minimum(ratio * adv, clipped * adv))
    value_loss = 0.5 * jnp.mean((value_fn(params, obs) - ret) ** 2)
    entropy = -jnp.mean(jnp.sum(jnp.exp(logp_all) * logp_all, axis=-1))
    return policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy


def make_optimizer(cfg):
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr))


def ppo_epochs(params, opt_state, data, key, cfg):
    """Several passes over one batch of experience, in shuffled minibatches."""
    tx = make_optimizer(cfg)
    n = data[0].shape[0]
    mb = n // cfg.minibatches

    def one_epoch(carry, k):
        perm = jax.random.permutation(k, n)[: mb * cfg.minibatches]
        shuffled = [x[perm].reshape((cfg.minibatches, mb) + x.shape[1:]) for x in data]

        def one_minibatch(carry, batch):
            params, opt_state = carry
            grads = jax.grad(ppo_loss)(params, batch, cfg)
            updates, opt_state = tx.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), None

        carry, _ = jax.lax.scan(one_minibatch, carry, shuffled)
        return carry, None

    (params, opt_state), _ = jax.lax.scan(one_epoch, (params, opt_state),
                                          jax.random.split(key, cfg.epochs))
    return params, opt_state


# ----------------------------------------------------------------------------- honest team

def init_team(key):
    return actor_critic_init(key, env.OBS_DIM, env.N_ACTIONS)


def make_team_update(cfg):
    @jax.jit
    def update(params, opt_state, key):
        k_roll, k_train = jax.random.split(key)
        traj, final, landmarks, _ = episode(params, k_roll, cfg.num_envs)
        adv, ret = gae(traj.reward, traj.value, cfg.gamma, cfg.gae_lambda)
        data = [x.reshape((-1,) + x.shape[3:])            # (T, B, N, ...) -> (T*B*N, ...)
                for x in (traj.obs, traj.action, traj.logp, adv, ret)]
        params, opt_state = ppo_epochs(params, opt_state, data, k_train, cfg)
        return params, opt_state, summary(traj, final, landmarks)

    return update


def run(update, params, key, total_steps, cfg, every=200, log=print):
    """Call update() enough times to use total_steps environment steps."""
    opt_state = make_optimizer(cfg).init(params)
    num_updates = total_steps // (cfg.num_envs * env.MAX_CYCLES)
    history = []
    for i in range(num_updates):
        key, k = jax.random.split(key)
        params, opt_state, stats = update(params, opt_state, k)
        history.append({name: float(v) for name, v in stats.items()})
        if log and (i % every == 0 or i == num_updates - 1):
            numbers = "  ".join(f"{n} {v:6.2f}" for n, v in history[-1].items())
            log(f"  update {i:5d}  {numbers}")
    return params, history


# ----------------------------------------------------------------------------- saboteur

def pick_saboteur(x, mask):
    """From (T, B, N, ...) keep only the saboteur's column, giving (T, B, ...)."""
    idx = jnp.argmax(mask, axis=-1)              # (B,) which agent is the saboteur
    idx = idx.reshape((1, -1, 1) + (1,) * (x.ndim - 3))
    return jnp.take_along_axis(x, idx, axis=2)[:, :, 0]


def make_saboteur_update(cfg, team, monitor=None, lam=0.0):
    """monitor(traj, landmarks) -> (B, N) suspicion logits. Its probability for the
    saboteur, times lam, is subtracted from the saboteur's reward on the final step."""

    @jax.jit
    def update(params, opt_state, key):
        k_roll, k_train = jax.random.split(key)
        traj, final, landmarks, mask = episode(team, k_roll, cfg.num_envs, saboteur=params)
        reward = traj.distance                   # (T, B): the saboteur wants it big
        suspicion = jnp.zeros(cfg.num_envs)
        if monitor is not None:
            probs = jax.nn.sigmoid(monitor(traj, landmarks))         # (B, N)
            suspicion = jnp.sum(probs * mask, axis=-1)               # (B,)
            reward = reward.at[-1].add(-lam * suspicion)
        value = pick_saboteur(traj.value, mask)
        adv, ret = gae(reward, value, cfg.gamma, cfg.gae_lambda)
        data = [x.reshape((-1,) + x.shape[2:]) for x in (
            pick_saboteur(traj.obs, mask), pick_saboteur(traj.action, mask),
            pick_saboteur(traj.logp, mask), adv, ret)]
        params, opt_state = ppo_epochs(params, opt_state, data, k_train, cfg)
        stats = summary(traj, final, landmarks)
        stats["suspicion"] = suspicion.mean()
        return params, opt_state, stats

    return update
