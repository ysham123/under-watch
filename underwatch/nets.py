"""Small neural networks written by hand: a list of layers, each a dict with "w" and "b"."""

import jax
import jax.numpy as jnp


def dense_init(key, n_in, n_out, scale):
    """One fully connected layer. Orthogonal weights keep signals a sensible size."""
    w = jax.nn.initializers.orthogonal(scale)(key, (n_in, n_out))
    return {"w": w, "b": jnp.zeros((n_out,))}


def mlp_init(key, sizes, out_scale=1.0):
    """sizes = [inputs, hidden, ..., outputs]. Returns a list of layers."""
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        scale = out_scale if last else jnp.sqrt(2.0)
        layers.append(dense_init(keys[i], n_in, n_out, scale))
    return layers


def mlp_apply(layers, x):
    """tanh between layers, no activation on the output."""
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    last = layers[-1]
    return x @ last["w"] + last["b"]


def actor_critic_init(key, in_dim, n_actions, hidden=64):
    """Two separate networks: the actor picks actions, the critic predicts future reward."""
    k_actor, k_critic = jax.random.split(key)
    return {
        "actor": mlp_init(k_actor, [in_dim, hidden, hidden, n_actions], out_scale=0.01),
        "critic": mlp_init(k_critic, [in_dim, hidden, hidden, 1], out_scale=1.0),
    }


def logits_fn(params, x):
    return mlp_apply(params["actor"], x)


def value_fn(params, x):
    return mlp_apply(params["critic"], x)[..., 0]
