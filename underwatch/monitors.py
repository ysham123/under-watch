"""Two monitors that read a finished episode and score every agent for suspicion.

Both return logits of shape (B, N): one number per agent per world. A higher number
means "more likely to be the saboteur". sigmoid(logit) turns it into a probability.

  * MLP monitor:  looks at one agent at a time, its own 25 observations flattened.
  * RGCN monitor: looks at the whole scene as a graph of agents and landmarks with typed
                  edges, then reads off one score per agent.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from underwatch import env
from underwatch.nets import dense_init, mlp_apply, mlp_init

T, N, L = env.MAX_CYCLES, env.N_AGENTS, env.N_LANDMARKS


# ----------------------------------------------------------------------------- MLP monitor

def mlp_monitor_init(key, hidden=128):
    return mlp_init(key, [T * env.OBS_DIM, hidden, hidden, 1])


def mlp_monitor_apply(params, traj, landmarks):
    x = jnp.transpose(traj.obs, (1, 2, 0, 3))          # (T, B, N, 18) -> (B, N, T, 18)
    x = x.reshape(x.shape[0], N, T * env.OBS_DIM)       # one long row per agent
    return mlp_apply(params, x)[..., 0]                 # (B, N)


# ----------------------------------------------------------------------------- training

def bce_loss(params, apply_fn, traj, landmarks, labels):
    logits = apply_fn(params, traj, landmarks)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))


def take_worlds(traj, idx):
    """Select worlds (axis 1 of every trajectory field) by index."""
    return jax.tree_util.tree_map(lambda x: x[:, idx], traj)


def train_monitor(kind, data, key, epochs=30, batch=256, lr=1e-3, log=print):
    """data = (traj, landmarks, labels) with labels (B, N) in {0, 1}."""
    init_fn, apply_fn = MONITORS[kind]
    traj, landmarks, labels = data
    k_init, key = jax.random.split(key)
    params = init_fn(k_init)
    tx = optax.adam(lr)
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, idx):
        loss, grads = jax.value_and_grad(bce_loss)(
            params, apply_fn, take_worlds(traj, idx), landmarks[idx], labels[idx])
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    n_worlds = landmarks.shape[0]
    for epoch in range(epochs):
        key, k = jax.random.split(key)
        perm = jax.random.permutation(k, n_worlds)
        losses = []
        for start in range(0, n_worlds - batch + 1, batch):
            params, opt_state, loss = step(params, opt_state, perm[start:start + batch])
            losses.append(float(loss))
        if log and (epoch % 10 == 0 or epoch == epochs - 1):
            log(f"  {kind} epoch {epoch:3d}  loss {np.mean(losses):.4f}")
    return params


# ----------------------------------------------------------------------------- scoring

def auroc(scores, labels):
    """Probability that a random saboteur scores higher than a random honest agent."""
    scores, labels = np.asarray(scores).ravel(), np.asarray(labels).ravel().astype(bool)
    pos, neg = scores[labels], scores[~labels]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate_monitor(kind, params, data):
    _, apply_fn = MONITORS[kind]
    traj, landmarks, labels = data
    scores = np.asarray(jax.jit(apply_fn)(params, traj, landmarks))
    labels = np.asarray(labels)
    caught = np.mean(scores.argmax(axis=1) == labels.argmax(axis=1))
    return {"auroc": float(auroc(scores, labels)), "caught": float(caught)}


# ----------------------------------------------------------------------------- RGCN monitor
#
# Nodes 0, 1, 2 are agents. Nodes 3, 4, 5 are landmarks.
# Node features (6 numbers): position (2), velocity (2), is_agent, is_landmark.
# Relations, written "source -> target", each with its own weight matrix:
#   0 agent -> agent, far        1 agent -> agent, near (touching distance)
#   2 landmark -> agent, far     3 landmark -> agent, near (agent is on the landmark)
#   4 agent -> landmark, far     5 agent -> landmark, near
NODE_DIM = 6
N_REL = 6
NEAR_AGENT = 2 * env.AGENT_SIZE + 0.1
NEAR_LANDMARK = 0.2


def graph_inputs(traj, landmarks):
    """Node features (B, T, 6, 6) and adjacency (B, T, R, 6, 6) with adj[..., r, i, j] = 1
    when there is an edge of relation r from node j into node i."""
    B = landmarks.shape[0]
    pos = jnp.transpose(traj.pos, (1, 0, 2, 3))                    # (B, T, N, 2)
    vel = jnp.transpose(traj.vel, (1, 0, 2, 3))
    lm = jnp.broadcast_to(landmarks[:, None], (B, T, L, 2))
    ones, zeros = jnp.ones((B, T, N, 1)), jnp.zeros((B, T, N, 1))
    agent_feat = jnp.concatenate([pos, vel, ones, zeros], -1)
    lm_feat = jnp.concatenate([lm, jnp.zeros((B, T, L, 2)), jnp.zeros((B, T, L, 1)),
                               jnp.ones((B, T, L, 1))], -1)
    nodes = jnp.concatenate([agent_feat, lm_feat], axis=2)          # (B, T, 6, 6)

    everything = jnp.concatenate([pos, lm], axis=2)                 # (B, T, 6, 2)
    gap = everything[:, :, :, None, :] - everything[:, :, None, :, :]
    dist = jnp.sqrt(jnp.sum(gap ** 2, axis=-1) + 1e-12)            # (B, T, 6, 6)

    is_agent = jnp.arange(N + L) < N
    a_to_a = is_agent[:, None] & is_agent[None, :] & ~jnp.eye(N + L, dtype=bool)
    l_to_a = is_agent[:, None] & ~is_agent[None, :]     # into an agent, from a landmark
    a_to_l = ~is_agent[:, None] & is_agent[None, :]
    near_a = dist < NEAR_AGENT
    near_l = dist < NEAR_LANDMARK
    adj = jnp.stack([a_to_a & ~near_a, a_to_a & near_a,
                     l_to_a & ~near_l, l_to_a & near_l,
                     a_to_l & ~near_l, a_to_l & near_l], axis=2)    # (B, T, R, 6, 6)
    return nodes, adj.astype(jnp.float32)


def rgcn_layer_init(key, d_in, d_out):
    k_self, k_rel = jax.random.split(key)
    w_rel = jax.nn.initializers.glorot_uniform()(k_rel, (N_REL, d_in, d_out))
    return {"self": dense_init(k_self, d_in, d_out, 1.0), "rel": w_rel}


def rgcn_layer(layer, h, adj):
    """h' = tanh( W_self h_i + b + sum_r (1/c_ir) sum_{j in N_r(i)} W_r h_j )."""
    count = jnp.maximum(adj.sum(axis=-1, keepdims=True), 1.0)       # c_ir, (B, T, R, 6, 1)
    # mean of the neighbours' features, separately for each relation
    messages = jnp.einsum("btrij,btjd->btrid", adj / count, h)
    messages = jnp.einsum("btrid,rde->btie", messages, layer["rel"])
    own = h @ layer["self"]["w"] + layer["self"]["b"]
    return jnp.tanh(own + messages)


def rgcn_monitor_init(key, hidden=32):
    k1, k2, k3 = jax.random.split(key, 3)
    layers = [rgcn_layer_init(k1, NODE_DIM, hidden), rgcn_layer_init(k2, hidden, hidden)]
    return {"layers": layers, "head": mlp_init(k3, [2 * hidden, hidden, 1])}


def rgcn_monitor_apply(params, traj, landmarks):
    h, adj = graph_inputs(traj, landmarks)
    for layer in params["layers"]:
        h = rgcn_layer(layer, h, adj)
    agents = h[:, :, :N]                                           # (B, T, N, hidden)
    # pool over time: the typical value (mean) and the most extreme one (max)
    pooled = jnp.concatenate([agents.mean(axis=1), agents.max(axis=1)], axis=-1)
    return mlp_apply(params["head"], pooled)[..., 0]               # (B, N)


# ----------------------------------------------------------------------------- registry

MONITORS = {
    "mlp": (mlp_monitor_init, mlp_monitor_apply),
    "rgcn": (rgcn_monitor_init, rgcn_monitor_apply),
}
