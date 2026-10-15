import jax
import jax.numpy as jnp
import numpy as np

from underwatch import env, monitors, ppo, rollout
from underwatch.nets import actor_critic_init


def _team(seed=0):
    return actor_critic_init(jax.random.PRNGKey(seed), env.OBS_DIM, env.N_ACTIONS)


def test_gae_matches_reference_loop():
    rng = np.random.default_rng(0)
    rewards, values = rng.normal(size=(25, 4, 3)), rng.normal(size=(25, 4, 3))
    adv, ret = ppo.gae(jnp.asarray(rewards), jnp.asarray(values), 0.99, 0.95)
    expected, running = np.zeros_like(rewards), np.zeros(rewards.shape[1:])
    for t in reversed(range(25)):
        next_value = values[t + 1] if t + 1 < 25 else 0.0
        running = rewards[t] + 0.99 * next_value - values[t] + 0.99 * 0.95 * running
        expected[t] = running
    np.testing.assert_allclose(np.asarray(adv), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ret), expected + values, rtol=1e-5, atol=1e-5)


def test_episode_shapes_and_one_saboteur_per_world():
    traj, final, landmarks, mask = rollout.episode(_team(), jax.random.PRNGKey(1), 8, _team(1))
    assert traj.obs.shape == (env.MAX_CYCLES, 8, env.N_AGENTS, env.OBS_DIM)
    assert traj.distance.shape == (env.MAX_CYCLES, 8)
    assert (np.asarray(mask).sum(axis=1) == 1).all()


def test_pick_saboteur_selects_the_masked_agent():
    x = jnp.arange(2 * 4 * 3 * 5.0).reshape(2, 4, 3, 5)
    mask = jnp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool)
    picked = np.asarray(ppo.pick_saboteur(x, mask))
    for b, seat in enumerate([0, 2, 1, 2]):
        np.testing.assert_array_equal(picked[:, b], np.asarray(x)[:, b, seat])


def test_a_saboteur_identical_to_the_team_does_no_damage():
    team = _team()
    assert rollout.damage(team, team, jax.random.PRNGKey(3), num_worlds=64) == 0.0


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    scores, labels = rng.normal(size=300), rng.random(300) < 0.3
    pos, neg = scores[labels], scores[~labels]
    expected = np.mean(pos[:, None] > neg[None, :])
    assert abs(monitors.auroc(scores, labels) - expected) < 1e-9


def test_monitors_score_every_agent():
    traj, _, landmarks, _ = rollout.episode(_team(), jax.random.PRNGKey(2), 4, _team(1))
    for kind, (init_fn, apply_fn) in monitors.MONITORS.items():
        logits = apply_fn(init_fn(jax.random.PRNGKey(0)), traj, landmarks)
        assert logits.shape == (4, env.N_AGENTS), kind
        assert bool(jnp.isfinite(logits).all()), kind


def test_graph_relations_partition_edges_by_node_type():
    traj, _, landmarks, _ = rollout.episode(_team(), jax.random.PRNGKey(4), 2)
    _, adj = monitors.graph_inputs(traj, landmarks)
    total = np.asarray(adj.sum(axis=2))[0, 0]          # (6, 6): number of relations per pair
    n = env.N_AGENTS
    expected = np.zeros((6, 6))
    expected[:n, :] = 1                                   # every node sends into every agent
    expected[n:, :n] = 1                                  # agents send into landmarks
    np.fill_diagonal(expected, 0)
    np.testing.assert_array_equal(total, expected)
