import jax.numpy as jnp

from drax.systems.block_push import BlockPush


def test_ocp() -> None:
    """Make sure we can construct the problem."""
    prob = BlockPush(horizon=10)

    # Test forward dynamics
    x = jnp.zeros(10)
    u = jnp.array([0.1, 0.7])

    for _ in range(10):
        x = prob.dynamics(x, u)
    assert x[1] > 0.0  # make sure the block moved

    # Test running cost
    ell = prob.running_cost(x, u)
    assert ell.shape == ()
    assert ell > 0.0

    # Test terminal cost
    phi = prob.terminal_cost(x)
    assert phi.shape == ()
    assert phi > 0.0

    # Generic NLP methods
    vars = jnp.zeros(prob.num_vars) + 1.23
    cost = prob.objective(vars)
    assert cost.shape == ()
    assert cost > 0.0

    res = prob.residual(vars)
    assert res.shape == (10 * 9,)
    assert jnp.all(res != 0.0)


if __name__ == "__main__":
    test_ocp()
