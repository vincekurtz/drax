import jax
import jax.numpy as jnp

from drax.solver import SolverOptions, solve_verbose
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


def test_vis() -> None:
    """Make sure we can visualize a trajectory."""
    horizon = 20
    prob = BlockPush(horizon=horizon)
    xs = jnp.zeros((horizon, prob.nx))
    xs = xs.at[:, 1].set(jnp.linspace(0.0, 0.5, horizon))
    us = jnp.zeros((horizon, prob.nu))
    if __name__ == "__main__":
        prob.visualize_trajectory(xs, us)


def test_solve() -> None:
    """Try solving a small problem instance."""
    prob = BlockPush(horizon=10)

    rng = jax.random.key(0)
    rng, guess_rng = jax.random.split(rng)
    guess = jax.random.uniform(guess_rng, (prob.num_vars,))
    options = SolverOptions(
        num_iters=20,
        method="diffusion",
        gradient_method="sampling",
    )

    data = solve_verbose(prob, options, guess, print_every=1)
    sol = data.x
    assert sol.shape == (prob.num_vars,)

    xs, us = prob.unflatten(sol)
    print(xs.shape)
    print(xs)
    breakpoint()


if __name__ == "__main__":
    # test_ocp()
    test_vis()
    # test_solve()
