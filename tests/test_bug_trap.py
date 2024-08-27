import jax.numpy as jnp
import matplotlib.pyplot as plt

from drax.solver import SolverOptions, solve_verbose
from drax.systems.bug_trap import BugTrap


def test_setup() -> None:
    """Test that we can construct the problem."""
    prob = BugTrap(horizon=10, x_init=jnp.array([-1.0, 0.0, 0.0]))

    x = jnp.array([0.1, 0.2, 0.3])
    u = jnp.array([0.4, 0.5])

    x_next = prob.dynamics(x, u)
    assert x_next.shape == x.shape

    ell = prob.running_cost(x, u)
    assert ell.shape == ()

    phi = prob.terminal_cost(x)
    assert phi.shape == ()

    vars = jnp.zeros(prob.num_vars) + 1.23
    cost = prob.objective(vars)
    assert cost.shape == ()
    assert cost > 0.0

    res = prob.residual(vars)
    assert res.shape == (3 * 9,)  # 3 * (horizon - 1)
    assert jnp.all(res != 0.0)


def test_plot() -> None:
    """Make sure we can plot the bug trap scenario."""
    prob = BugTrap(horizon=10, x_init=jnp.array([-1.0, 0.0, 0.0]))
    prob.plot_scenario()
    if __name__ == "__main__":
        plt.show()


def test_solve() -> None:
    """Make sure we can solve the bug trap problem."""
    prob = BugTrap(horizon=50, x_init=jnp.array([-1.0, 0.0, 0.0]))
    options = SolverOptions(
        num_iters=20000,
        method="diffusion",
        rho=0.01,
        initial_noise_level=4.0,
        mu=10.0,
        gradient_method="autodiff",
    )

    guess = jnp.zeros(prob.num_vars) + 0.1
    sol = solve_verbose(prob, options, guess)
    assert sol.x.shape == (prob.num_vars,)

    xs, us = prob.unflatten(sol.x)
    assert xs.shape == (prob.horizon, prob.nx)
    assert us.shape == (prob.horizon - 1, prob.nu)

    if __name__ == "__main__":
        plt.subplot(2, 1, 1)
        prob.plot_scenario()
        plt.plot(xs[:, 0], xs[:, 1], "o-")

        plt.subplot(2, 1, 2)
        plt.plot(us)
        plt.xlabel("Time step")
        plt.ylabel("Control")

        plt.show()


if __name__ == "__main__":
    test_setup()
    test_plot()
    test_solve()
