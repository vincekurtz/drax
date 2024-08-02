import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from drax.nlp import NonlinearProgram
from drax.ocp import OptimalControlProblem
from drax.solver import SolverOptions, solve
from drax.systems.pendulum import PendulumSwingup


def test_ocp() -> None:
    """Make sure we can construct the optimal control problem."""
    prob = PendulumSwingup(horizon=10, x_init=jnp.array([0.0, 0.0]))

    assert isinstance(prob, PendulumSwingup)
    assert isinstance(prob, OptimalControlProblem)
    assert isinstance(prob, NonlinearProgram)

    x = jnp.array([0.1, 0.2])
    u = jnp.array([0.3])

    # OCP-specific methods
    x_next = prob.dynamics(x, u)
    assert x_next.shape == x.shape
    ell = prob.running_cost(x, u)
    assert ell.shape == ()
    phi = prob.terminal_cost(x)
    assert phi.shape == ()
    g = prob.constraints(x, u)
    assert g.shape == (0,)

    # Generic NLP methods
    assert prob.nx == 2
    assert prob.nu == 1
    assert prob.num_vars == 27  # (horizon - 1) * 3

    vars = jnp.zeros(prob.num_vars) + 1.23

    cost = prob.objective(vars)
    assert cost.shape == ()
    assert cost > 0.0

    res = prob.residual(vars)
    assert res.shape == (2 * 9,)  # 2 * (horizon - 1)
    assert jnp.all(res != 0.0)


def test_plot() -> None:
    """Make sure we can plot the pendulum scenario."""
    prob = PendulumSwingup(horizon=10, x_init=jnp.array([0.0, 0.0]))
    prob.plot_scenario()
    if __name__ == "__main__":
        plt.show()


def test_solve() -> None:
    """Make sure we can solve the pendulum swingup problem."""
    prob = PendulumSwingup(horizon=50, x_init=jnp.array([3.1, 0.0]))

    rng = jax.random.key(0)
    guess = jax.random.uniform(rng, (prob.num_vars,), minval=-0.9, maxval=0.9)
    options = SolverOptions(num_iters=20000, print_every=5000)

    data = solve(prob, options, guess)
    sol = data.x
    assert sol.shape == (prob.num_vars,)

    xs, us = prob.unflatten(sol)
    if __name__ == "__main__":
        plt.subplot(2, 1, 1)
        prob.plot_scenario()
        plt.plot(xs[:, 0], xs[:, 1], "ro-")

        plt.subplot(2, 1, 2)
        plt.plot(us, "bo-")

        plt.show()


if __name__ == "__main__":
    # test_ocp()
    # test_plot()
    test_solve()
