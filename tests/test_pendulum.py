import jax.numpy as jnp

from drax.nlp import NonlinearProgram
from drax.ocp import OptimalControlProblem
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


if __name__ == "__main__":
    test_ocp()
