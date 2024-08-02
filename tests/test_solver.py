import jax.numpy as jnp
import pytest

from drax.solver import SolverOptions, solve
from drax.systems.pendulum import PendulumSwingup


def test_solve() -> None:
    """Test the basic mechanics of the solver."""
    prob = PendulumSwingup(horizon=20, x_init=jnp.array([3.1, 0.0]))
    guess = jnp.zeros(prob.num_vars)

    options = SolverOptions(num_iters=100, print_every=10)
    data1 = solve(prob, options, guess)
    assert data1.x.shape == (prob.num_vars,)
    assert data1.lmbda.shape == data1.h.shape
    assert jnp.mean(jnp.square(data1.h)) < 0.1
    assert data1.k == options.num_iters

    options = SolverOptions(num_iters=100, print_every=1000)
    data2 = solve(prob, options, guess)
    assert jnp.allclose(data1.x, data2.x, atol=1e-4)
    assert jnp.allclose(data1.lmbda, data2.lmbda, atol=1e-4)

    with pytest.raises(ValueError):
        options = SolverOptions(gradient_method="not_a_valid_method")
        solve(prob, options, guess)

    options = SolverOptions(num_iters=100, gradient_method="sampling")
    data3 = solve(prob, options, guess)
    assert jnp.allclose(data1.f, data3.f, atol=0.1)


if __name__ == "__main__":
    test_solve()
