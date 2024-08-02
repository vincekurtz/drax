import jax
import jax.numpy as jnp
import pytest

from drax.solver import (
    SolverOptions,
    _calc_gradient_data_autodiff,
    _calc_gradient_data_sampling,
    _make_solver_data,
    _optimizer_step,
    solve,
    solve_from_warm_start,
)
from drax.systems.pendulum import PendulumSwingup


def test_solve() -> None:
    """Test the basic mechanics of the solver."""
    prob = PendulumSwingup(horizon=20, x_init=jnp.array([3.1, 0.0]))
    guess = jnp.zeros(prob.num_vars)

    # Run with lots of print statements
    options = SolverOptions(num_iters=100, print_every=10)
    data1 = solve(prob, options, guess)
    assert data1.x.shape == (prob.num_vars,)
    assert data1.lmbda.shape == data1.h.shape
    assert jnp.mean(jnp.square(data1.h)) < 0.1
    assert data1.k == options.num_iters

    # Run with only 1 print statment
    options = SolverOptions(num_iters=100, print_every=1000)
    data2 = solve(prob, options, guess)
    assert jnp.allclose(data1.x, data2.x, atol=1e-4)
    assert jnp.allclose(data1.lmbda, data2.lmbda, atol=1e-4)

    # Try with a bad gradient method
    with pytest.raises(ValueError):
        options = SolverOptions(gradient_method="not_a_valid_method")
        solve(prob, options, guess)

    # Run with sampling-based gradients
    options = SolverOptions(num_iters=100, gradient_method="sampling")
    data3 = solve(prob, options, guess)
    assert jnp.allclose(data1.f, data3.f, atol=0.1)

    # Run manually
    options = SolverOptions(gradient_method="autodiff")
    data4 = _make_solver_data(prob, guess)
    jit_step = jax.jit(_optimizer_step, static_argnums=(1, 2))
    for _ in range(100):
        data4 = jit_step(data4, prob, options)
    assert jnp.allclose(data1.x, data4.x, atol=1e-4)
    assert jnp.allclose(data1.lmbda, data4.lmbda, atol=1e-4)

    # Run a few times from a warm start
    options = SolverOptions(num_iters=20)
    data5 = _make_solver_data(prob, guess)
    for _ in range(5):
        data5 = solve_from_warm_start(prob, options, data5)
    assert jnp.allclose(data1.x, data5.x, atol=1e-4)
    assert jnp.allclose(data1.lmbda, data5.lmbda, atol=1e-4)


def test_sampling_gradient() -> None:
    """Test our sampling-based gradient approximation."""
    prob = PendulumSwingup(horizon=10, x_init=jnp.array([3.1, 0.0]))
    guess = jnp.zeros(prob.num_vars)
    data = _make_solver_data(prob, guess)

    # Compute the gradient using autodiff
    options = SolverOptions(gradient_method="autodiff")
    autodiff_data = jax.jit(
        _calc_gradient_data_autodiff, static_argnums=(1, 2)
    )(data, prob, options)

    # Compute the gradient using sampling
    options = SolverOptions(gradient_method="sampling", num_samples=256)
    sampling_data = jax.jit(
        _calc_gradient_data_sampling, static_argnums=(1, 2)
    )(data, prob, options)

    assert sampling_data.f == autodiff_data.f
    assert jnp.all(sampling_data.h == autodiff_data.h)
    assert sampling_data.lagrangian == autodiff_data.lagrangian
    assert sampling_data.grad.shape == autodiff_data.grad.shape

    err = jnp.mean(jnp.square(sampling_data.grad - autodiff_data.grad))
    assert err < 5.0

    # It should get more precise as we crank up the number of samples
    options = SolverOptions(gradient_method="sampling", num_samples=2048)
    sampling_data = jax.jit(
        _calc_gradient_data_sampling, static_argnums=(1, 2)
    )(data, prob, options)

    err = jnp.mean(jnp.square(sampling_data.grad - autodiff_data.grad))
    assert err < 1.0


if __name__ == "__main__":
    test_solve()
    test_sampling_gradient()
