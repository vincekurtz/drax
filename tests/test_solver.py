import jax
import jax.numpy as jnp
import pytest

from drax.solver import (
    SolverData,
    SolverOptions,
    _calc_gradient_data_autodiff,
    _calc_gradient_data_sampling,
    make_warm_start,
    optimizer_step,
    solve,
    solve_verbose,
)
from drax.systems.pendulum import PendulumSwingup


def _is_same_solver_data(data1: SolverData, data2: SolverData) -> bool:
    """Check if two SolverData objects are the same."""
    return (
        data1.k == data2.k
        and jnp.allclose(data1.x, data2.x)
        and jnp.allclose(data1.lmbda, data2.lmbda)
        and jnp.allclose(data1.f, data2.f)
        and jnp.allclose(data1.h, data2.h)
        and jnp.allclose(data1.grad, data2.grad)
        and jnp.allclose(data1.lagrangian, data2.lagrangian)
    )


def test_solve() -> None:
    """Test the basic mechanics of the solver."""
    prob = PendulumSwingup(horizon=20, x_init=jnp.array([3.1, 0.0]))
    guess = jnp.zeros(prob.num_vars)
    warm_start = make_warm_start(prob, guess)
    options = SolverOptions(num_iters=100)

    # Run the standard solve
    sol1 = solve(prob, options, warm_start)
    assert sol1.x.shape == (prob.num_vars,)
    assert sol1.lmbda.shape == sol1.h.shape
    assert jnp.mean(jnp.square(sol1.h)) < 0.1
    assert sol1.k == options.num_iters

    # Solve with print statements
    sol2 = solve_verbose(prob, options, guess, print_every=10)
    assert _is_same_solver_data(sol1, sol2)

    # Step through the solve manually
    jit_step = jax.jit(optimizer_step, static_argnums=(1, 2))
    sol3 = warm_start
    for _ in range(options.num_iters):
        sol3 = jit_step(sol3, prob, options)
    assert _is_same_solver_data(sol1, sol3)

    # Try with a bad gradient method
    with pytest.raises(ValueError):
        options = SolverOptions(gradient_method="not_a_valid_method")
        solve(prob, options, guess)

    # Run with sampling-based gradients
    options = SolverOptions(num_iters=100, gradient_method="sampling")
    sol4 = solve(prob, options, warm_start)
    assert jnp.allclose(sol1.f, sol4.f, atol=0.1)


def test_sampling_gradient() -> None:
    """Test our sampling-based gradient approximation."""
    prob = PendulumSwingup(horizon=10, x_init=jnp.array([3.1, 0.0]))
    guess = jnp.zeros(prob.num_vars)
    data = make_warm_start(prob, guess)

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
