from datetime import datetime
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from drax.nlp import NonlinearProgram


class SolverOptions(NamedTuple):
    """Object holding optimizer parameters.

    Parameters:
        num_iters: The total number of iterations to run.
        print_every: How many iterations to wait between printing status.
        alpha: The step size.
        mu: The augmented Lagrangian penalty parameter.
        rho: The log barrier parameter for bound constraints.
    """

    num_iters: int = 5000
    print_every: int = 500
    alpha: float = 0.01
    mu: float = 10.0
    rho: float = 0.1


class SolverData(NamedTuple):
    """Object holding iteration/solution data.

    Parameters:
        k: The current iteration number.
        x: The decision variables at the current iteration.
        lmbda: The Lagrange multipliers at the current iteration.
        f: The cost at the current iteration.
        h: The constraint residuals at the current iteration.
        grad: The gradient of the Lagrangian w.r.t. x at the current iteration.
        lagrangian: The augmented Lagrangian at the current iteration.
        rng: The random number generator state.

    """

    k: int
    x: jnp.ndarray
    lmbda: jnp.ndarray
    f: jnp.ndarray
    h: jnp.ndarray
    grad: jnp.ndarray
    lagrangian: jnp.ndarray
    rng: jnp.ndarray


def _calc_lagrangian(
    x: jnp.ndarray,
    lmbda: jnp.ndarray,
    prob: NonlinearProgram,
    options: SolverOptions,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the augmented Lagrangian, including log barrier terms.

    Args:
        x: The decision variables.
        lmbda: The Lagrange multipliers.
        prob: The nonlinear program to solve.
        options: The optimizer parameters.

    Returns:
        The augmented Lagrangian, L(x, λ) = f(x) + λᵀh(x) + μ/2 h(x)².
        The cost f(x) + log barrier terms
        The equality constraint residuals h(x)
    """
    f = prob.objective(x)
    h = prob.residual(x)

    # Add log barrier terms to the cost to enforce l <= x <= u.
    u_err = jnp.maximum(prob.upper - x[prob.bounded_above], 1e-4)
    l_err = jnp.maximum(x[prob.bounded_below] - prob.lower, 1e-4)
    f -= jnp.sum(options.rho * jnp.log(u_err))
    f -= jnp.sum(options.rho * jnp.log(l_err))

    L = f + lmbda.T @ h + 0.5 * options.mu * h.T @ h
    return L, (f, h)


def _calc_gradient_data_autodiff(
    data: SolverData, prob: NonlinearProgram, options: SolverOptions
) -> SolverData:
    """Compute the current cost, constraints, and lagrangian, and gradient.

    Use JAX autodiff to compute the gradient of the Lagrangian.

    Args:
        data: The current iteration data.
        prob: The nonlinear program to solve.
        options: The optimizer parameters.

    Returns:
        A SolverData object with updated f, h, grad, and lagrangian.
    """
    grad_fn = jax.value_and_grad(
        lambda x: _calc_lagrangian(x, data.lmbda, prob, options), has_aux=True
    )

    (L, (f, h)), grad = grad_fn(data.x)
    return data._replace(f=f, h=h, grad=grad, lagrangian=L)


def _calc_gradient_data_sampling(
    data: SolverData, prob: NonlinearProgram, options: SolverOptions
) -> SolverData:
    """Compute the current cost, constraints, and lagrangian, and gradient.

    Use a zero-order sampling approximation for the gradient of the Lagrangian.

    Args:
        data: The current iteration data.
        prob: The nonlinear program to solve.
        options: The optimizer parameters.

    Returns:
        A SolverData object with updated f, h, grad, and lagrangian.
    """
    lagrangian, (f, h) = _calc_lagrangian(data.x, data.lmbda, prob, options)

    # TODO: make parameters
    sigma = 0.01
    num_rollouts = 512

    rng, noise_rng = jax.random.split(data.rng)
    noise = sigma * jax.random.normal(noise_rng, (num_rollouts, prob.num_vars))

    X = data.x + noise
    L, _ = jax.vmap(_calc_lagrangian, in_axes=(0, None, None, None))(
        X, data.lmbda, prob, options
    )

    grad = jnp.einsum("i,ij->j", L - lagrangian, noise)
    grad /= num_rollouts * sigma**2

    return data._replace(f=f, h=h, grad=grad, lagrangian=lagrangian, rng=rng)


def _optimizer_step(
    data: SolverData, prob: NonlinearProgram, options: SolverOptions
) -> SolverData:
    """Take a single optimizer step.

    Args:
        data: The current iteration data.
        prob: The nonlinear program to solve.
        options: The optimizer parameters.

    Returns:
        The updated iteration data.
    """
    # Compute the current cost, constraints, Lagrangian, and gradient.
    # data = _calc_gradient_data_autodiff(data, prob, options)
    data = _calc_gradient_data_sampling(data, prob, options)

    # Flow the decision variable and Lagrange multiplier according to
    #   ẋ = -∂L/∂x,
    #   λ̇ = ∂L/∂λ
    # See Platt and Barr, "Constrained Differential Optimization",
    # NeurIPS 1987 for more details.
    x = data.x - options.alpha * data.grad
    lmbda = data.lmbda + options.alpha * options.mu * data.h

    # Clip to the feasible region. This should be a no-op if the log barrier
    # is working, but that sometimes needs to be relaxed.
    x = x.at[prob.bounded_below].set(
        jnp.maximum(x[prob.bounded_below], prob.lower)
    )
    x = x.at[prob.bounded_above].set(
        jnp.minimum(x[prob.bounded_above], prob.upper)
    )

    return data._replace(k=data.k + 1, x=x, lmbda=lmbda)


def solve(
    prob: NonlinearProgram, options: SolverOptions, guess: jnp.ndarray
) -> SolverData:
    """Solve the nonlinear optimization problem.

    Args:
        prob: The nonlinear program to solve.
        options: The optimizer parameters.
        guess: An initial guess for the decision variables.

    Returns:
        The solution, including decision variables and other data.
    """
    # Initialize the solver data
    h = jax.jit(prob.residual)(guess)  # TODO: define prob.num_eq_cons
    data = SolverData(
        k=0,
        x=guess,
        lmbda=jnp.zeros_like(h),
        f=0.0,
        h=h,
        grad=jnp.zeros_like(guess),
        lagrangian=0.0,
        rng=jax.random.key(0),
    )

    # Determine how many times to print status, and how many iterations to run
    # between each print.
    print_every = min(options.num_iters, options.print_every)
    num_prints = options.num_iters // print_every

    # Update function takes runs N iterations before printing status
    scan_fn = lambda data, _: (_optimizer_step(data, prob, options), None)
    update_fn = jax.jit(
        lambda data: jax.lax.scan(scan_fn, data, jnp.arange(print_every))[0]
    )

    start_time = datetime.now()
    for _ in range(num_prints):
        # Do a bunch of iterations
        data = update_fn(data)

        # Print status
        cons_viol = jnp.mean(jnp.square(data.h))
        grad_norm = jnp.linalg.norm(data.grad)
        print(
            f"Iter {data.k}: cost = {data.f:.4f}, cons = {cons_viol:.4f}, "
            f"lagrangian = {data.lagrangian:.4f}, grad = {grad_norm:.4f}, "
            f"time = {datetime.now() - start_time}"
        )

    return data
