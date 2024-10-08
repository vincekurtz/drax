from datetime import datetime
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from drax.nlp import NonlinearProgram


class SolverOptions(NamedTuple):
    """Object holding optimizer parameters.

    Parameters:
        method: Solver method to use. Must be one of:
            "gradient_descent" - standard vanilla gradient descent.
            "diffusion" - equality constrained Langevin diffusion.
            "bfgs" - BFGS quasi-Newton method.
        num_iters: The total number of iterations to run.
        alpha: The step size.
        mu: The augmented Lagrangian penalty parameter.
        rho: The log barrier parameter for bound constraints.
        gradient_method: How to compute gradients ("autodiff" or "sampling").
        sigma: Variance for sampling-based gradient estimation.
        num_samples: Number of samples for sampling-based gradient estimation.
        initial_noise_level: Initial noise level for the diffusion method.
        seed: Random number generator seed.
    """

    method: str = "gradient_descent"
    num_iters: int = 5000
    alpha: float = 0.01
    mu: float = 10.0
    rho: float = 0.1
    gradient_method: str = "autodiff"
    sigma: float = 0.01
    num_samples: int = 128
    initial_noise_level: float = 0.1
    seed: int = 0


class SolverData(NamedTuple):
    """Object holding iteration/solution data.

    Parameters:
        k: The current iteration number.
        x: The decision variables at the current iteration.
        lmbda: The Lagrange multipliers at the current iteration.
        f: The cost at the current iteration.
        h: The constraint residuals at the current iteration.
        H: The Hessian approximation at the current iteration.
        grad: The gradient of the Lagrangian w.r.t. x at the current iteration.
        lagrangian: The augmented Lagrangian at the current iteration.
        noise_decay_factor: Noise decay factor for the diffusion method. Noise
            level is computed as σₖ = σ₀ * exp(-noise_decay_factor * k).
        rng: The random number generator state.

    """

    k: int
    x: jnp.ndarray
    lmbda: jnp.ndarray
    f: jnp.ndarray
    h: jnp.ndarray
    grad: jnp.ndarray
    H: jnp.ndarray
    lagrangian: jnp.ndarray
    noise_decay_factor: float
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

    rng, noise_rng = jax.random.split(data.rng)
    noise = options.sigma * jax.random.normal(
        noise_rng, (options.num_samples, prob.num_vars)
    )

    X = data.x + noise
    L, _ = jax.vmap(_calc_lagrangian, in_axes=(0, None, None, None))(
        X, data.lmbda, prob, options
    )

    grad = jnp.einsum("i,ij->j", L - lagrangian, noise)
    grad /= options.num_samples * options.sigma**2

    return data._replace(f=f, h=h, grad=grad, lagrangian=lagrangian, rng=rng)


def _calc_update_gradient_descent(
    data: SolverData, prob: NonlinearProgram, options: SolverOptions
) -> SolverData:
    """Update decision variables and Lagrange multipliers via gradient descent.

        x = x - α ∂L/∂x,
        λ = λ + α μ ∂L/∂λ.

    Args:
        data: The current iteration data.
        prob: The nonlinear program to solve.
        options: The optimizer parameters.

    Returns:
        Updated iteration data with the new x and λ.
    """
    x = data.x - options.alpha * data.grad
    lmbda = data.lmbda + options.alpha * options.mu * data.h
    return data._replace(x=x, lmbda=lmbda)


def _calc_update_diffusion(
    data: SolverData, prob: NonlinearProgram, options: SolverOptions
) -> SolverData:
    """Update decision variables with a constrained diffusion process.

    Flows the decision variables according to the Langevin dynamics

            ẋ = -∂L/∂x + σₖξ,
            λ̇ = ∂L/∂λ.

    where ξ ~ N(0, I) is a standard normal random variable and σₖ is an annealed
    noise level.

    Args:
        data: The current iteration data.
        prob: The nonlinear program to solve.
        options: The optimizer parameters.

    Returns:
        Updated iteration data with the new x and λ.
    """
    rng, noise_rng = jax.random.split(data.rng)
    noise = jax.random.normal(noise_rng, data.x.shape)
    noise_level = options.initial_noise_level * jnp.exp(
        -data.k * data.noise_decay_factor
    )
    noise_level *= jnp.sqrt(2 * options.alpha)  # Euler-Maruyama discretization

    x = data.x - options.alpha * data.grad + noise_level * noise
    lmbda = data.lmbda + options.alpha * data.h

    return data._replace(x=x, lmbda=lmbda, rng=rng)


def _linesearch(
    x: jnp.ndarray,
    p: jnp.ndarray,
    lmbda: jnp.ndarray,
    prob: NonlinearProgram,
    options: SolverOptions,
) -> jnp.ndarray:
    """Perform a parallel line search on the Lagrangian.

    Approximately solves

        min_{α} L(x + αp, λ).

    Args:
        x: The current decision variables.
        p: The search direction.
        lmbda: The current Lagrange multipliers.
        prob: The nonlinear program to solve.
        options: The optimizer parameters.

    Returns:
        The step size α.
    """
    # TODO: choose candidates more intelligently
    # TODO: check convergence conditions
    candidates = jnp.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    L = lambda x: _calc_lagrangian(x, lmbda, prob, options)[0]
    costs = jax.vmap(L)(x + candidates[:, None] * p)
    return candidates[jnp.argmin(costs)]


def _calc_update_bfgs(
    data: SolverData, prob: NonlinearProgram, options: SolverOptions
) -> SolverData:
    """Update decision variables via BFGS + augmented Lagrangian.

    Args:
        data: The current iteration data.
        prob: The nonlinear program to solve.
        options: The optimizer parameters.

    Returns:
        Updated iteration data with the new x, λ, and Hessian approximation.
    """
    # Update the decision variables
    p = -data.H @ data.grad
    alpha = _linesearch(data.x, p, data.lmbda, prob, options)
    s = alpha * p
    x = data.x + s

    # Update the Hessian approximation
    # TODO: avoid re-computing this gradient, and give an option for using
    # the sampling-based version.
    grad_next = jax.grad(
        lambda x: _calc_lagrangian(x, data.lmbda, prob, options)[0]
    )(x)
    y = grad_next - data.grad
    rho = 1.0 / jnp.dot(y, s)
    I = jnp.eye(data.x.size)  # noqa: E741

    H = jnp.where(
        1e5,  # Ignore the update if yᵀs is too small
        (I - rho * s[:, None] @ y[None, :])
        @ data.H
        @ (I - rho * y[:, None] @ s[None, :])
        + rho * s[:, None] @ s[None, :],
        data.H,
    )

    # Update the Lagrange multipliers
    lmbda = data.lmbda + options.alpha * options.mu * data.h

    return data._replace(x=x, lmbda=lmbda, H=H)


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
    # N.B. standard if statement is fine here b/c options is a static argument
    if options.gradient_method == "autodiff":
        data = _calc_gradient_data_autodiff(data, prob, options)
    elif options.gradient_method == "sampling":
        data = _calc_gradient_data_sampling(data, prob, options)
    else:
        raise ValueError(f"Unknown gradient method: {options.gradient_method}")

    # Update the decision variables x and Lagrange multipliers λ.
    if options.method == "gradient_descent":
        data = _calc_update_gradient_descent(data, prob, options)
    elif options.method == "diffusion":
        data = _calc_update_diffusion(data, prob, options)
    elif options.method == "bfgs":
        data = _calc_update_bfgs(data, prob, options)
    else:
        raise ValueError(f"Unknown solve method: {options.method}")

    # Clip to the feasible region. This should be a no-op if the log barrier
    # is working, but that sometimes needs to be relaxed.
    x = data.x.at[prob.bounded_below].set(
        jnp.maximum(data.x[prob.bounded_below], prob.lower)
    )
    x = x.at[prob.bounded_above].set(
        jnp.minimum(x[prob.bounded_above], prob.upper)
    )

    return data._replace(k=data.k + 1, x=x)


def make_warm_start(
    prob: NonlinearProgram, options: SolverOptions, guess: jnp.ndarray
) -> SolverData:
    """Initialize solver data from an initial guess for x.

    Args:
        prob: The nonlinear program to solve.
        options: The optimizer parameters.
        guess: The initial guess for the decision variables x

    Returns:
        Full initial solver data, including lagrange multipliers, etc.
    """
    h = jax.jit(prob.residual)(guess)
    return SolverData(
        k=0,
        x=guess,
        lmbda=jnp.zeros_like(h),
        f=0.0,
        h=h,
        grad=jnp.zeros_like(guess),
        H=jnp.eye(guess.size),
        lagrangian=0.0,
        noise_decay_factor=4.0 / options.num_iters,
        rng=jax.random.key(options.seed),
    )


def solve(
    prob: NonlinearProgram, options: SolverOptions, data: SolverData
) -> SolverData:
    """Solve the optimization problem.

    Args:
        prob: The nonlinear program to solve.
        options: The optimizer parameters.
        data: A SolverData object with initial guess and other warm-start data.

    Returns:
        The solution, including decision variables and other data.
    """
    # TODO: use while loop and add early termination for errors
    scan_fn = lambda data, _: (_optimizer_step(data, prob, options), None)
    data, _ = jax.lax.scan(scan_fn, data, jnp.arange(options.num_iters))
    return data


def solve_verbose(
    prob: NonlinearProgram,
    options: SolverOptions,
    guess: jnp.ndarray,
    print_every: int = 500,
) -> SolverData:
    """Solve the optimization problem and print status updates along the way.

    Args:
        prob: The nonlinear program to solve.
        options: The optimizer parameters.
        guess: An initial guess for the decision variables.
        print_every: How many iterations to wait between printing status.

    Returns:
        The solution, including decision variables and other data.
    """
    data = make_warm_start(prob, options, guess)

    # Determine how many iterations to run between printouts
    print_every = min(options.num_iters, print_every)
    num_prints = options.num_iters // print_every

    # Update function solves a subset of the total iterations
    update_fn = jax.jit(
        lambda data: solve(prob, options._replace(num_iters=print_every), data)
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
