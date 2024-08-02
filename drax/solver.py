from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from drax.nlp import NonlinearProgram


class SolverOptions(NamedTuple):
    """Object holding optimizer parameters.

    Parameters:
        num_iters: The total number of iterations to run.
        alpha: The step size.
        mu: The augmented Lagrangian penalty parameter.
        rho: The log barrier parameter for bound constraints.
    """

    num_iters: int = 5000
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

    """

    k: int
    x: jnp.ndarray
    lmbda: jnp.ndarray
    f: jnp.ndarray
    h: jnp.ndarray
    grad: jnp.ndarray
    lagrangian: jnp.ndarray


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


def _calc_gradient_data(
    data: SolverData, prob: NonlinearProgram, options: SolverOptions
) -> SolverData:
    """Compute the current cost, constraints, and lagrangian, and gradient.

    Args:
        data: The current iteration data.
        prob: The nonlinear program to solve.
        options: The optimizer parameters.

    Returns:
        A SolverData object with updated f, h, grad, and lagrangian.
    """
    # TODO: support sampling-based approximation
    grad_fn = jax.value_and_grad(
        lambda x: _calc_lagrangian(x, data.lmbda, prob, options), has_aux=True
    )

    (L, (f, h)), grad = grad_fn(data.x)
    return data._replace(f=f, h=h, grad=grad, lagrangian=L)


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
    data = _calc_gradient_data(data, prob, options)

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


def solve_from_warm_start(
    prob: NonlinearProgram, data: SolverData, options: SolverOptions
) -> SolverData:
    """Solve using a warm start from a previous solution."""
    pass


def solve_verbose(
    prob: NonlinearProgram,
    guess: jnp.ndarray,
    options: SolverOptions,
    print_every: int = 100,
) -> SolverData:
    """Solve with print statements every `print_every` iterations."""
    pass


def solve(prob: NonlinearProgram, guess: jnp.ndarray) -> jnp.ndarray:
    """Solve a generic nonlinear program using a direct method."""
    options = SolverOptions()

    data = SolverData(
        k=0,
        x=guess,
        lmbda=jnp.zeros_like(prob.residual(guess)),  # TODO: avoid reevaluating
        f=0.0,
        h=jnp.zeros_like(prob.residual(guess)),
        grad=jnp.zeros_like(guess),
        lagrangian=0.0,
    )

    jit_step = jax.jit(_optimizer_step, static_argnums=(1, 2))

    for k in range(options.num_iters):
        data = jit_step(data, prob, options)
        if k % 100 == 0:
            cons_viol = jnp.mean(jnp.square(data.h))
            grad_norm = jnp.linalg.norm(data.grad)
            print(
                f"Iter {k}: cost = {data.f:.4f}, cons = {cons_viol:.4f}, "
                f"lagrangian = {data.lagrangian:.4f}, grad = {grad_norm:.4f}"
            )

    return data.x


def old_solve(prob: NonlinearProgram, guess: jnp.ndarray) -> jnp.ndarray:
    """Solve a generic nonlinear program using a direct method.

    Args:
        prob: The nonlinear program to solve.
        guess: An initial guess for the decision variables.

    Returns:
        The optimal decision variables.
    """
    # TODO: make a solver parameters struct instead of hardcoding here
    num_iters = 5000
    print_every = 100
    step_size = 0.01
    mu = 10.0
    rho = 0.1

    def lagrangian(x: jnp.ndarray, lmbda: jnp.ndarray) -> jnp.ndarray:
        """The augmented Lagrangian L(x, λ) = f(x) + λᵀh(x) + μ/2 h(x)².

        Also adds log barrier terms to the cost to account enforce l <= x <= u.
        """
        f = prob.objective(x)
        h = prob.residual(x)

        # Add log barrier terms to the cost to enforce l <= x <= u.
        u_err = jnp.maximum(prob.upper - x[prob.bounded_above], 1e-4)
        l_err = jnp.maximum(x[prob.bounded_below] - prob.lower, 1e-4)
        f -= jnp.sum(rho * jnp.log(u_err))
        f -= jnp.sum(rho * jnp.log(l_err))

        # TODO: return f, h, and constraint violation
        return f + lmbda.T @ h + 0.5 * mu * h.T @ h

    # TODO: support sampling approximation
    # TODO: move all jitting outside this solve function
    jit_lagrangian_and_grad = jax.jit(jax.value_and_grad(lagrangian))
    jit_cost = jax.jit(prob.objective)
    jit_constraints = jax.jit(prob.residual)

    # TODO: user can provide an initial guess for both x and λ
    x = guess
    lmbda = jnp.zeros_like(jit_constraints(x))

    # TODO: replace with lax.while
    for i in range(num_iters):
        L, dL = jit_lagrangian_and_grad(x, lmbda)

        # Flow the decision variable and Lagrange multiplier according to
        #   ẋ = -∂L/∂x,
        #   λ̇ = ∂L/∂λ
        # See Platt and Barr, "Constrained Differential Optimization",
        # NeurIPS 1987 for more details.
        x += -step_size * dL
        lmbda += step_size * mu * jit_constraints(x)  # TODO: avoid reevaluating

        # Clip to the feasible region. This should be a no-op if the log barrier
        # is working, but that sometimes needs to be relaxed.
        x = x.at[prob.bounded_below].set(
            jnp.maximum(x[prob.bounded_below], prob.lower)
        )
        x = x.at[prob.bounded_above].set(
            jnp.minimum(x[prob.bounded_above], prob.upper)
        )

        if i % print_every == 0:
            # TODO: replace this with a status callback
            cost = jit_cost(x)
            cons = jnp.mean(jnp.square(jit_constraints(x)))
            grad = jnp.linalg.norm(dL)
            print(
                f"Iter {i}: cost = {cost:.4f}, cons = {cons:.4f}, "
                f"lagrangian = {L:.4f}, grad = {grad:.4f}"
            )

    return x
