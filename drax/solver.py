import jax
import jax.numpy as jnp

from drax.nlp import NonlinearProgram


def solve(prob: NonlinearProgram, guess: jnp.ndarray) -> jnp.ndarray:
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

    def lagrangian(x: jnp.ndarray, lmbda: jnp.ndarray) -> jnp.ndarray:
        """The Lagrangian L(x, λ) = f(x) + λᵀh(x) + μ/2 h(x)²."""
        h = prob.residual(x)
        return prob.objective(x) + lmbda.T @ h + 0.5 * mu * h.T @ h

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
        x -= step_size * dL
        lmbda += step_size * mu * jit_constraints(x)

        # TODO: enforce bound constraints with log barrier + clipping

        if i % print_every == 0:
            # TODO: replace this with a status callback
            cost = jit_cost(x)
            cons = jnp.mean(jnp.square(jit_constraints(x)))
            grad = jnp.linalg.norm(dL)
            print(
                f"Iter {i}: cost = {cost:.4f}, cons = {cons:.4f}, "
                f"ℒ = {L:.4f}, grad = {grad:.4f}"
            )

    return x
