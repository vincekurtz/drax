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
        """The augmented Lagrangian L(x, λ) = f(x) + λᵀh(x) + μ/2 h(x)²."""
        # TODO: add log barriers for the feasible region
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

        # Flow the decision variable and Lagrange multiplier according to
        #   ẋ = -∂L/∂x,
        #   λ̇ = ∂L/∂λ
        # See Platt and Barr, "Constrained Differential Optimization",
        # NeurIPS 1987 for more details.
        x += -step_size * dL
        lmbda += step_size * mu * jit_constraints(x)  # TODO: avoid reevaluating

        # Clip to the feasible region. This should be a no-op if the log barrier
        # is working, but that sometimes needs to be relaxed.
        x = jnp.clip(x, prob.lower, prob.upper)

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
