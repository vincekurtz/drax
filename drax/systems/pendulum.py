import jax.numpy as jnp

from drax.ocp import OptimalControlProblem


class PendulumSwingup(OptimalControlProblem):
    """An inverted pendulum swingup problem."""

    def __init__(self, horizon: int, x_init: jnp.ndarray):
        """Initialize the pendulum swingup problem.

        Args:
            horizon: The number of time steps T.
            x_init: The initial state x₀.
        """
        # Constants
        self.m = 1.0
        self.g = 9.81
        self.l = 1.0
        self.dt = 0.1

        # Bounds
        x_min = jnp.array([-jnp.inf, -jnp.inf])
        x_max = jnp.array([jnp.inf, jnp.inf])
        u_min = jnp.array([-1.0])
        u_max = jnp.array([1.0])

        super().__init__(x_min, x_max, u_min, u_max, 2, 1, horizon, x_init)

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """The pendulum dynamics xₜ₊₁ = f(xₜ, uₜ)."""
        theta, theta_dot = x
        tau = u[0]
        theta_ddot = (
            tau - self.m * self.g * self.l * jnp.sin(theta - jnp.pi)
        ) / (self.m * self.l**2)
        xdot = jnp.array([theta_dot, theta_ddot])
        return x + self.dt * xdot

    def running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """The running cost ℓ(xₜ, uₜ)."""
        return self.dt * jnp.sum(u**2)

    def terminal_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """The terminal cost ϕ(x_T)."""
        theta, theta_dot = x
        return 10 * theta**2 + 1 * theta_dot**2

    def constraints(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Additional constraints g(xₜ, uₜ) ≤ 0."""
        return jnp.zeros(0)  # there are no constraints, return an empty array
