import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
        return self.dt * 0.01 * jnp.sum(u**2)

    def terminal_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """The terminal cost ϕ(x_T)."""
        theta, theta_dot = x
        return 10 * theta**2 + 1 * theta_dot**2

    def plot_scenario(self) -> None:
        """Make a vector field plot on the current matplotlib axes."""
        theta_range = (-1.5 * jnp.pi, 1.5 * jnp.pi)
        theta_dot_range = (-8.0, 8.0)
        plt.xlim(*theta_range)
        plt.ylim(*theta_dot_range)

        th = jnp.linspace(*theta_range, 20)
        thd = jnp.linspace(*theta_dot_range, 20)
        TH, THD = jnp.meshgrid(th, thd)
        X = jnp.stack([TH, THD], axis=-1)
        U = jnp.zeros((20, 20, 1))

        dX = jax.vmap(jax.vmap(self.dynamics))(X, U) - X
        plt.quiver(X[:, :, 0], X[:, :, 1], dX[:, :, 0], dX[:, :, 1], color="k")
        plt.xlabel("Angle (rad)")
        plt.ylabel("Angular velocity (rad/s)")
