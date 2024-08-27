import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from drax.ocp import OptimalControlProblem


class BugTrap(OptimalControlProblem):
    """A robot with unicycle dynamics in a U-shaped maze."""

    def __init__(self, horizon: int, x_init: jnp.ndarray):
        """Initialize the bug trap problem.

        Args:
            horizon: The number of time steps T.
            x_init: The initial state x₀.
        """
        self.target_position = jnp.array([1.0, 0.0])
        self.obs_positions = jnp.array(
            [
                [-1.0, 1.0],
                [-0.5, 1.0],
                [0.0, 1.0],
                [0.0, 0.5],
                [0.0, 0.0],
                [0.0, -0.5],
                [0.0, -1.0],
                [-0.5, -1.0],
                [-1.0, -1.0],
            ]
        )
        self.dt = 0.1

        # Bounds
        x_min = jnp.array([-3.0, -3.0, -jnp.inf])  # x, y, theta
        x_max = jnp.array([3.0, 3.0, jnp.inf])
        u_min = jnp.array([-1.0, -1.0])
        u_max = jnp.array([1.0, 1.0])

        super().__init__(x_min, x_max, u_min, u_max, 3, 2, horizon, x_init)

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Unicycle dynamics xₜ₊₁ = f(xₜ, uₜ)."""
        px, py, theta = x
        v, omega = u
        xdot = jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), omega])
        return x + self.dt * xdot

    def running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """The running cost ℓ(xₜ, uₜ)."""
        return self.dt * (jnp.sum(u**2) + 1000 * self._obstacle_cost(x))

    def terminal_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """The terminal cost ϕ(x_T)."""
        return 10 * jnp.sum((x[0:2] - self.target_position) ** 2)

    def _obstacle_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """Cost associated with being close to the obstacle.

        Args:
            x: The position of the robot.

        Returns:
            The obstacle-related cost at the given position.
        """

        def scan_fn(total_cost: float, obs_pos: jnp.ndarray):
            cost = jnp.exp(-5 * jnp.linalg.norm(x[0:2] - obs_pos) ** 2)
            return total_cost + cost, None

        total_cost, _ = jax.lax.scan(scan_fn, 0.0, self.obs_positions)
        return total_cost

    def plot_scenario(self) -> None:
        """Make a matplotlib visualization of the maze and goal."""
        # Green star at the target position.
        plt.plot(*self.target_position, "g*", markersize=20)

        # Red contour plot of the obstacle cost
        px = jnp.linspace(-3, 3, 100)
        py = jnp.linspace(-3, 3, 100)
        PX, PY = jnp.meshgrid(px, py)
        X = jnp.stack([PX, PY], axis=-1)
        C = jax.vmap(jax.vmap(self._obstacle_cost))(X)
        plt.contourf(PX, PY, C, cmap="Reds", levels=100)

        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
