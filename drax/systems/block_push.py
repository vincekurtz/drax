import jax.numpy as jnp
import mujoco

from drax import ROOT
from drax.ocp import OptimalControlProblem


class BlockPush(OptimalControlProblem):
    """A block pushing problem."""

    def __init__(self, horizon: int, x_init: jnp.ndarray):
        """Initialize the block pushing problem.

        Args:
            horizon: The number of time steps T.
            x_init: The initial state x₀.
        """
        # Load the mujoco model
        model_path = ROOT + "/systems/block_push.xml"
        mj_model = mujoco.MjModel.from_xml_path(model_path)

        # Create the mjx model

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """The block dynamics xₜ₊₁ = f(xₜ, uₜ)."""
        pass

    def running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """The running cost ℓ(xₜ, uₜ)."""
        pass

    def terminal_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """The terminal cost ϕ(x_T)."""
        pass
