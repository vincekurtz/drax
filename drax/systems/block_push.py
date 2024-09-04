import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from drax import ROOT
from drax.ocp import OptimalControlProblem


class BlockPush(OptimalControlProblem):
    """A block pushing problem."""

    def __init__(self, horizon: int, target: jnp.ndarray = None):
        """Initialize the block pushing problem.

        This system has 10 state variables:
            - Block position (x, y, theta)
            - Pusher position (x, y)
            - Block velocity (x, y, theta)
            - Pusher velocity (x, y)

        And 2 controls:
            - Target pusher position (x, y)

        Args:
            horizon: The number of time steps T.
            target: (optional) The target block position.
        """
        # Create the mjx model
        model_path = ROOT + "/systems/block_push.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mjx_model = mjx.put_model(self.mj_model)

        # State and control constraints
        x_min = jnp.array([-jnp.inf for _ in range(10)])
        x_max = jnp.array([jnp.inf for _ in range(10)])
        u_min = jnp.array([-1.5, -1.5])
        u_max = jnp.array([1.5, 1.5])

        # Initial state
        x_init = jnp.zeros(10)

        # Target state
        self.x_target = target
        if target is None:
            self.x_target = jnp.array(
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )

        # Cost matrices
        self.Q = jnp.diag(
            jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0])
        )
        self.R = jnp.diag(jnp.array([0.001, 0.001]))

        # Jittable step function
        self._jit_step = jax.jit(lambda data: mjx.step(self.mjx_model, data))

        super().__init__(x_min, x_max, u_min, u_max, 10, 2, horizon, x_init)

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """The block dynamics xₜ₊₁ = f(xₜ, uₜ)."""
        mjx_data = mjx.make_data(self.mjx_model)
        mjx_data = mjx_data.tree_replace(
            {"qpos": x[:5], "qvel": x[5:], "ctrl": u}
        )
        mjx_data = self._jit_step(mjx_data)
        return jnp.concatenate([mjx_data.qpos, mjx_data.qvel])

    def running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """The running cost ℓ(xₜ, uₜ)."""
        dt = self.mjx_model.opt.timestep
        x_err = x - self.x_target
        return dt * (jnp.sum(x_err @ self.Q @ x_err) + jnp.sum(u @ self.R @ u))

    def terminal_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """The terminal cost ϕ(x_T)."""
        x_err = x - self.x_target
        return jnp.sum(x_err @ self.Q @ x_err)

    def visualize_trajectory(self, xs: jnp.ndarray, us: jnp.ndarray) -> None:
        """Visualize the trajectory in the Mujoco viewer."""
        # Convert to numpy arrays
        xs = np.array(xs)
        us = np.array(us)
        mj_data = mujoco.MjData(self.mj_model)
        dt = float(self.mj_model.opt.timestep)

        # Run the "simulation", which just visualizes the trajectory
        t = 0
        with mujoco.viewer.launch_passive(self.mj_model, mj_data) as viewer:
            while viewer.is_running():
                start_time = time.time()

                # Update the position
                mj_data.qpos = xs[t, :5]
                mj_data.qvel = xs[t, 5:]
                mj_data.ctrl = us[t]

                # Update the viewer
                mujoco.mj_forward(self.mj_model, mj_data)
                viewer.sync()

                # Try to run in realtime
                elapsed_time = time.time() - start_time
                if elapsed_time < dt:
                    time.sleep(dt - elapsed_time)

                # Loop over the trajectory
                t += 1
                if t == len(xs):
                    time.sleep(1.0)
                    t = 0
