from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp

from drax.nlp import NonlinearProgram


class OptimalControlProblem(NonlinearProgram, ABC):
    """Abstract base class for a generic optimal control problem.

    Defines a trajectory optimization problem of the form

        min_{x, u} ∑ₜ ℓ(xₜ, uₜ) + ϕ(x_T)
        s.t. xₜ₊₁ = f(xₜ, uₜ)
             g(xₜ, uₜ) ≤ 0
             x_min ≤ xₜ ≤ x_max
             u_min ≤ uₜ ≤ u_max
             x₀ = x_init

    and translates it into a generic nonlinear program.
    """

    def __init__(
        self,
        x_min: jnp.ndarray,
        x_max: jnp.ndarray,
        u_min: jnp.ndarray,
        u_max: jnp.ndarray,
        nx: int,
        nu: int,
        horizon: int,
        x_init: jnp.ndarray,
    ):
        """Initialize the optimal control problem.

        Args:
            x_min: The lower bounds on the state variables.
            x_max: The upper bounds on the state variables.
            u_min: The lower bounds on the control variables.
            u_max: The upper bounds on the control variables.
            nx: The number of state variables.
            nu: The number of control variables.
            horizon: The number of time steps T.
            x_init: The initial state x₀.
        """
        assert x_min.shape == (nx,)
        assert x_max.shape == (nx,)
        assert u_min.shape == (nu,)
        assert u_max.shape == (nu,)
        assert x_init.shape == (nx,)

        self.nx = nx
        self.nu = nu
        self.x_init = x_init

        # Total number of decision variables (state and control variables).
        # Note that the initial state x₀ is fixed and not a decision variable.
        # TODO: support inequality constraints via slack variables
        num_vars = (nx + nu) * (horizon - 1)

        # Tile the bounds to cover all decision variables.
        lower = jnp.concatenate(
            [jnp.tile(x_min, horizon - 1), jnp.tile(u_min, horizon - 1)]
        )
        upper = jnp.concatenate(
            [jnp.tile(x_max, horizon - 1), jnp.tile(u_max, horizon - 1)]
        )

        super().__init__(num_vars, lower, upper)

    @abstractmethod
    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """The state transition function xₜ₊₁ = f(xₜ, uₜ).

        Args:
            x: The state variables.
            u: The control variables.

        Returns:
            The next state xₜ₊₁.
        """
        raise NotImplementedError

    @abstractmethod
    def constraints(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """The inequality constraints g(x, u) <= 0.

        Args:
            x: The state variables.
            u: The control variables.

        Returns:
            A vector of inequality constraint values g(x, u).
        """
        raise NotImplementedError

    @abstractmethod
    def running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """The running cost ℓ(x, u).

        Args:
            x: The state variables.
            u: The control variables.

        Returns:
            The (scalar) running cost ℓ(x, u).
        """
        raise NotImplementedError

    @abstractmethod
    def terminal_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """The terminal cost ϕ(x).

        Args:
            x: The state variables.

        Returns:
            The (scalar) terminal cost ϕ(x).
        """
        raise NotImplementedError

    def _unflatten(self, vars: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Reshape the decision variables into states and controls.

        Args:
            vars: vector of states and controls at each time step

        Returns:
            An array of states [x₀, x₁, ... x_T], size (horizon, nz)
            An array of controls [u₀, u₁, ..., u_T-1], size (horizon - 1, nu)
        """
        nx_vars = self.nx * (self.horizon - 1)
        states = vars[:nx_vars].reshape(self.horizon - 1, self.nx)
        controls = vars[nx_vars:].reshape(self.horizon - 1, self.nu)
        states = jnp.vstack([self.x_init, states])
        return states, controls

    def objective(self, vars: jnp.ndarray) -> jnp.ndarray:
        """The total cost ∑ₜ ℓ(xₜ, uₜ) + ϕ(x_T).

        Args:
            vars: vector of states and controls at each time step

        Returns:
            The (scalar) cost.
        """
        xs, us = self._unflatten(vars)
        running = jax.vmap(self.running_cost(xs[:-1], us))
        terminal = self.terminal_cost(xs[-1])
        return jnp.sum(running) + terminal

    def residual(self, vars: jnp.ndarray) -> jnp.ndarray:
        """Stacked equality constraints from dynamics and other constraints.

        Args:
            vars: vector of states and controls at each time step

        Returns:
            A vector of equality constraint residuals.
        """
        xs, us = self._unflatten(vars)

        x_pred = jax.vmap(self.dynamics)(xs[:-1], us)
        x_next = xs[1:]
        dynamics_residual = (x_pred - x_next).flatten()

        # TODO: support inequality constraints via slack variables
        return dynamics_residual
