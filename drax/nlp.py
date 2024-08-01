from abc import ABC, abstractmethod

import jax.numpy as jnp


class NonlinearProgram(ABC):
    """Abstract base class for a generic nonlinear program.

    Defines a bound and equality-constrained problem

        min_x f(x)
        s.t. h(x) = 0
             l <= x <= u

    Note that problems with inequality constraints (g(x) <= 0) can be converted
    to this form by introducing slack variables.
    """

    def __init__(self, num_vars: int, lower: jnp.ndarray, upper: jnp.ndarray):
        """Initialize the nonlinear program.

        Args:
            num_vars: The number of decision variables.
            lower: The lower bounds on the decision variables.
            upper: The upper bounds on the decision variables.
        """
        assert lower.shape == (num_vars,)
        assert upper.shape == (num_vars,)
        self.num_vars = num_vars
        self.lower = lower
        self.upper = upper

    @abstractmethod
    def objective(self, x: jnp.ndarray) -> jnp.ndarray:
        """The cost function f(x).

        Args:
            x: The decision variables.

        Returns:
            The (scalar) cost.
        """
        raise NotImplementedError

    @abstractmethod
    def residual(self, x: jnp.ndarray) -> jnp.ndarray:
        """The equality constraints h(x).

        Args:
            x: The decision variables.

        Returns:
            A vector of equality constraint residuals.
        """
        raise NotImplementedError
