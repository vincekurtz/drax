##
#
# An example of using Drax to solve a pendulum swingup problem.
#
##

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from drax.solver import SolverOptions, make_warm_start, solve_from_warm_start
from drax.systems.pendulum import PendulumSwingup

# Set up the optimization problem
prob = PendulumSwingup(horizon=50, x_init=jnp.array([3.1, 0.0]))

# Set the solver options
options = SolverOptions(
    num_iters=20000,
    print_every=5000,
    alpha=0.01,
    gradient_method="autodiff",
    sigma=0.1,
    num_samples=100,
)

# Create a random initial guess
rng = jax.random.key(0)
rng, init_rng = jax.random.split(rng, 2)
guess = jax.random.uniform(init_rng, (prob.num_vars,), minval=-0.5, maxval=0.5)
data = make_warm_start(prob, guess)

# Solve the problem
data = solve_from_warm_start(prob, options, data)
xs, us = prob.unflatten(data.x)  # Recover the state and control trajectories

# Plot the solution
plt.subplot(2, 1, 1)
prob.plot_scenario()
plt.plot(xs[:, 0], xs[:, 1], "ro-")

plt.subplot(2, 1, 2)
plt.plot(jnp.arange(prob.horizon - 1) * prob.dt, us, "bo-")
plt.xlabel("Time (s)")
plt.ylabel("Control Torque (Nm)")

plt.show()
