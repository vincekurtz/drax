##
#
# An example of using Drax to solve a pendulum swingup problem.
#
##

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from drax.solver import SolverOptions, make_warm_start, solve_from_warm_start
from drax.systems.pendulum import PendulumSwingup

# Set up the optimization problem
prob = PendulumSwingup(horizon=50, x_init=jnp.array([3.1, 0.0]))

# Set the solver options
options = SolverOptions(
    num_iters=100,
    print_every=100,
    alpha=0.01,
    mu=10.0,
    rho=0.01,
    gradient_method="sampling",
    sigma=0.01,
    num_samples=128,
)

# Create a random initial guess
rng = jax.random.key(0)
rng, init_rng = jax.random.split(rng, 2)
guess = jax.random.uniform(init_rng, (prob.num_vars,), minval=-6.0, maxval=6.0)
data = make_warm_start(prob, guess)

# Solve the problem, recording the solution at intermediate steps
all_data = [data]
for _ in range(50):
    # N.B. this is not the fastest way to do this, this gets re-jitted each time
    data = solve_from_warm_start(prob, options, data)
    all_data.append(data)
xs, us = prob.unflatten(data.x)  # Recover the state and control trajectories

# Plot the solution
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
prob.plot_scenario()
plt.plot(xs[:, 0], xs[:, 1], "ro-")

plt.subplot(2, 1, 2)
plt.plot(jnp.arange(prob.horizon - 1) * prob.dt, us, "bo-")
plt.xlabel("Time (s)")
plt.ylabel("Control Torque (Nm)")

# Make an animation of the solution process
plt.figure()

prob.plot_scenario()
path = plt.plot([], [], "ro-")[0]


def _update(i: int):
    xs, _ = prob.unflatten(all_data[i].x)
    path.set_data(xs[:, 0], xs[:, 1])
    return path


ani = FuncAnimation(plt.gcf(), _update, frames=len(all_data), interval=100)

plt.show()
