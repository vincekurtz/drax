##
#
# An example of using Drax to solve a pendulum swingup problem.
#
##

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from drax.solver import SolverOptions, make_warm_start, optimizer_step, solve
from drax.systems.pendulum import PendulumSwingup


def optimize() -> None:
    """Solve the swingup problem and print the solution."""
    prob = PendulumSwingup(horizon=50, x_init=jnp.array([3.1, 0.0]))

    # Set the solver options
    options = SolverOptions(
        num_iters=20000,
        print_every=5000,
        alpha=0.01,
        mu=10.0,
        rho=0.01,
        gradient_method="autodiff",
        sigma=0.01,
        num_samples=128,
    )

    # Solve from a zero initial guess
    sol = solve(prob, options, jnp.zeros(prob.num_vars))

    # Plot the solution
    xs, us = prob.unflatten(sol.x)
    plt.subplot(2, 1, 1)
    prob.plot_scenario()
    plt.plot(xs[:, 0], xs[:, 1], "ro-")

    plt.subplot(2, 1, 2)
    plt.plot(jnp.arange(prob.horizon - 1) * prob.dt, us, "bo-")
    plt.xlabel("Time (s)")
    plt.ylabel("Control Torque (Nm)")

    plt.show()


def animate() -> None:
    """Solve the swingup problem and animate the solution process."""
    prob = PendulumSwingup(horizon=50, x_init=jnp.array([3.1, 0.0]))

    # Set the solver options
    options = SolverOptions(
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
    guess = jax.random.uniform(
        init_rng, (prob.num_vars,), minval=-6.0, maxval=6.0
    )
    data = make_warm_start(prob, guess)

    # Solve the problem, recording the solution at intermediate steps
    all_data = [data]
    jit_optimizer_step = jax.jit(optimizer_step, static_argnums=(1, 2))
    for _ in range(5000):
        data = jit_optimizer_step(data, prob, options)
        if data.k % 100 == 0:
            all_data.append(data)

    # Recover the state and control trajectories from the solution
    xs, us = prob.unflatten(data.x)

    # Make an animation of the solution process
    plt.figure()

    prob.plot_scenario()
    path = plt.plot([], [], "ro-")[0]

    def _update(i: int):
        xs, _ = prob.unflatten(all_data[i].x)
        path.set_data(xs[:, 0], xs[:, 1])
        return path

    anim = FuncAnimation(  # noqa: F841 (anim needs to stay in scope)
        plt.gcf(), _update, frames=len(all_data), interval=100
    )

    plt.show()


if __name__ == "__main__":
    optimize()
    animate()
