##
#
# An example of using Drax to solve a pendulum swingup problem.
#
##

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from drax.solver import SolverOptions, make_warm_start, solve, solve_verbose
from drax.systems.pendulum import PendulumSwingup


def optimize() -> None:
    """Solve the swingup problem and print the solution."""
    prob = PendulumSwingup(horizon=50, x_init=jnp.array([3.1, 0.0]))

    # Set the solver options
    options = SolverOptions(
        num_iters=1000,
        alpha=0.01,
        mu=10.0,
        rho=0.1,
        gradient_method="autodiff",
        sigma=0.01,
        num_samples=128,
        # method="gradient_descent",
        method="bfgs",
        initial_noise_level=0.1,
    )

    # Solve from a zero initial guess
    sol = solve_verbose(prob, options, jnp.zeros(prob.num_vars), print_every=10)

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


def optimize_parallel() -> None:
    """Solve a bunch of swingups from different initial conditions."""
    N = 100  # number of parallel problems to solve

    # Sample a bunch of initial states
    rng = jax.random.PRNGKey(0)
    x_inits = jax.random.uniform(rng, (N, 2), minval=-4.0, maxval=4.0)

    # Set the solver options
    options = SolverOptions(
        num_iters=20000,
        gradient_method="autodiff",
    )

    # Set up an optimization function that maps x0 -> solution
    def optimize_single(x_init: jnp.ndarray) -> jnp.ndarray:
        prob = PendulumSwingup(horizon=50, x_init=x_init)
        warm_start = make_warm_start(prob, options, jnp.zeros(prob.num_vars))
        sol = solve(prob, options, warm_start)
        return prob.unflatten(sol.x)

    # Solve a single problem
    st = time.time()
    xs, us = optimize_single(x_inits[0])
    jax.block_until_ready(xs)
    print(f"Solved a single problem in {time.time() - st:.2f} s")

    # Solve all the problems in parallel
    st = time.time()
    xs, us = jax.vmap(optimize_single)(x_inits)
    jax.block_until_ready(xs)
    print(f"Solved {N} problems in {time.time() - st:.2f} s")

    # Plot the results
    PendulumSwingup(10, jnp.zeros(2)).plot_scenario()  # dummy prob for plots
    for xs_i in xs:
        plt.plot(xs_i[:, 0], xs_i[:, 1], "bo-", alpha=0.3)
    plt.show()


def animate() -> None:
    """Solve the swingup problem and animate the solution process."""
    rng = jax.random.key(0)
    N = 10  # Number of initial guesses to use at once
    prob = PendulumSwingup(horizon=50, x_init=jnp.array([3.1, 0.0]))

    # Set the solver options
    options = SolverOptions(
        alpha=0.01,
        mu=10.0,
        rho=0.01,
        gradient_method="sampling",
        sigma=0.01,
        num_samples=128,
        method="diffusion",
        initial_noise_level=1.0,
    )

    # Create random initial guesses
    def initialize(rng: jnp.ndarray) -> jnp.ndarray:
        return make_warm_start(
            prob,
            options,
            jax.random.uniform(rng, (prob.num_vars,), minval=-6.0, maxval=6.0),
        )

    rng, init_rng = jax.random.split(rng, 2)
    init_rngs = jax.random.split(init_rng, N)
    data = jax.vmap(initialize)(init_rngs)

    # Solve the problem, recording the solution at intermediate steps
    vmap_solve = jax.vmap(solve, in_axes=(None, None, 0))
    jit_solve = jax.jit(vmap_solve, static_argnums=(0, 1))
    num_saves = options.num_iters // 100
    all_data = [data]
    options = options._replace(num_iters=100)
    for _ in range(num_saves):
        data = jit_solve(prob, options, data)
        all_data.append(data)

    # Make an animation of the solution process
    plt.figure()

    prob.plot_scenario()
    path = plt.plot([], [], "ro")[0]

    def _update(i: int):
        xs, _ = jax.vmap(prob.unflatten)(all_data[i].x)
        path.set_data(xs[..., 0], xs[..., 1])
        return path

    anim = FuncAnimation(  # noqa: F841 (anim needs to stay in scope)
        plt.gcf(), _update, frames=len(all_data), interval=100
    )

    plt.show()


if __name__ == "__main__":
    optimize()
    # optimize_parallel()
    # animate()
