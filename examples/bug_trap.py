import jax.numpy as jnp
import matplotlib.pyplot as plt

from drax.solver import SolverOptions, solve_verbose
from drax.systems.bug_trap import BugTrap

if __name__ == "__main__":
    prob = BugTrap(horizon=50, x_init=jnp.array([-1.0, 0.0, 0.0]))
    options = SolverOptions(
        num_iters=20000,
        method="bfgs",
        rho=0.01,
        initial_noise_level=4.0,
        mu=10.0,
        gradient_method="autodiff",
    )

    guess = jnp.zeros(prob.num_vars) + 0.1
    sol = solve_verbose(prob, options, guess)
    assert sol.x.shape == (prob.num_vars,)

    xs, us = prob.unflatten(sol.x)
    assert xs.shape == (prob.horizon, prob.nx)
    assert us.shape == (prob.horizon - 1, prob.nu)

    prob.plot_scenario()
    plt.plot(xs[:, 0], xs[:, 1], "o-")

    # axes off
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])

    plt.tight_layout()
    plt.show()
