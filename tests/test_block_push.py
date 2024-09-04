import jax.numpy as jnp

from drax.systems.block_push import BlockPush


def test_ocp() -> None:
    """Make sure we can construct the problem."""
    prob = BlockPush(horizon=10, x_init=jnp.array([0.0, 0.0]))


if __name__ == "__main__":
    test_ocp()
