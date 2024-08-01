# Drax

A python library for direct trajectory optimization on accelerators using JAX.

Drax solves optimal control problems of the form
$$
\begin{align}
\min_{x, u} & \sum_{t=0}^{T-1} \ell(x_t, u_t) + \phi(x_T) \\
\mathrm{s.t.}~& x_{t+1} = f(x_t, u_t) \\
              & g(x_t, u_t) \leq 0 \\
              & x_{min} \leq x_t \leq x_{max} \\
              & u_{min} \leq u_t \leq u_{max} \\
              & x_0 = x_{\mathrm{init}}
\end{align}
$$
on GPU or other accelerator hardware. $x_t$ is the system state, $u_t$ is a
control input, $f$ are the system dynamics, and $g$ are extra constrains, e.g.
for collision avoidance. The functions $\ell$ and $\phi$ are running and
terminal costs.

Drax targets applications in robotics and model predictive control, leading to 
the following emphasis:

- **Fast iteration times over precise convergence**. MPC problems are rarely solved
  to convergence, so we care more about arriving at a reasonable solution quickly.
- **Complex dynamics constraints**. The constraint $x_{t+1} = f(x_t, u_t)$
  is often the most difficult and nonlinear part of the problem.
- **GPU friendly operations**. Hardware accelerators are good at large matrix 
  multiplications like (e.g., $y = Ax$), but not so good at matrix inversions
  and linear solves (e.g., $x = A^{-1}y$). This makes standard non-convex 
  optimization methods like SQP less attractive.
- **Support for sampling-based approximations**. The dynamics gradients 
  ($\nabla_x f, \nabla_u f$) are often poorly defined or difficult to compute,
  Drax supports gradient-free optimization via randomized smoothing.

## Setup (Conda)

Set up a conda env with Cuda 12.3 support (first time only):

```bash
conda env create -n [env_name] -f environment.yml
```

Enter the conda env:

```bash
conda activate [env_name]
```

Install dependencies:

```bash
pip install -e . --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Set up pre-commit hooks:

```bash
pre-commit autoupdate
pre-commit install
```

## Usage

Run unit tests:

```bash
pytest
```

Other demos can be found in the `examples` folder.
