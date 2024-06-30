[![DOI](https://zenodo.org/badge/814172654.svg)](https://zenodo.org/doi/10.5281/zenodo.11669862)
[![MIT License](https://img.shields.io/github/license/milzj/EnsembleControl/)](LICENSE)

# EnsembleControl

The package is designed to solve optimal control problems that take the form

$$
\min_{u \in U} \frac{1}{N} \sum_{i=1}^N F(x^u(1,\xi^i), \xi^i) + (\alpha/2)\|u\|_{L^2(0,t_f;\mathbb{R}^m)}^2,
$$

where for each parameter $\xi \in \Xi$ and control $u(\cdot) \in L^2(0,t_f;\mathbb{R}^m)$, $x^u(\cdot, \xi) = x(\cdot, \xi)$ solves the uncertain dynamical system

$$
\dot{x}(t, \xi)  = f(x(t,\xi), u(t), \xi), \quad t \in (0,t_f), \quad x(0,\xi) = x_0(\xi).
$$

Here $\xi^i \in \Xi \subset \mathbb{R}^p$ are parameters, $t_f > 0$ is the final time, and
$f \colon \mathbb{R}^n  \times \mathbb{R}^m \times \Xi \to \mathbb{R}$ is the parameterized right-hand side.
This parameterized initial value problem allows for uncertain right-hand sides and initial values.
The set $U$ is a subset of $L^2(0,1;\mathbb{R}^m)$ such that 

$$
a_j \leq u_j(t) \leq b_j, \quad j = 1, \ldots, m, \quad t \in (0,t_f).
$$

Here $a_j$ and $b_j$ are numbers in $[-\infty, \infty]$ for $j=1, \dots, m$.

# Documentation

None.

The control problem is discretized using a multiple shooting approach
following [Direct multiple shooting](https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_multiple_shooting.py).
An explicit 4th order Runge--Kutta method is used to discretize the dynamical system.

# Installation

```
pip install git+https://github.com/milzj/EnsembleControl
```

# Demo

See [here](/demo).


# Docker

Create a local docker container via

```
docker build -t ensemblecontrol .
```

or

```
docker build -t ensemblecontrol . --no-cache --network=host
```

Run the docker container using

```
docker run -it ensemblecontrol sh
```

# Contributing

Information about how to contribute can be found
[here](CONTRIBUTING.md).

# Licence

See [here](LICENSE).
