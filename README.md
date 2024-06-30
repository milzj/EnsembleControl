[![DOI](https://zenodo.org/badge/814172654.svg)](https://zenodo.org/doi/10.5281/zenodo.11669862)

# EnsembleControl

The package is designed to solve optimal control problems that take the form

$$
\min_{u \in U} \, \frac{1}{N} \sum_{i=1}^N F(x^u(1,\xi^i), \xi^i) + (\alpha/2)\|\|_{L^2(0,1;\mathbb{R}^m)}^2,
$$

where for each parameter $\xi \in \Xi$ and control $u(\cdot) \in L^2(0,1;\mathbb{R}^m)$, $x^u(\cdot, \xi) = x(\cdot, \xi)$ solves the uncertain

$$
\dot{x}(t, \xi)  = f(x(t,\xi), u(t), \xi), \quad t \in (0,1), \quad x(0,\xi) = x_0(\xi).
$$

Here $f \colon \mathbb{R}^n  \times \mathbb{R}^m \times \Xi \to \mathbb{R}$ is the parameterized right-hand side. This parameterized initial value problem allows for uncertain right-hand sides and initial values. The set $U$ is a subset of L^2(0,1;\mathbb{R}^m) such that 

$$
    a_i \leq u_i(t) \leq b_i, \quad i = 1, \ldots, m, \quad t \in (0,1).
$$


# Installation

```
pip install git+https://github.com/milzj/EnsembleControl
```

# Demo

See [/demo/](demo).

