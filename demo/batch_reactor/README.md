# Batch reactor: Temperature control under parametric uncertainty

A temperature-controlled batch reactor running the second-order reaction
$2A \to B \to C$ — $B$ is the desired product, $C$ an autocatalytic waste. The
temperature profile $T(t)$ is optimized while the **decomposition collision factor
$k_{20}$ is uncertain**, so the control must perform well across the whole
distribution of $k_{20}$ rather than at a single guessed value. This demo is based on the first numerical example of

> P. Terwiesch and M. Agarwal, "Robust input policies for batch reactors under
> parametric uncertainty," *Chemical Engineering Communications* **131** (1995),
> 33–52, https://doi.org/10.1080/00986449508936282

The paper's three strategies map directly onto the sample average approximation
(SAA) risk settings:

| Paper | Here | Meaning |
| --- | --- | --- |
| nominal | **nominal** | solve at the mean $k_{20} = 1000$ |
| "robust" (risk neutral) | **risk-neutral** | optimize the *expected* yield over $k_{20}$ |
| minimax (worst case) | **CVaR risk-averse** ($\beta = 0.5, 0.95$) | optimize the worst $(1-\beta)$ tail; two points on the risk dial, $\beta \to 1 \approx$ minimax |

conditional value-at-risk (CVaR)

## The model — [batch_reactor.py](batch_reactor.py)

With $x_1 = [A]$ and $x_2 = [B]$:

- **State** $x = (x_1, x_2) \in \mathbb{R}^2$, **control** $T(t)$ (temperature),
  **horizon** $t_f = 1$.
- **Dynamics** (dimensionless, Eqs. 22–23; $[C]$ eliminated by the mass balance
  $[A] + 2[B] + 2[C] = \mathrm{const}$):

$$
\dot{x}_1 = -2 k_1 x_1^2, \qquad
\dot{x}_2 = k_1 x_1^2 - \frac{1}{2} k_2 x_2 (1 - x_1 - 2 x_2),
$$

with Arrhenius rates $k_1 = k_{10} \mathrm{e}^{-E_1/(R T)}$ and
$k_2 = k_{20} \mathrm{e}^{-E_2/(R T)}$.

- **Initial state**: $x_1(0) = 1 - 2[C]_0 = 0.99$, $x_2(0) = 0$.
- **Objective** (maximized): the final product $x_2(t_f)$ — coded as
  $F(x) = -x_2(t_f)$ (minimized).
- **Control box**: $T \in [340, 420]$ K.
- **Constants**: $R = 1.987$, $E_1 = 3\cdot10^3$, $E_2 = 4\cdot10^3$,
  $k_{10} = 100$, $[C]_0 = 5\cdot10^{-3}$.
- **Uncertainty**: $k_{20} \sim \mathrm{TruncatedNormal}(1000, 500)$ on
  $[500, 2000]$, drawn i.i.d. by Monte Carlo sampling. The nominal problem fixes $k_{20}$ at
  its mean $\mathbb{E}[k_{20}] \approx 1000$.
- **Discretization**: $q = 50$ control intervals (stair function,
  $\Delta t = 0.02$), single shooting, RK4.

## Requirements & how to run

All commands below are run **from this demo folder** (`demo/batch_reactor/`).
Create the project virtualenv once, **activate** it, and install `ensemblecontrol`
(editable); then the drivers run with a plain `python`:

```bash
python3 -m venv ../../.venv
source ../../.venv/bin/activate
pip install -e ../..
python saa_batch_reactor.py   # policies + control/state + yield + confidence intervals
python clt_batch_reactor.py   # central-limit-theorem study
```

`saa_batch_reactor.py` solves the nominal, risk-neutral, and CVaR ($\beta = 0.95$)
policies with **IPOPT**; writes the control overlay and per-policy control/state
plots to [output/controls-state/](output/controls-state/); writes the
yield vs $k_{20}$ and out-of-sample yield-distribution figures; and runs the
plug-in and subsampling confidence intervals. Flags:
`--algorithm {plugin,subsampling,both}` (default both), `--m` / `--b`
(subsampling count / block size), `--workers`.

## Problem formulation

Let $x_2(t_f, \xi)$ be the final product $[B]$ at scenario $\xi = k_{20}$ under a
temperature policy $T$. All policies share the dynamics and $T \in [340, 420]$;
they differ only in how the uncertain terminal yield is scored.

**Nominal** — maximize the yield at the mean parameter only:

$$
\max_T x_2(t_f, 1000).
$$

**Risk-neutral** (the paper's *robust* policy) — maximize the *expected* yield; the
SAA replaces the expectation by the sample mean over $N$ i.i.d. scenarios:

$$
\max_T \mathbb{E}[x_2(t_f, \xi)]
\approx
\max_T \frac{1}{N}\sum_{i=1}^{N} x_2(t_f, \xi_i).
$$

**CVaR risk-averse** — maximize the mean yield over the worst $(1 - \beta)$
fraction of scenarios (the low-yield tail, which occurs at large $k_{20}$, i.e.
fast decomposition). Writing the per-scenario loss $F_i = -x_2(t_f, \xi_i)$, the
empirical CVaR of the loss $F$ has the Rockafellar–Uryasev variational form

$$
\mathrm{CVaR}_\beta(F) = \min_{\tau \in \mathbb{R}} \{ t + \frac{1}{(1-\beta)N}\sum_{i=1}^{N}\max\\{0,F_i - \tau\\} \},
$$

where $\tau$ is the Value-at-Risk
level. Introducing one slack $s_i$ per scenario to lift each $\max\\{0,F_i - \tau\\}$ turns the
SAA into a smooth joint minimization over the control $u$, the threshold $t$, and
the slacks $s$ — a deterministic multi-scenario optimal control problem
(implemented in [`risk_measures.py`](../../src/ensemblecontrol/risk_measures.py)):

$$
\min_{u, \tau, s} t + \frac{1}{(1-\beta)N}\sum_{i=1}^{N} s_i
$$

subject to $s_i \ge F_i - \tau$ and $s_i \ge 0$ for $i = 1, \ldots, N$. The scenarios
couple only through the shared control $u$ and threshold $t$. We solve two points on
the risk-aversion dial, $\beta = 0.5$ and $\beta = 0.95$ (the mean over the worst
50% and the worst 5% of scenarios); $\beta \to 1$ approaches the paper's **minimax**
(worst-case) policy.

## Control profiles

The optimal temperature profiles $T(t)$ for the four policies, overlaid on the
$[340, 420]$ box (each policy drawn with a distinct color and line style): hot early
to drive the main reaction, then cooling toward the end to suppress the $B \to C$
decomposition — the more risk-averse the policy, the cooler it runs through the
tail-sensitive second half, with CVaR $\beta = 0.95$ coolest and $\beta = 0.5$
sitting between it and risk-neutral.

![all controls](output/controls-state/all_controls.png)

Per-policy temperature profiles are in
[output/controls-state/](output/controls-state/) as
`{nominal,risk-neutral,cvar-0.50,cvar-0.95}_control.png`.

The corresponding **product state** $B(t)$ -- its ensemble mean
$\mathbb{E}[B(t, \xi)]$ with a $\pm 3$ standard-deviation band, obtained by simulating each fixed
control across the *same* $k_{20}$ ensemble (so even the nominal policy, whose own
solve carries a single scenario, shows a meaningful spread) on a shared $y$-axis —
makes the robustness ordering visible: the nominal control, tuned to $k_{20} = 1000$,
spreads most through the tail-sensitive second half; risk-neutral is tighter, CVaR
$\beta = 0.5$ tighter still, and CVaR $\beta = 0.95$ tightest.

| nominal | risk-neutral | CVaR $\beta = 0.5$ | CVaR $\beta = 0.95$ |
| --- | --- | --- | --- |
| ![nominal B](output/controls-state/nominal_states.png) | ![risk-neutral B](output/controls-state/risk-neutral_states.png) | ![CVaR 0.5 B](output/controls-state/cvar-0.50_states.png) | ![CVaR 0.95 B](output/controls-state/cvar-0.95_states.png) |

## The mean–tail trade-off out of sample

The control is fixed before $k_{20}$ is known, so a good policy must hedge the
whole distribution. **Per scenario**, the terminal yield as a function of $k_{20}$
(the paper's Figure 2) shows the trade-off: the nominal policy peaks near
$k_{20} = 1000$ but plunges in the tail; risk-neutral is flatter, CVaR $\beta = 0.5$
flatter still, and CVaR $\beta = 0.95$ is flattest.

![yield vs k20](output/yield_vs_k20.png)

**Aggregated** over a large independent out-of-sample draw
$k_{20} \sim \mathrm{TruncatedNormal}(1000, 500)$, the histogram of the terminal
yield $[B]$ under each policy (filled marker = the mean $\mathbb{E}[B]$; open marker
= the *worst-5% mean*, both on the $x$-axis) makes the ordering precise:

![yield distribution](output/yield_distribution.png)

The **worst-5% mean** is the average yield over the 5% of scenarios with the
*lowest* yield — the mean $[B]$ you obtain on the unluckiest 5% of batches (the
high $k_{20}$, fast-decomposition draws). It is the empirical CVaR of the yield and
is eactly the tail quantity the CVaR policy maximizes; a larger value means a
better worst case.

| policy | mean $\mathbb{E}[B]$ | worst-5% mean | min $[B]$ |
| --- | ---: | ---: | ---: |
| nominal | 0.3684 | 0.3210 | 0.3062 |
| risk-neutral (robust) | **0.3689** | 0.3324 | 0.3211 |
| CVaR, $\beta = 0.5$ | 0.3671 | 0.3429 | 0.3357 |
| CVaR, $\beta = 0.95$ | 0.3620 | **0.3468** | **0.3425** |

- **Risk-neutral beats nominal** on the *mean* yield ($0.3689 > 0.3684$): tuning to
  $k_{20} = 1000$ alone leaves expected yield on the table across the distribution.
- **CVaR $\beta = 0.95$ beats risk-neutral** on the *tail* (worst-5% mean
  $0.3468 > 0.3324$; worst case $0.3425 > 0.3211$): it accepts a lower mean
  ($0.3620$) to lift the worst outcomes — the risk-averse choice when a bad batch is
  costly.
- **CVaR $\beta = 0.5$ is the intermediate dial setting**: it captures most of the
  tail gain (worst-5% mean $0.3429$) for almost no loss of mean yield ($0.3671$),
  interpolating between risk-neutral and $\beta = 0.95$.

(The nominal yield at $k_{20} = 1000$ is $\approx 0.377$, matching the paper's
Table 1 value $0.3773$.)

## Statistical inference

Only the **risk-neutral** SAA optimal value $\hat J_N^* = \mathbb{E}_N[-B]$ is
analyzed (the nominal and CVaR policies are not swept). 

**Plug-in CI**  and **subsampling CI** , $\hat J_N^*$ vs $N$ with the interval at 95%:

| plug-in CI | subsampling CI |
| --- | --- |
| ![plug-in CI](output/risk-neutral-inference/risk-neutral_plugin_ci95.png) | ![subsampling CI](output/risk-neutral-inference/risk-neutral_subsampling_ci95.png) |


**Central-limit-theorem study** ([clt_batch_reactor.py](clt_batch_reactor.py)) — the
histograms of $\sqrt{N}(\hat J_N^* - \hat J_{\mathrm{ref}}^*)$ across
$N \in \{32, 64, 128\}$, which approach a centered Gaussian
as $N$ grows:

![CLT histograms](output/risk-neutral-limit-theorem/risk-neutral_clt_all.png)

(The SAA optimal value here is $J = \mathbb{E}[-B] < 0$, since the package minimizes
and the objective maximizes $B$; the confidence intervals bound that value.)
