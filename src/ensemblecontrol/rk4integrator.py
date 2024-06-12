from casadi import *

__all__ = ["RK4Integrator"]

def RK4Integrator(rhs, final_time, nintervals, nstates, ncontrols, steps_per_interval=4):
  # Fixed step Runge-Kutta 4 integrator
  # Adapted from https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_multiple_shooting.py
  DT = final_time/nintervals/steps_per_interval
  X0 = MX.sym('X0', nstates)
  U = MX.sym('U', ncontrols)
  X = X0
  Q = 0.0
  for j in range(steps_per_interval):
    k1, k1_q = rhs(X, U)
    k2, k2_q = rhs(X + DT/2 * k1, U)
    k3, k3_q = rhs(X + DT/2 * k2, U)
    k4, k4_q = rhs(X + DT * k3, U)
    X = X + DT/6*(k1 +2*k2 +2*k3 +k4)
    Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
  integrator = Function('integrator', [X0, U], [X, Q],['x0','p'],['xf','qf'])
  return integrator
