from casadi import *

def MultipleShootingProblem(final_cost_function,
                            dynamics,
                            initial_state,
                            control_bounds,
                            nsamples,
                            nintervals):

    # Start with an empty NLP
    # Adapted from https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_multiple_shooting.py
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0.0
    g=[]
    lbg = []
    ubg = []

    lbu = control_bounds[0]
    ubu = control_bounds[1]

    # Take average of lower and upper bound as initial control
    u0 = [(x+y)/2 for x,y in zip(lbu, ubu)]
    u0 = list(np.nan_to_num(u0, posinf=0.0, neginf=0.0))

    nstates = len(initial_state)
    ncontrols = len(lbu)

    # "Lift" initial conditions
    Xk  = MX.sym('X0', nstates)
    w   += [Xk]
    lbw += initial_state
    ubw += initial_state
    w0  += initial_state

    xk = initial_state

    # Formulate the NLP
    for k in range(nintervals):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k), ncontrols)
        w   += [Uk]
        lbw += lbu
        ubw += ubu
        w0  += u0

        # Integrate till the end of the interval
        Fk = dynamics(x0=Xk, p=Uk)
        Xk_end = Fk['xf']
        J += Fk['qf']

        Fk = dynamics(x0=xk, p=u0)
        xk = Fk['xf']

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), nstates)
        w   += [Xk]
        lbw += nstates*[-inf]
        ubw += nstates*[inf]
        w0 += list(xk.full().flatten())

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += nstates*[0.0]
        ubg += nstates*[0.0]

    for i in range(nsamples):
      idx = np.arange(nstates // nsamples)+i*(nstates // nsamples)
      # TODO: Improve implementation of averaging
      J += final_cost_function(Xk[idx])/nsamples

    objective = J
    constraints = vertcat(*g)
    decisions = vertcat(*w)

    return objective, constraints, decisions, w0, [lbw, ubw, lbg, ubg]
