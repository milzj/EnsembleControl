from casadi import *

def MultipleShootingProblem(final_cost_function,
                            dynamics,
                            initial_state,
                            control_bounds,
                            nsamples,
                            nintervals,
                            beta=0.0):

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

    if beta > 0 and beta < 1.0:
        # TODO: Use a priori bounds on t for initial value
        t = MX.sym('t', 1)
        w += [t]
        lbw += [-inf]
        ubw += [inf]
        t0 = 0.0
        w0 += [t0]
        J += t 
    
    xk = xk.full().flatten()[idx]
    for i in range(nsamples):
        idx = np.arange(nstates // nsamples)+i*(nstates // nsamples)
        # TODO: Improve implementation of averaging
        if beta == 0.0:
            J += final_cost_function(Xk[idx])/nsamples
        elif beta > 0.0 and beta < 1.0:
            # https://doi.org/10.1061/AJRUA6.0000816
            j = final_cost_function(Xk[idx])
            j0 = final_cost_function(xk[idx])
            r = MX.sym('r_' + str(i), 1)
            w += [r]
            lbw += [0.0]
            ubw += [inf]
            w0 += [max(j0-t0,0.0)+1.0]
            J += (1/(1-beta)/nsamples)*r
            g   += [j-t-r]
            lbg += [-inf]
            ubg += [0.0]
        
    objective = J
    constraints = vertcat(*g)
    decisions = vertcat(*w)

    return objective, constraints, decisions, w0, [lbw, ubw, lbg, ubg]
