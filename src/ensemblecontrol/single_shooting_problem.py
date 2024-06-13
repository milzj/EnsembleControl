from casadi import *


def SingleShootingOptimizationProblem(objective_function,
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

    nstates = len(initial_state)
    ncontrols = len(lbu)

    Xk  = MX(initial_state)

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
        Xk = Fk['xf']
        J += Fk['qf']

    for i in range(nsamples):
      idx = np.arange(nstates // nsamples)+i*(nstates // nsamples)
      # TODO: Improve implementation of averaging
      J += objective_function(Xk[idx])/nsamples

    objective = J
    constraints = vertcat(*g)
    decisions = vertcat(*w)

    return objective, constraints, decisions, w0, lbw, ubw, lbg, ubg
