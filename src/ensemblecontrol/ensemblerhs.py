from casadi import *

def EnsembleRHS(parameterized_rhs,
                objective,
                control,
                parameterized_initial_value,
                samples, nstates):

    # ensemble right hand side
    ensemble_initial_value = []
    Ydot = []
    nsamples = len(samples)
    Y = MX.sym('Y', nstates*nsamples)

    for i in range(nsamples):
      params = samples[i]
      idx = np.arange(nstates)+i*nstates
      ydot = parameterized_rhs(Y[idx], control, params)
      Ydot = vertcat(Ydot, ydot)
      ensemble_initial_value += parameterized_initial_value(params)

    EnsembleRHS = Function("ensembleRHS", [Y, control], [Ydot, objective],
                           ["Y", "u"], ["rhs", "objective"])

    return EnsembleRHS, ensemble_initial_value
