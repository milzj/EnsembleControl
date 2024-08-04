from casadi import *

def risk_measure(J, w, w0, lbw, ubw, g, lbg, ubg, final_cost_function, Xk, xk, beta, nstates, nsamples):
    if beta > 0 and beta < 1.0:
        # TODO: Use a priori bounds on t for initial value
        t = MX.sym('t', 1)
        w += [t]
        lbw += [-inf]
        ubw += [inf]
        t0 = 0.0
        w0 += [t0]
        J += t 
    
    
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

    return J, w, lbw, ubw, g, lbg, ubg