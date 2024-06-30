from scipy.stats import qmc
from casadi import *

class ReferenceSampler(object):

    def sample(self, nsamples, nparams):

        n = nsamples
        if not (n & (n-1) == 0) and n != 0:
            raise ValueError("Sample size should be a power of 2.")

        sampler = qmc.Sobol(d=nparams, scramble=False)

        m = int(np.log2(nsamples))
        return 2*np.psampler.random_base2(m=m)


if __name__ == "__main__":

    reference_sampler = ReferenceSampler()
    sample = reference_sampler.sample(4, 3)
    print(sample)
    print(len(sample))
