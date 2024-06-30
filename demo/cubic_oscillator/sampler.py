import numpy as np
from casadi import *

class Sampler(object):

    def __init__(self, nreplications=1):

        entropy = 0x3034c61a9ae04ff8cb62ab8ec2c4b501
        rng = np.random.default_rng(entropy)
        self.rngs = rng.spawn(nreplications)

    def sample(self, replication, nsamples, nparams):

        return 2*np.pi*self.rngs[replication].uniform(0, 1.0, (nsamples, nparams))


if __name__ == "__main__":

    sampler = Sampler(nreplications=2)
    sample = sampler.sample(0, 4, 3)
    print(sample)
    print(len(sample))

