"""Shared reference configuration for the harmonic-oscillator studies.

Single source of truth for the parameters that fix the reference optimal value
J*_{N_ref} -- the large-sample SAA proxy for the population optimum J*.  The CLT
study (clt_harmonic_oscillator.py) and the coverage study
(coverage_harmonic_oscillator.py) both proxy J* by this same value, so it MUST be
computed identically in both: same root seed, same reference sample size, and
same reference solve tolerance.  Keeping these three here stops the two drivers
from drifting apart (they previously used N_ref = 1024 vs 4096 and tol 1e-8 vs
1e-5, which made their J_hat_ref* disagree).

Only the REFERENCE knobs live here.  Each study keeps its own replicate settings
(R, the fast 1e-5 replicate tolerance in coverage, etc.), since those do not
affect J_hat_ref*.
"""

REF_SEED = 12345      # root entropy; the reference stream is spawn(1+R)[0], which
                      # numpy SeedSequence makes independent of R, so both studies
                      # draw the identical reference sample despite different R.
N_REF = 4096          # reference sample size (proxies J*); larger => lower bias.
REF_TOL = 1e-8        # reference solve tolerance (tight: this is the "truth" proxy,
                      # not one of the many cheap replicate solves).
