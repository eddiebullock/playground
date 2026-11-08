"""Matrix-level check: one contrast, n actors, effect = (g - E g) + small noise, frozen
signflip_maxt (exact flips), both tails; and a studentized bootstrap-t for comparison."""
import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from scripts import stimulus_interventions as si

def boot_t_p(x, direction, B=1999, rng=None):
    n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    t = m / se
    idx = rng.integers(0, n, (B, n)); xb = x[idx]
    tb = (xb.mean(1) - m) / (xb.std(1, ddof=1) / np.sqrt(n) + 1e-12)
    return ((direction * tb >= direction * t).sum() + 1) / (B + 1)

rng = np.random.default_rng(7)
R = 4000
for n in (8, 17):
    for sd in (0.0, 0.3, 0.5):
        m = np.exp(sd**2 / 2)
        rej = {+1: 0, -1: 0}; bt = {+1: 0, -1: 0}
        for r in range(R):
            g = np.exp(rng.normal(0, sd, n))
            x = 2 * (g - m) + rng.normal(0, 0.15, n)
            for d in (+1, -1):
                p = si.signflip_maxt(x[:, None], alternative=[d])["p_raw"][0]
                rej[d] += p < .05
                if r < 1500:
                    bt[d] += boot_t_p(x, d, rng=rng) < .05
        print(f"n={n} gainSD={sd}: signflip upper(long-tail side)={rej[1]/R:.3f} lower={rej[-1]/R:.3f} | boot-t upper={bt[1]/1500:.3f} lower={bt[-1]/1500:.3f}", flush=True)
