"""Maximum-likelihood fitting of candidate distributions (owner: Member A)."""
import numpy as np
from scipy import stats, optimize


# ---------- counts per window ----------
def fit_poisson(x):
    lam = x.mean()
    ll = stats.poisson.logpmf(x, lam).sum()
    return dict(lam=lam, loglik=ll, k=1)


def fit_negbin(x):
    """Profile MLE for the negative binomial (r, p) with mean fixed at the sample mean."""
    mean = x.mean()

    def nll(log_r):
        r = np.exp(log_r)
        p = r / (r + mean)
        return -stats.nbinom.logpmf(x, r, p).sum()

    res = optimize.minimize_scalar(nll, bounds=(-8, 8), method="bounded")
    r = float(np.exp(res.x))
    p = r / (r + mean)
    return dict(r=r, p=p, loglik=-res.fun, k=2)


def aic(fit):
    return 2 * fit["k"] - 2 * fit["loglik"]


# ---------- inter-loss gaps ----------
def fit_exponential(g):
    lam = 1 / g.mean()
    return dict(rate=lam, loglik=stats.expon.logpdf(g, scale=1 / lam).sum(), k=1)


def hyperexp_cdf(t, w, l1, l2):
    return w * (1 - np.exp(-l1 * t)) + (1 - w) * (1 - np.exp(-l2 * t))


def fit_hyperexp(g, iters=300, tol=1e-9):
    """EM for a 2-phase hyperexponential: f(t) = w*l1*e^{-l1 t} + (1-w)*l2*e^{-l2 t}.

    Phase 1 (fast rate) = gaps inside a loss burst; phase 2 (slow rate) = gaps between bursts.
    """
    q = np.quantile(g, [0.25, 0.75])
    w, l1, l2 = 0.5, 1 / max(q[0], 1e-6), 1 / q[1]
    prev = -np.inf
    for _ in range(iters):
        a = w * l1 * np.exp(-l1 * g)
        b = (1 - w) * l2 * np.exp(-l2 * g)
        tot = a + b + 1e-300
        z = a / tot
        ll = np.log(tot).sum()
        w = z.mean()
        l1 = z.sum() / (z * g).sum()
        l2 = (1 - z).sum() / ((1 - z) * g).sum()
        if ll - prev < tol:
            break
        prev = ll
    if l1 < l2:  # keep phase 1 as the fast (burst) phase
        w, l1, l2 = 1 - w, l2, l1
    return dict(w=w, rate_fast=l1, rate_slow=l2, loglik=ll, k=3)
