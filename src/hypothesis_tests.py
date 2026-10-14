"""Hypothesis-testing suite (owner: Member B)."""
import numpy as np
from scipy import stats

from .distributions import fit_exponential, fit_hyperexp, hyperexp_cdf


# ---------- goodness of fit for counts ----------
def chisq_gof_counts(x, pmf, n_params):
    """Pearson chi-square GOF; the upper tail is pooled so every expected count is >= 5."""
    n = len(x)
    kmax = int(x.max())
    obs = np.bincount(x, minlength=kmax + 1).astype(float)
    exp_ = n * pmf(np.arange(kmax + 1))
    exp_[-1] += n - exp_.sum()           # put the remaining tail mass in the last bin
    # pool from the right until each bin has expected >= 5
    o, e = list(obs), list(exp_)
    while len(e) > 2 and e[-1] < 5:
        last_e, last_o = e.pop(), o.pop()
        e[-1] += last_e
        o[-1] += last_o
    o, e = np.array(o), np.array(e)
    stat = ((o - e) ** 2 / e).sum()
    df = len(e) - 1 - n_params
    return dict(stat=stat, df=df, p=stats.chi2.sf(stat, df), bins=len(e))


def dispersion_test(x):
    """Index-of-dispersion test. H0: Poisson (var = mean). Stat = (n-1) s^2 / xbar ~ chi2(n-1)."""
    n, m, v = len(x), x.mean(), x.var(ddof=1)
    stat = (n - 1) * v / m
    return dict(D=v / m, stat=stat, df=n - 1, p=stats.chi2.sf(stat, n - 1))


# ---------- goodness of fit for gaps ----------
def ks_exponential_bootstrap(g, B, rng):
    """Lilliefors-type KS test for exponentiality: the rate is estimated, so the
    null distribution of D is obtained by parametric bootstrap."""
    scale = g.mean()
    d_obs = stats.kstest(g, "expon", args=(0, scale)).statistic
    n, count = len(g), 0
    for _ in range(B):
        s = rng.exponential(scale, n)
        if stats.kstest(s, "expon", args=(0, s.mean())).statistic >= d_obs:
            count += 1
    return dict(D=d_obs, p=(count + 1) / (B + 1), B=B)


def ks_hyperexp_bootstrap(g, fit, B, rng):
    cdf = lambda t, f: hyperexp_cdf(t, f["w"], f["rate_fast"], f["rate_slow"])
    d_obs = stats.kstest(g, lambda t: cdf(t, fit)).statistic
    n, count = len(g), 0
    for _ in range(B):
        comp = rng.random(n) < fit["w"]
        s = np.where(comp, rng.exponential(1 / fit["rate_fast"], n),
                     rng.exponential(1 / fit["rate_slow"], n))
        f2 = fit_hyperexp(s, iters=150)
        if stats.kstest(s, lambda t: cdf(t, f2)).statistic >= d_obs:
            count += 1
    return dict(D=d_obs, p=(count + 1) / (B + 1), B=B)


# ---------- independence of losses ----------
def runs_test(loss):
    """Wald-Wolfowitz runs test on the 0/1 loss sequence. Too FEW runs => clustering."""
    x = loss.astype(np.int8)
    n1 = int(x.sum()); n0 = len(x) - n1; n = n1 + n0
    runs = 1 + int(np.count_nonzero(np.diff(x)))
    mu = 2 * n1 * n0 / n + 1
    var = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n ** 2 * (n - 1))
    z = (runs - mu) / np.sqrt(var)
    return dict(runs=runs, expected=mu, z=z, p=2 * stats.norm.sf(abs(z)))


def transition_counts(loss):
    x = loss.astype(np.int8)
    prev, nxt = x[:-1], x[1:]
    n_nn = int(np.sum((prev == 0) & (nxt == 0)))
    n_nl = int(np.sum((prev == 0) & (nxt == 1)))
    n_ln = int(np.sum((prev == 1) & (nxt == 0)))
    n_ll = int(np.sum((prev == 1) & (nxt == 1)))
    return n_nn, n_nl, n_ln, n_ll


def markov_lrt(loss):
    """Likelihood-ratio test. H0: losses are i.i.d. Bernoulli(p).
    H1: first-order Markov chain (Gilbert model) with P(L|N)=a, P(L|L)=b.
    G^2 = 2 (ll_H1 - ll_H0) ~ chi2(1)."""
    n_nn, n_nl, n_ln, n_ll = transition_counts(loss)
    xlogy = lambda k, p: k * np.log(p) if k > 0 else 0.0
    n_l = n_nl + n_ll; n = n_nn + n_nl + n_ln + n_ll
    p = n_l / n
    ll0 = xlogy(n_l, p) + xlogy(n - n_l, 1 - p)
    a = n_nl / (n_nn + n_nl); b = n_ll / (n_ln + n_ll)
    ll1 = xlogy(n_nl, a) + xlogy(n_nn, 1 - a) + xlogy(n_ll, b) + xlogy(n_ln, 1 - b)
    g2 = 2 * (ll1 - ll0)
    return dict(p_L_given_N=a, p_L_given_L=b, G2=g2, df=1, p=stats.chi2.sf(g2, 1))


# ---------- comparing networks ----------
def compare_scenarios(counts_by_scn, losses_by_scn):
    names = list(counts_by_scn)
    kw = stats.kruskal(*[counts_by_scn[k] for k in names])
    table = np.array([[l.sum(), len(l) - l.sum()] for l in (losses_by_scn[k] for k in names)])
    chi2, p_chi, dof, _ = stats.chi2_contingency(table)
    pairs = []
    m = len(names) * (len(names) - 1) // 2
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            u = stats.mannwhitneyu(counts_by_scn[names[i]], counts_by_scn[names[j]],
                                   alternative="two-sided")
            pairs.append(dict(a=names[i], b=names[j], U=float(u.statistic),
                              p_bonf=min(1.0, u.pvalue * m)))
    return dict(kruskal_H=kw.statistic, kruskal_p=kw.pvalue,
                chi2_homog=chi2, chi2_df=dof, chi2_p=p_chi, pairwise=pairs)


# ---------- Monte Carlo power study ----------
def power_study(params, durations_s, reps, packet_rate, alpha, rng, simulate):
    """Probability that each test rejects its null (independence) for a capture of a given length."""
    out = {"dispersion": [], "runs": [], "markov_lrt": []}
    for T in durations_s:
        n = T * packet_rate
        rej = np.zeros(3)
        valid = 0
        for _ in range(reps):
            loss = simulate(n, rng=rng, **params)
            if loss.sum() < 2:
                continue
            valid += 1
            c = loss.reshape(-1, packet_rate).sum(axis=1)
            if c.mean() > 0 and dispersion_test(c)["p"] < alpha:
                rej[0] += 1
            if runs_test(loss)["p"] < alpha:
                rej[1] += 1
            if markov_lrt(loss)["p"] < alpha:
                rej[2] += 1
        # captures with < 2 losses count as non-rejections
        for key, r in zip(out, rej):
            out[key].append(r / reps)
    return out
