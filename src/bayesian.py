"""Bayesian inference for packet-loss probability (owner: Member C)."""
import numpy as np
from scipy import stats

from .hypothesis_tests import transition_counts

PRIOR_A, PRIOR_B = 1.0, 1.0   # Beta(1,1) = uniform prior on the loss probability


def beta_posterior(k, n, a=PRIOR_A, b=PRIOR_B):
    """Conjugate update: Beta(a,b) prior + k losses in n packets -> Beta(a+k, b+n-k)."""
    A, B = a + k, b + n - k
    lo, hi = stats.beta.ppf([0.025, 0.975], A, B)
    return dict(alpha=A, beta=B, mean=A / (A + B), ci_low=lo, ci_high=hi)


def sequential_posteriors(loss, packet_rate, checkpoints_s):
    out = []
    for t in checkpoints_s:
        n = int(t * packet_rate)
        out.append(dict(t=t, **beta_posterior(int(loss[:n].sum()), n)))
    return out


def prob_ordering(posts, names, draws, rng):
    """Monte Carlo estimate of P(p_1 < p_2 < ... ) and pairwise P(p_i < p_j)."""
    s = {k: rng.beta(posts[k]["alpha"], posts[k]["beta"], draws) for k in names}
    res = {"all_ordered": float(np.mean(np.all([s[names[i]] < s[names[i + 1]]
                                                for i in range(len(names) - 1)], axis=0)))}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            res[f"P({names[i]} < {names[j]})"] = float(np.mean(s[names[i]] < s[names[j]]))
    return res


def markov_posterior(loss, draws, rng):
    """Independent Beta posteriors for a = P(L|N) and b = P(L|L) (Gilbert model).
    Burst amplification = b / a; expected burst length = 1 / (1 - b)."""
    n_nn, n_nl, n_ln, n_ll = transition_counts(loss)
    a = rng.beta(1 + n_nl, 1 + n_nn, draws)
    b = rng.beta(1 + n_ll, 1 + n_ln, draws)
    ratio = b / a
    blen = 1 / (1 - b)
    q = lambda v: [float(x) for x in np.quantile(v, [0.025, 0.5, 0.975])]
    return dict(P_b_gt_a=float(np.mean(b > a)), amp_q=q(ratio), burst_len_q=q(blen),
                a_q=q(a), b_q=q(b), samples_a=a, samples_b=b)


def posterior_predictive_tail(post, window_n, k_thresh, observed_counts):
    """Under the i.i.d. Beta-Binomial predictive, P(X >= k) in a window, vs the observed share."""
    pred = stats.betabinom.sf(k_thresh - 1, window_n, post["alpha"], post["beta"])
    emp = float(np.mean(observed_counts >= k_thresh))
    return dict(k=k_thresh, predicted=float(pred), observed=emp)


def design_effect(loss, block_packets):
    """Variance inflation of block loss counts relative to binomial, floored at 1."""
    m = len(loss) // block_packets
    c = loss[: m * block_packets].reshape(m, block_packets).sum(axis=1)
    p = loss.mean()
    if p <= 0 or p >= 1 or m < 2:
        return 1.0
    return max(1.0, c.var(ddof=1) / (block_packets * p * (1 - p)))


def adjusted_posterior(loss, block_packets):
    """Effective-sample-size correction: divide k and n by the design effect,
    which widens the credible interval to reflect correlated (bursty) losses."""
    deff = design_effect(loss, block_packets)
    k, n = loss.sum(), len(loss)
    return beta_posterior(k / deff, n / deff) | {"deff": deff}


def coverage_study(params, true_p, duration_s, reps, packet_rate, block_s, rng, simulate):
    """How often does a nominal 95% interval contain the true long-run loss rate?"""
    n = duration_s * packet_rate
    hit_naive = hit_adj = 0
    w_naive = w_adj = 0.0
    for _ in range(reps):
        loss = simulate(n, rng=rng, **params)
        pn = beta_posterior(int(loss.sum()), n)
        pa = adjusted_posterior(loss, block_s * packet_rate)
        hit_naive += pn["ci_low"] <= true_p <= pn["ci_high"]
        hit_adj += pa["ci_low"] <= true_p <= pa["ci_high"]
        w_naive += pn["ci_high"] - pn["ci_low"]
        w_adj += pa["ci_high"] - pa["ci_low"]
    return dict(naive=hit_naive / reps, adjusted=hit_adj / reps,
                width_naive=w_naive / reps, width_adjusted=w_adj / reps)
