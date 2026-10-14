"""Statistical modelling of packet-loss patterns — full pipeline.

Run:  python3 run_case_study.py
Outputs go to ./results (figures, results.json, summary.md).
"""
import json
import os
import time

import numpy as np

from src import config as C
from src import loss_model as LM
from src import distributions as D
from src import hypothesis_tests as HT
from src import bayesian as BY
from src import plots as P


def log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main():
    t0 = time.time()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    n_packets = C.PACKET_RATE * C.DURATION_S
    ppw = C.PACKET_RATE * C.WINDOW_S
    labels = {k: v["label"] for k, v in C.SCENARIOS.items()}
    params = {k: {kk: vv for kk, vv in v.items() if kk != "label"} for k, v in C.SCENARIOS.items()}

    # ---------------- 1. traffic traces ----------------
    log("Generating 1-hour packet traces for 3 network scenarios", t0)
    losses, counts, gaps = {}, {}, {}
    for k, prm in params.items():
        losses[k] = LM.simulate_losses(n_packets, rng=rng, **prm)
        counts[k] = LM.window_counts(losses[k], ppw)
        gaps[k] = LM.inter_loss_gaps(losses[k], C.PACKET_RATE, rng)

    R = {"config": {"packet_rate": C.PACKET_RATE, "duration_s": C.DURATION_S, "seed": C.SEED},
         "scenarios": {}}

    # ---------------- 2. descriptive + distribution fitting ----------------
    log("Descriptive statistics and MLE fits (Poisson, NegBin, Exponential, Hyperexponential)", t0)
    fits, gap_fits = {}, {}
    for k in params:
        l, c, g = losses[k], counts[k], gaps[k]
        bl = LM.burst_lengths(l)
        fits[k] = {"poisson": D.fit_poisson(c), "negbin": D.fit_negbin(c)}
        gap_fits[k] = {"exp": D.fit_exponential(g), "hyper": D.fit_hyperexp(g)}
        R["scenarios"][k] = {
            "label": labels[k],
            "true_long_run_loss_rate": LM.theoretical_loss_rate(**params[k]),
            "descriptive": dict(packets=int(len(l)), losses=int(l.sum()), loss_rate=float(l.mean()),
                                window_mean=float(c.mean()), window_var=float(c.var(ddof=1)),
                                max_losses_in_window=int(c.max()),
                                mean_gap_s=float(g.mean()), median_gap_s=float(np.median(g)),
                                mean_burst_len=float(bl.mean()), max_burst_len=int(bl.max()),
                                share_losses_in_bursts=float(bl[bl > 1].sum() / bl.sum())),
            "fits": {
                "poisson_lambda": fits[k]["poisson"]["lam"],
                "negbin_r": fits[k]["negbin"]["r"], "negbin_p": fits[k]["negbin"]["p"],
                "aic_poisson": D.aic(fits[k]["poisson"]), "aic_negbin": D.aic(fits[k]["negbin"]),
                "exp_rate": gap_fits[k]["exp"]["rate"],
                "hyper_w_fast": gap_fits[k]["hyper"]["w"],
                "hyper_rate_fast": gap_fits[k]["hyper"]["rate_fast"],
                "hyper_rate_slow": gap_fits[k]["hyper"]["rate_slow"],
                "aic_exp": D.aic(gap_fits[k]["exp"]), "aic_hyper": D.aic(gap_fits[k]["hyper"]),
            },
        }

    # ---------------- 3. hypothesis tests ----------------
    log("Hypothesis tests: chi-square GOF, dispersion, runs, Markov LRT", t0)
    from scipy import stats
    for k in params:
        c, l = counts[k], losses[k]
        lam = fits[k]["poisson"]["lam"]; nb = fits[k]["negbin"]
        R["scenarios"][k]["tests"] = {
            "chisq_poisson": HT.chisq_gof_counts(c, lambda x: stats.poisson.pmf(x, lam), 1),
            "chisq_negbin": HT.chisq_gof_counts(c, lambda x: stats.nbinom.pmf(x, nb["r"], nb["p"]), 2),
            "dispersion": HT.dispersion_test(c),
            "runs": HT.runs_test(l),
            "markov_lrt": HT.markov_lrt(l),
        }

    log(f"Parametric-bootstrap KS tests ({C.N_BOOT_KS} reps exp, {C.N_BOOT_KS_MIX} reps mixture) — slowest step", t0)
    for k in params:
        R["scenarios"][k]["tests"]["ks_exponential"] = HT.ks_exponential_bootstrap(gaps[k], C.N_BOOT_KS, rng)
        R["scenarios"][k]["tests"]["ks_hyperexp"] = HT.ks_hyperexp_bootstrap(
            gaps[k], gap_fits[k]["hyper"], C.N_BOOT_KS_MIX, rng)
        log(f"   KS done for {labels[k]}", t0)

    R["comparison"] = HT.compare_scenarios(counts, losses)

    # ---------------- 4. Bayesian inference ----------------
    log("Bayesian inference: Beta posteriors, ordering, Markov posteriors, predictive check", t0)
    posts, mposts = {}, {}
    for k in params:
        l = losses[k]
        posts[k] = BY.beta_posterior(int(l.sum()), len(l))
        mposts[k] = BY.markov_posterior(l, 100_000, rng)
        R["scenarios"][k]["bayes"] = {
            "posterior": posts[k],
            "adjusted_posterior": BY.adjusted_posterior(l, C.BLOCK_S * C.PACKET_RATE),
            "markov": {kk: vv for kk, vv in mposts[k].items() if not kk.startswith("samples")},
            "predictive_tail": BY.posterior_predictive_tail(posts[k], ppw, 5, counts[k]),
        }
    seq = BY.sequential_posteriors(losses["S2_WiFi"], C.PACKET_RATE, [10, 60, 600, 3600])
    R["sequential_S2"] = seq
    R["ordering"] = BY.prob_ordering(posts, list(params), 200_000, rng)

    # ---------------- 5. Monte Carlo studies ----------------
    log(f"Monte Carlo power study ({C.POWER_REPS} reps x {len(C.POWER_DURATIONS_S)} lengths x 3 networks)", t0)
    power = {}
    for k, prm in params.items():
        power[k] = HT.power_study(prm, C.POWER_DURATIONS_S, C.POWER_REPS, C.PACKET_RATE, C.ALPHA, rng,
                                  LM.simulate_losses)
        log(f"   power done for {labels[k]}", t0)
    R["power"] = {"durations_s": C.POWER_DURATIONS_S, **power}

    log(f"Monte Carlo coverage study ({C.COVERAGE_REPS} reps x 3 networks)", t0)
    cov = {}
    for k, prm in params.items():
        cov[k] = BY.coverage_study(prm, LM.theoretical_loss_rate(**prm), C.COVERAGE_DURATION_S,
                                   C.COVERAGE_REPS, C.PACKET_RATE, C.BLOCK_S, rng, LM.simulate_losses)
    R["coverage"] = cov

    # ---------------- 6. figures ----------------
    log("Drawing figures", t0)
    d = C.RESULTS_DIR
    P.timeline(counts, labels, f"{d}/fig1_loss_timeline.png")
    P.count_fits(counts, fits, labels, f"{d}/fig2_count_distribution_fits.png")
    P.gap_survival(gaps, gap_fits, labels, f"{d}/fig3_gap_survival.png")
    P.posteriors(posts, seq, labels, f"{d}/fig4_bayesian_posteriors.png")
    P.markov_posteriors(mposts, labels, f"{d}/fig5_markov_posteriors.png")
    P.power_curves(power, C.POWER_DURATIONS_S, labels, f"{d}/fig6_power_curves.png")
    P.coverage_bar(cov, labels, f"{d}/fig7_interval_coverage.png")

    # ---------------- 7. save ----------------
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    R = clean(R)
    with open(f"{d}/results.json", "w") as f:
        json.dump(R, f, indent=2)
    write_summary(R, labels, f"{d}/summary.md")
    log(f"DONE. Results in ./{d}/", t0)


def fmt_p(p, B=None):
    if B is not None and p <= 1 / (B + 1) + 1e-12:
        return f"< {1 / B:.3g}"
    return "< 1e-10" if p < 1e-10 else f"{p:.3g}"


def write_summary(R, labels, path):
    L = ["# Results summary — packet-loss case study", ""]
    S = R["scenarios"]
    L += ["## 1. Descriptive statistics (1 hour, 50 packets/s, 180,000 packets per network)", "",
          "| Network | Loss rate | Mean losses/s | Variance | Var/Mean | Mean burst | Max burst | % losses in bursts |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in S.items():
        d = s["descriptive"]
        L.append(f"| {labels[k]} | {d['loss_rate']*100:.2f}% | {d['window_mean']:.3f} | {d['window_var']:.3f} | "
                 f"{d['window_var']/d['window_mean']:.2f} | {d['mean_burst_len']:.2f} | {d['max_burst_len']} | "
                 f"{d['share_losses_in_bursts']*100:.1f}% |")
    L += ["", "## 2. Distribution fitting (MLE) and goodness of fit", "",
          "| Network | Poisson λ | χ² Poisson (p) | NegBin r | χ² NegBin (p) | AIC Pois | AIC NB | Dispersion D (p) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in S.items():
        f, t = s["fits"], s["tests"]
        L.append(f"| {labels[k]} | {f['poisson_lambda']:.3f} | {t['chisq_poisson']['stat']:.1f} ({fmt_p(t['chisq_poisson']['p'])}) | "
                 f"{f['negbin_r']:.3f} | {t['chisq_negbin']['stat']:.1f} ({fmt_p(t['chisq_negbin']['p'])}) | "
                 f"{f['aic_poisson']:.0f} | {f['aic_negbin']:.0f} | {t['dispersion']['D']:.2f} ({fmt_p(t['dispersion']['p'])}) |")
    L += ["", "| Network | Exp rate (1/s) | KS exp D (p) | Hyperexp w_fast | fast rate | slow rate | KS hyperexp D (p) | AIC exp | AIC hyper |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, s in S.items():
        f, t = s["fits"], s["tests"]
        L.append(f"| {labels[k]} | {f['exp_rate']:.3f} | {t['ks_exponential']['D']:.3f} ({fmt_p(t['ks_exponential']['p'], t['ks_exponential']['B'])}) | "
                 f"{f['hyper_w_fast']:.2f} | {f['hyper_rate_fast']:.1f} | {f['hyper_rate_slow']:.3f} | "
                 f"{t['ks_hyperexp']['D']:.3f} ({fmt_p(t['ks_hyperexp']['p'], t['ks_hyperexp']['B'])}) | {f['aic_exp']:.0f} | {f['aic_hyper']:.0f} |")
    L += ["", "## 3. Independence tests", "",
          "| Network | Runs observed | Runs expected | z | p | P(L given N) | P(L given L) | G² | p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, s in S.items():
        r, m = s["tests"]["runs"], s["tests"]["markov_lrt"]
        L.append(f"| {labels[k]} | {r['runs']} | {r['expected']:.0f} | {r['z']:.2f} | {fmt_p(r['p'])} | "
                 f"{m['p_L_given_N']*100:.3f}% | {m['p_L_given_L']*100:.1f}% | {m['G2']:.1f} | {fmt_p(m['p'])} |")
    c = R["comparison"]
    L += ["", f"**Comparing networks:** Kruskal–Wallis H = {c['kruskal_H']:.1f}, p = {fmt_p(c['kruskal_p'])}; "
          f"χ² homogeneity of loss proportions = {c['chi2_homog']:.1f} (df {c['chi2_df']}), p = {fmt_p(c['chi2_p'])}.", ""]
    for pr in c["pairwise"]:
        L.append(f"- Mann–Whitney {labels[pr['a']]} vs {labels[pr['b']]}: Bonferroni p = {fmt_p(pr['p_bonf'])}")
    L += ["", "## 4. Bayesian inference (Beta(1,1) prior)", "",
          "| Network | Posterior mean | 95% credible interval | Design effect | Adjusted 95% interval | P(L|L) > P(L|N) | Burst amplification b/a (95%) | Mean burst length (95%) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in S.items():
        b = s["bayes"]; p, a, m = b["posterior"], b["adjusted_posterior"], b["markov"]
        L.append(f"| {labels[k]} | {p['mean']*100:.3f}% | [{p['ci_low']*100:.3f}%, {p['ci_high']*100:.3f}%] | {a['deff']:.2f} | "
                 f"[{a['ci_low']*100:.3f}%, {a['ci_high']*100:.3f}%] | {m['P_b_gt_a']:.4f} | "
                 f"{m['amp_q'][1]:.0f} [{m['amp_q'][0]:.0f}, {m['amp_q'][2]:.0f}] | "
                 f"{m['burst_len_q'][1]:.2f} [{m['burst_len_q'][0]:.2f}, {m['burst_len_q'][2]:.2f}] |")
    L += ["", "Posterior predictive check — P(≥5 losses in a 1-s window):", ""]
    for k, s in S.items():
        t = s["bayes"]["predictive_tail"]
        L.append(f"- {labels[k]}: i.i.d. model predicts {t['predicted']:.2e}, observed share {t['observed']:.2e}")
    L += ["", "Probability ordering of loss rates: " +
          ", ".join(f"{k} = {v:.4f}" for k, v in R["ordering"].items()), "",
          "Sequential updating (Campus Wi-Fi): " +
          "; ".join(f"after {s['t']} s: mean {s['mean']*100:.2f}%, 95% CI [{s['ci_low']*100:.2f}%, {s['ci_high']*100:.2f}%]"
                    for s in R["sequential_S2"]), ""]
    L += ["## 5. Monte Carlo studies", "", "Power to reject independence (α = 0.05):", "",
          "| Network | Test | " + " | ".join(f"{t}s" for t in R["power"]["durations_s"]) + " |",
          "|---|---|" + "---|" * len(R["power"]["durations_s"])]
    for k in S:
        for test in ["dispersion", "runs", "markov_lrt"]:
            L.append(f"| {labels[k]} | {test} | " + " | ".join(f"{v:.2f}" for v in R["power"][k][test]) + " |")
    L += ["", "Coverage of nominal 95% intervals for the true loss rate (10-minute captures):", "",
          "| Network | Naive coverage | Adjusted coverage | Naive width | Adjusted width |", "|---|---|---|---|---|"]
    for k, cv in R["coverage"].items():
        L.append(f"| {labels[k]} | {cv['naive']*100:.1f}% | {cv['adjusted']*100:.1f}% | "
                 f"{cv['width_naive']*100:.3f} pp | {cv['width_adjusted']*100:.3f} pp |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
