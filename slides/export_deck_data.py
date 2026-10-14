"""Collect every number the Review-1 slide deck shows into slides/deck_data.json.

Run after run_case_study.py, from the project folder:
    python3 slides/export_deck_data.py

Nothing here is typed in by hand: values come from src/config.py, results/results.json,
or from re-simulating the same traces with the same seed.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C                      # noqa: E402
from src import loss_model as LM                 # noqa: E402
from src import distributions as D               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TITLE_SCENARIO = "S3_Mobile"   # trace shown on the title slide
TITLE_SECONDS = 120
COMPARE_SECONDS = 60           # length of the independent-vs-bursty comparison
COMPARE_SEED = 7               # seed for the independent comparison stream
PMF_MAX_K = 12                 # largest count drawn in the count-distribution chart
SURV_GRID_S = [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 10]


def main():
    R = json.load(open(os.path.join(ROOT, C.RESULTS_DIR, "results.json")))
    keys = list(C.SCENARIOS)
    params = {k: {kk: vv for kk, vv in v.items() if kk != "label"} for k, v in C.SCENARIOS.items()}

    # Re-create the exact traces of run_case_study.py (same seed, same order of random draws).
    rng = np.random.default_rng(C.SEED)
    n = C.PACKET_RATE * C.DURATION_S
    ppw = C.PACKET_RATE * C.WINDOW_S
    losses, counts, gaps = {}, {}, {}
    for k, p in params.items():
        losses[k] = LM.simulate_losses(n, rng=rng, **p)
        counts[k] = LM.window_counts(losses[k], ppw)
        gaps[k] = LM.inter_loss_gaps(losses[k], C.PACKET_RATE, rng)

    # sanity check: the re-created traces match the saved results
    for k in keys:
        assert int(losses[k].sum()) == R["scenarios"][k]["descriptive"]["losses"], "trace mismatch"

    seq_k = R["sequential"]["scenario"]
    out = {
        "config": dict(seed=C.SEED, packet_rate=C.PACKET_RATE, duration_s=C.DURATION_S, window_s=C.WINDOW_S,
                       alpha=C.ALPHA, block_s=C.BLOCK_S, power_reps=C.POWER_REPS, coverage_reps=C.COVERAGE_REPS,
                       coverage_duration_s=C.COVERAGE_DURATION_S, n_boot_ks=C.N_BOOT_KS,
                       ordering_draws=C.ORDERING_DRAWS, heavy_k=C.HEAVY_LOSS_K, packets=n),
        "keys": keys,
        "scenarios": {},
        "series": {},
    }

    for k in keys:
        p = params[k]
        f = R["scenarios"][k]["fits"]
        c = counts[k]
        hk = C.HEAVY_LOSS_K
        out["scenarios"][k] = dict(
            label=C.SCENARIOS[k]["label"],
            p_gb=p["p_gb"], p_bg=p["p_bg"], h_g=p["h_g"], h_b=p["h_b"],
            pi_b=LM.stationary_bad_prob(p["p_gb"], p["p_bg"]),
            mean_bad_spell=1 / p["p_bg"],
            heavy_seconds=dict(
                observed=int(np.sum(c >= hk)),
                poisson=float(len(c) * stats.poisson.sf(hk - 1, f["poisson_lambda"])),
                negbin=float(len(c) * stats.nbinom.sf(hk - 1, f["negbin_r"], f["negbin_p"])),
            ),
            max_losses_in_window=int(c.max()),
        )

    # title slide trace
    out["series"]["title"] = dict(scenario=TITLE_SCENARIO, seconds=TITLE_SECONDS,
                                  counts=counts[TITLE_SCENARIO][:TITLE_SECONDS].tolist())

    # same long-run rate, independent vs bursty
    rate = float(losses[seq_k].mean())
    r2 = np.random.default_rng(COMPARE_SEED)
    iid = (r2.random(COMPARE_SECONDS * ppw) < rate).reshape(COMPARE_SECONDS, ppw).sum(1)
    out["series"]["compare"] = dict(scenario=seq_k, seconds=COMPARE_SECONDS, rate=rate,
                                    iid=iid.tolist(), bursty=counts[seq_k][:COMPARE_SECONDS].tolist())

    # count distribution for the sequential-updating scenario
    c = counts[seq_k]
    ks = np.arange(PMF_MAX_K + 1)
    f = R["scenarios"][seq_k]["fits"]
    out["series"]["pmf"] = dict(
        scenario=seq_k, k=ks.tolist(),
        observed=(np.bincount(c, minlength=PMF_MAX_K + 1)[: PMF_MAX_K + 1] / len(c)).tolist(),
        poisson=stats.poisson.pmf(ks, f["poisson_lambda"]).tolist(),
        negbin=stats.nbinom.pmf(ks, f["negbin_r"], f["negbin_p"]).tolist(),
    )

    # gap survival for the same scenario
    g = gaps[seq_k]
    t = np.array(SURV_GRID_S)
    w, l1, l2 = f["hyper_w_fast"], f["hyper_rate_fast"], f["hyper_rate_slow"]
    out["series"]["survival"] = dict(
        scenario=seq_k, t=t.tolist(),
        observed=[float(np.mean(g > x)) for x in t],
        exponential=np.exp(-f["exp_rate"] * t).tolist(),
        hyperexp=(w * np.exp(-l1 * t) + (1 - w) * np.exp(-l2 * t)).tolist(),
    )

    out["results"] = R
    with open(os.path.join(HERE, "deck_data.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote slides/deck_data.json")


if __name__ == "__main__":
    main()
