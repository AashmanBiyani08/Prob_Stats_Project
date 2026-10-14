"""Checks for the packet-loss case study.

Run from the project folder:   python3 -m pytest tests -q

Three groups:
  1. edge cases that used to crash or return NaN silently;
  2. statistical checks against known answers (recovery, Type I error, coverage);
  3. the full pipeline under different settings.
"""
import contextlib
import copy
import io
import json
import os
import sys

import numpy as np
import pytest
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import bayesian as BY                     # noqa: E402
from src import config as C                        # noqa: E402
from src import distributions as D                 # noqa: E402
from src import hypothesis_tests as HT             # noqa: E402
from src import loss_model as LM                   # noqa: E402

GE = dict(p_gb=0.002, p_bg=0.15, h_g=0.004, h_b=0.6)


# ---------------------------------------------------------------- 1. edge cases
@pytest.mark.parametrize("n", [0, 1, 2, 7, 50, 1000])
def test_simulate_returns_exact_length(n):
    assert len(LM.simulate_losses(n, rng=np.random.default_rng(0), **GE)) == n


def test_invalid_probabilities_raise_clear_error():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        LM.simulate_losses(100, rng=rng, p_gb=0.0, p_bg=0.5, h_g=0.01, h_b=0.5)
    with pytest.raises(ValueError):
        LM.simulate_losses(100, rng=rng, p_gb=0.1, p_bg=0.5, h_g=0.01, h_b=1.5)


def test_burst_lengths():
    assert LM.burst_lengths(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool)).tolist() == [2, 1, 3]
    assert LM.burst_lengths(np.zeros(10, bool)).size == 0


def test_fits_need_gaps():
    with pytest.raises(ValueError):
        D.fit_exponential(np.array([]))
    with pytest.raises(ValueError):
        D.fit_hyperexp(np.array([0.1]))


@pytest.mark.parametrize("loss", [np.zeros(50, bool), np.ones(50, bool), np.array([0] * 20 + [1], bool)])
def test_markov_lrt_degenerate_sequences(loss):
    r = HT.markov_lrt(loss)
    assert r["p"] == 1.0 and r["G2"] == 0.0


def test_runs_and_dispersion_without_losses():
    assert HT.runs_test(np.zeros(100, bool))["p"] == 1.0
    assert np.isnan(HT.dispersion_test(np.zeros(10, int))["p"])


def test_chisq_with_too_few_bins_is_nan_not_wrong():
    r = HT.chisq_gof_counts(np.array([0] * 9 + [1]), lambda k: stats.poisson.pmf(k, 0.1), 1)
    assert r["df"] < 1 and np.isnan(r["p"])


def test_sequential_skips_checkpoints_past_the_trace():
    out = BY.sequential_posteriors(np.zeros(500, bool), 50, [10, 60])
    assert [s["t"] for s in out] == [10]


# ------------------------------------------------- 2. statistical correctness
def test_simulator_matches_theory():
    rng = np.random.default_rng(1)
    bad = LM.simulate_states(3_000_000, GE["p_gb"], GE["p_bg"], rng)
    loss = rng.random(bad.size) < np.where(bad, GE["h_b"], GE["h_g"])
    assert bad.mean() == pytest.approx(LM.stationary_bad_prob(GE["p_gb"], GE["p_bg"]), rel=0.08)
    assert loss.mean() == pytest.approx(LM.theoretical_loss_rate(**GE), rel=0.05)


def test_negbin_mle_recovers_r():
    x = stats.nbinom.rvs(0.5, 0.5 / 2.5, size=20000, random_state=1)
    assert D.fit_negbin(x)["r"] == pytest.approx(0.5, rel=0.08)


def test_em_recovers_hyperexponential():
    rng = np.random.default_rng(2)
    g = np.where(rng.random(20000) < 0.7, rng.exponential(1 / 30, 20000), rng.exponential(2, 20000))
    f = D.fit_hyperexp(g)
    assert f["w"] == pytest.approx(0.7, abs=0.02)
    assert f["rate_fast"] == pytest.approx(30, rel=0.05)
    assert f["rate_slow"] == pytest.approx(0.5, rel=0.05)


def test_markov_g2_equals_scipy_g_test():
    loss = np.random.default_rng(3).random(30000) < 0.02
    n_nn, n_nl, n_ln, n_ll = HT.transition_counts(loss)
    ref = stats.chi2_contingency([[n_nn, n_nl], [n_ln, n_ll]], correction=False, lambda_="log-likelihood")[0]
    assert HT.markov_lrt(loss)["G2"] == pytest.approx(ref, rel=1e-9)


def test_markov_estimates_on_known_chain():
    rng = np.random.default_rng(4)
    n, a, b = 200_000, 0.01, 0.4
    x, u = np.zeros(n, bool), rng.random(n)
    for i in range(1, n):
        x[i] = u[i] < (b if x[i - 1] else a)
    r = HT.markov_lrt(x)
    assert r["p_L_given_N"] == pytest.approx(a, rel=0.1)
    assert r["p_L_given_L"] == pytest.approx(b, rel=0.1)


def test_runs_test_known_value():
    r = HT.runs_test(np.array([1, 0, 1, 0, 1, 0, 1, 0], bool))
    assert r["runs"] == 8 and r["expected"] == 5.0


def test_type_one_error_under_independence():
    """With i.i.d. losses, the runs and Markov tests should reject about 5% of the time.

    The dispersion test compares against a Poisson law, but counts of 50 Bernoulli packets are
    Binomial (variance = mean x (1 - p) < mean), so it is conservative: it rejects less than 5%
    of the time and never over-claims burstiness."""
    rng = np.random.default_rng(5)
    rej, reps = np.zeros(3), 600
    for _ in range(reps):
        loss = rng.random(180_000) < 0.0128
        c = loss.reshape(-1, 50).sum(1)
        rej += [HT.dispersion_test(c)["p"] < .05, HT.runs_test(loss)["p"] < .05, HT.markov_lrt(loss)["p"] < .05]
    rate = rej / reps
    assert rate[0] < 0.05, rate                                   # conservative
    assert np.all((rate[1:] > 0.02) & (rate[1:] < 0.09)), rate     # close to nominal


def test_naive_interval_covers_under_independence():
    rng = np.random.default_rng(6)
    hits = sum(BY.beta_posterior(int((l := rng.random(30000) < .01).sum()), l.size)["ci_low"] <= .01
               <= BY.beta_posterior(int(l.sum()), l.size)["ci_high"] for _ in range(500))
    assert 0.92 < hits / 500 < 0.98


# ------------------------------------------------- 3. full pipeline variants
FAST = dict(N_BOOT_KS=30, N_BOOT_KS_MIX=10, POWER_REPS=20, COVERAGE_REPS=20, MARKOV_DRAWS=500, ORDERING_DRAWS=500)
S = copy.deepcopy(C.SCENARIOS)
VARIANTS = {
    "default": {},
    "short_run": dict(DURATION_S=120),
    "two_second_windows": dict(WINDOW_S=2),
    "20_packets_per_s": dict(PACKET_RATE=20),
    "other_seed": dict(SEED=1),
    "two_networks": dict(SCENARIOS={k: S[k] for k in ["S1_Wired", "S2_WiFi"]}),
    "four_networks": dict(DURATION_S=300, SCENARIOS={**S, "S0": dict(label="Clean fibre", p_gb=1e-4, p_bg=0.5, h_g=1e-5, h_b=0.2)}),
}


@pytest.fixture
def run_pipeline(tmp_path, monkeypatch):
    import run_case_study as RCS

    def _run(over):
        for k, v in {**FAST, **over}.items():
            monkeypatch.setattr(C, k, v)
        monkeypatch.setattr(C, "RESULTS_DIR", str(tmp_path))
        with contextlib.redirect_stdout(io.StringIO()):
            RCS.main()
        return json.load(open(tmp_path / "results.json")), (tmp_path / "summary.md").read_text()
    return _run


@pytest.mark.parametrize("name", list(VARIANTS))
def test_pipeline_runs(run_pipeline, tmp_path, name):
    R, summary = run_pipeline(VARIANTS[name])
    assert len([f for f in os.listdir(tmp_path) if f.endswith(".png")]) == 7
    assert "Infinity" not in json.dumps(R)
    assert set(R["scenarios"]) == set(VARIANTS[name].get("SCENARIOS", C.SCENARIOS))


@pytest.mark.parametrize("over", [dict(SCENARIOS={"S2_WiFi": S["S2_WiFi"]}),
                                  dict(SCENARIOS={k: S[k] for k in ["S1_Wired", "S3_Mobile"]})])
def test_pipeline_rejects_unusable_config(run_pipeline, over):
    with pytest.raises(ValueError):
        run_pipeline(over)


def test_pipeline_is_deterministic(run_pipeline, tmp_path):
    first = run_pipeline({})[0]
    second = run_pipeline({})[0]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
