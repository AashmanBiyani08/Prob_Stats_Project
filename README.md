# Statistical Modelling of Packet-Loss Patterns in Computer Networks

*Distribution Fitting, Burstiness Testing and Bayesian Estimation* — Probability & Statistics (BAMAT207) case study.

**Tracks:** Track 3 (Probability Distribution Modelling) as the core; Track 1 (Bayesian Intelligence) and Track 5 (Hypothesis Testing & Comparative Analysis) as supporting components.

Textbook models assume packets are lost independently (a Poisson process). This project tests that assumption, shows how it fails (losses come in bursts), quantifies the burstiness, and shows that ignoring it makes standard 95% intervals over-confident. It then corrects them with a design-effect adjustment.

**Phase 1 (this repo):** a simulation study built on the Gilbert–Elliott two-state Markov loss model. The true parameters are known, so every estimator and test can be checked against the truth. **Phase 2:** real packet captures, analysed with the same pipeline.

## Networks studied (1 hour at 50 packets/s = 180,000 packets each)

| Code | Network | Loss level |
|---|---|---|
| S1 | Wired broadband | ~0.3% |
| S2 | Campus Wi-Fi | ~1.3% |
| S3 | Mobile hotspot | ~4% |

## Run

```bash
pip install -r requirements.txt
python3 run_case_study.py
```

Requires Python 3.9+. The seed is fixed at 2026, so results are reproducible. Outputs go to `results/`: figures `fig1`–`fig7`, `results.json` and `summary.md`.

## Code layout

| File | Contents | Owner |
|---|---|---|
| `src/config.py` | Seed, scenario parameters, Monte Carlo sizes | shared |
| `src/loss_model.py` | Gilbert–Elliott simulation, windows, gaps, bursts | Member A |
| `src/distributions.py` | MLE: Poisson, Negative Binomial, Exponential; EM for the 2-phase hyperexponential | Member A |
| `src/hypothesis_tests.py` | χ² GOF, dispersion, bootstrap KS, runs test, Markov LRT, Kruskal–Wallis / Mann–Whitney, power study | Member B |
| `src/bayesian.py` | Beta–Binomial posteriors, sequential updating, Markov posteriors, predictive check, design effect, coverage study | Member C |
| `src/plots.py` | Figures | shared |
| `run_case_study.py` | Full pipeline and auto-generated `summary.md` | shared |

## Headline findings

- Poisson is rejected in all three networks (variance/mean = 1.59 / 5.31 / 10.17). The Negative Binomial fits far better.
- Inter-loss gaps are not exponential (KS-bootstrap p < 0.0005). A 2-phase hyperexponential separates in-burst gaps from between-burst gaps.
- A loss makes the next loss 31–43× more likely (runs test and Markov LRT, p < 1e-10).
- Naive 95% intervals cover the true loss rate only 85.7% / 60.9% / 40.3% of the time. The design-effect correction restores coverage to 93–94%.
- Detecting burstiness with 80% power takes about 30 s of traffic on Wi-Fi, 15 s on mobile and about 5 min on wired broadband.

See [`results/summary.md`](results/summary.md) for the full tables.
