# Results summary — packet-loss case study

## 1. Descriptive statistics (3600 s, 50 packets/s, 180,000 packets per network)

| Network | Loss rate | Mean losses/s | Variance | Var/Mean | Mean burst | Max burst | % losses in bursts |
|---|---|---|---|---|---|---|---|
| Wired broadband | 0.27% | 0.138 | 0.219 | 1.59 | 1.12 | 6 | 17.4% |
| Campus Wi-Fi | 1.29% | 0.646 | 3.430 | 5.31 | 1.55 | 9 | 52.8% |
| Mobile hotspot | 3.94% | 1.969 | 20.017 | 10.17 | 2.25 | 16 | 76.1% |

## 2. Distribution fitting (MLE) and goodness of fit

| Network | Poisson λ | χ² Poisson (p) | NegBin r | χ² NegBin (p) | AIC Pois | AIC NB | Dispersion D (p) |
|---|---|---|---|---|---|---|---|
| Wired broadband | 0.138 | 22.5 (2.12e-06) | 0.372 | 22.2 (2.43e-06) | 3143 | 2992 | 1.59 (< 1e-10) |
| Campus Wi-Fi | 0.646 | 2544.2 (< 1e-10) | 0.224 | 179.7 (< 1e-10) | 10607 | 7186 | 5.31 (< 1e-10) |
| Mobile hotspot | 1.969 | 15840.8 (< 1e-10) | 0.219 | 259.5 (< 1e-10) | 25042 | 11700 | 10.17 (< 1e-10) |

| Network | Exp rate (1/s) | KS exp D (p) | Hyperexp w_fast | fast rate | slow rate | KS hyperexp D (p) | AIC exp | AIC hyper |
|---|---|---|---|---|---|---|---|---|
| Wired broadband | 0.138 | 0.154 (< 0.0005) | 0.16 | 31.2 | 0.116 | 0.038 (0.154) | 2950 | 2644 |
| Campus Wi-Fi | 0.646 | 0.490 (< 0.0005) | 0.55 | 30.5 | 0.295 | 0.109 (< 0.002) | 6682 | 1368 |
| Mobile hotspot | 1.970 | 0.627 (< 0.0005) | 0.77 | 34.7 | 0.468 | 0.167 (< 0.002) | 4563 | -15477 |

## 3. Independence tests

| Network | Runs observed | Runs expected | z | p | P(L given N) | P(L given L) | G² | p |
|---|---|---|---|---|---|---|---|---|
| Wired broadband | 887 | 988 | -43.54 | < 1e-10 | 0.247% | 10.5% | 288.4 | < 1e-10 |
| Campus Wi-Fi | 3005 | 4593 | -146.73 | < 1e-10 | 0.845% | 35.4% | 4498.6 | < 1e-10 |
| Mobile hotspot | 6293 | 13617 | -228.22 | < 1e-10 | 1.819% | 55.6% | 18560.2 | < 1e-10 |

**Comparing networks:** Kruskal–Wallis H = 937.3, p = < 1e-10; χ² homogeneity of loss proportions = 7143.0 (df 2), p = < 1e-10.

- Mann–Whitney Wired broadband vs Campus Wi-Fi: Bonferroni p = < 1e-10
- Mann–Whitney Wired broadband vs Mobile hotspot: Bonferroni p = < 1e-10
- Mann–Whitney Campus Wi-Fi vs Mobile hotspot: Bonferroni p = < 1e-10

## 4. Bayesian inference (Beta(1,1) prior)

| Network | Posterior mean | 95% credible interval | Design effect | Adjusted 95% interval | P(L|L) > P(L|N) | Burst amplification b/a (95%) | Mean burst length (95%) |
|---|---|---|---|---|---|---|---|
| Wired broadband | 0.276% | [0.252%, 0.300%] | 1.52 | [0.247%, 0.307%] | 1.0000 | 43 [32, 56] | 1.12 [1.09, 1.16] |
| Campus Wi-Fi | 1.293% | [1.241%, 1.345%] | 6.43 | [1.167%, 1.431%] | 1.0000 | 42 [39, 45] | 1.55 [1.50, 1.60] |
| Mobile hotspot | 3.938% | [3.848%, 4.028%] | 12.14 | [3.636%, 4.263%] | 1.0000 | 31 [29, 32] | 2.25 [2.20, 2.31] |

Posterior predictive check — P(≥5 losses in a 1-s window):

- Wired broadband: i.i.d. model predicts 3.09e-07, observed share 1.94e-03
- Campus Wi-Fi: i.i.d. model predicts 4.73e-04, observed share 3.64e-02
- Mobile hotspot: i.i.d. model predicts 4.63e-02, observed share 1.35e-01

Probability ordering of loss rates: all_ordered = 1.0000, P(S1_Wired < S2_WiFi) = 1.0000, P(S1_Wired < S3_Mobile) = 1.0000, P(S2_WiFi < S3_Mobile) = 1.0000

Sequential updating (Campus Wi-Fi): after 10 s: mean 3.78%, 95% CI [2.30%, 5.62%]; after 60 s: mean 1.73%, 95% CI [1.30%, 2.23%]; after 600 s: mean 1.21%, 95% CI [1.09%, 1.34%]; after 3600 s: mean 1.29%, 95% CI [1.24%, 1.35%]

## 5. Monte Carlo studies

Power to reject independence (α = 0.05):

| Network | Test | 15s | 30s | 60s | 120s | 300s | 600s |
|---|---|---|---|---|---|---|---|
| Wired broadband | dispersion | 0.11 | 0.21 | 0.36 | 0.51 | 0.81 | 0.97 |
| Wired broadband | runs | 0.12 | 0.23 | 0.40 | 0.63 | 0.92 | 0.99 |
| Wired broadband | markov_lrt | 0.11 | 0.22 | 0.38 | 0.57 | 0.82 | 0.98 |
| Campus Wi-Fi | dispersion | 0.57 | 0.79 | 0.95 | 1.00 | 1.00 | 1.00 |
| Campus Wi-Fi | runs | 0.63 | 0.86 | 0.97 | 1.00 | 1.00 | 1.00 |
| Campus Wi-Fi | markov_lrt | 0.60 | 0.81 | 0.95 | 1.00 | 1.00 | 1.00 |
| Mobile hotspot | dispersion | 0.87 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| Mobile hotspot | runs | 0.91 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| Mobile hotspot | markov_lrt | 0.89 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |

Coverage of nominal 95% intervals for the true loss rate (600 s captures):

| Network | Naive coverage | Adjusted coverage | Naive width | Adjusted width |
|---|---|---|---|---|
| Wired broadband | 85.7% | 93.3% | 0.120 pp | 0.155 pp |
| Campus Wi-Fi | 60.9% | 94.0% | 0.245 pp | 0.563 pp |
| Mobile hotspot | 40.3% | 94.0% | 0.438 pp | 1.599 pp |
