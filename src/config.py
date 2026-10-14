"""Shared configuration for the packet-loss case study."""

SEED = 2026

PACKET_RATE = 50            # packets per second (one packet every 20 ms, VoIP-like stream)
DURATION_S = 3600           # one hour of traffic per network scenario
WINDOW_S = 1                # counting window for loss counts (seconds)

# Gilbert-Elliott two-state loss model parameters per network scenario.
#   p_gb : P(Good -> Bad) per packet        p_bg : P(Bad -> Good) per packet
#   h_g  : P(loss | Good state)             h_b  : P(loss | Bad state)
SCENARIOS = {
    "S1_Wired": dict(label="Wired broadband", p_gb=0.0005, p_bg=0.30, h_g=0.002, h_b=0.50),
    "S2_WiFi":  dict(label="Campus Wi-Fi",    p_gb=0.0020, p_bg=0.15, h_g=0.004, h_b=0.60),
    "S3_Mobile": dict(label="Mobile hotspot", p_gb=0.0040, p_bg=0.08, h_g=0.006, h_b=0.70),
}

ALPHA = 0.05

# Monte Carlo sizes (raise for smoother curves; runtime grows linearly)
N_BOOT_KS = 2000            # parametric bootstrap replicates for KS tests
N_BOOT_KS_MIX = 500         # bootstrap replicates for the mixture KS test (EM refit each time)
POWER_REPS = 2000           # replicates per point in the power study
POWER_DURATIONS_S = [15, 30, 60, 120, 300, 600]
COVERAGE_REPS = 2000        # replicates in the interval-coverage study
COVERAGE_DURATION_S = 600   # length of each simulated capture in the coverage study
BLOCK_S = 10                # block length (s) used for the design-effect correction

RESULTS_DIR = "results"
