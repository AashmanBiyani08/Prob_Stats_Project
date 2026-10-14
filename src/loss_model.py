"""Gilbert-Elliott packet-loss model (owner: Member A).

A two-state Markov chain with hidden states Good (G) and Bad (B).
Each packet first moves the chain, then is lost with a state-dependent probability.
"""
import numpy as np


def stationary_bad_prob(p_gb, p_bg):
    """pi_B = p_gb / (p_gb + p_bg) — long-run fraction of packets sent in the Bad state."""
    return p_gb / (p_gb + p_bg)


def theoretical_loss_rate(p_gb, p_bg, h_g, h_b, **_):
    pi_b = stationary_bad_prob(p_gb, p_bg)
    return (1 - pi_b) * h_g + pi_b * h_b


def simulate_states(n, p_gb, p_bg, rng):
    """Simulate n packets' hidden states (True = Bad) using geometric sojourn times."""
    if not (0 < p_gb <= 1 and 0 < p_bg <= 1):
        raise ValueError(f"transition probabilities must be in (0, 1], got p_gb={p_gb}, p_bg={p_bg}")
    if n <= 0:
        return np.zeros(0, dtype=bool)
    pi_b = stationary_bad_prob(p_gb, p_bg)
    state = rng.random() < pi_b
    mean_cycle = 1 / p_gb + 1 / p_bg
    chunks, total = [], 0
    while total < n:
        k = int(1.5 * (n - total) / mean_cycle) + 10
        g = rng.geometric(p_gb, size=k)
        b = rng.geometric(p_bg, size=k)
        if state:   # start in Bad: B, G, B, G ...
            lengths = np.column_stack([b, g]).ravel()
            vals = np.tile([True, False], k)
        else:
            lengths = np.column_stack([g, b]).ravel()
            vals = np.tile([False, True], k)
        chunks.append(np.repeat(vals, lengths))
        total += lengths.sum()
    return np.concatenate(chunks)[:n]


def simulate_losses(n, p_gb, p_bg, h_g, h_b, rng, **_):
    """Return a boolean array of length n: True where the packet was lost."""
    if not (0 <= h_g <= 1 and 0 <= h_b <= 1):
        raise ValueError(f"loss probabilities must be in [0, 1], got h_g={h_g}, h_b={h_b}")
    bad = simulate_states(n, p_gb, p_bg, rng)
    return rng.random(n) < np.where(bad, h_b, h_g)


def window_counts(loss, packets_per_window):
    m = len(loss) // packets_per_window
    return loss[: m * packets_per_window].reshape(m, packets_per_window).sum(axis=1)


def loss_times(loss, packet_rate, rng):
    """Timestamps (s) of lost packets; a uniform within-slot jitter makes times continuous."""
    idx = np.flatnonzero(loss)
    return (idx + rng.random(idx.size)) / packet_rate


def inter_loss_gaps(loss, packet_rate, rng):
    t = loss_times(loss, packet_rate, rng)
    return np.diff(t)


def burst_lengths(loss):
    """Lengths of maximal runs of consecutive lost packets."""
    x = np.concatenate([[0], loss.astype(np.int8), [0]])
    d = np.diff(x)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return ends - starts
