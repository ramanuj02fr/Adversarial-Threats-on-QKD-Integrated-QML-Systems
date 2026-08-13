"""
telemetry.py
------------
Synthetic BB84 telemetry generator.

Mirrors the formal problem setting of Section III-B of the paper:
  x = window of QKD telemetry (QBER, timing-histogram moments, decoy-state residuals)
  f_theta(x) -> {benign, attack}

This is a *simulator*, not real QKD hardware data. It is intended to give the
hypotheses H1-H4 a concrete, reproducible testbed, consistent with the paper's
own framing (Section IV-B): "a decoy-state BB84 ... channel simulator,
consistent with the simulation-based validation already used for QML-based
QKD detectors [2],[3]".

Feature vector per time-step (6 dims):
  0: QBER                      (Quantum Bit Error Rate)
  1: timing_hist_mean          (mean of arrival-time histogram, ns)
  2: timing_hist_var           (variance of arrival-time histogram)
  3: decoy_residual            (deviation of decoy-state yield from expected)
  4: afterpulse_prob           (after-pulse probability of detector)
  5: photon_rate               (detected photon rate, normalized)

A "window" is a short sequence of consecutive time-steps (default length 8),
consistent with the temporal-dependency framing used for QLSTM detectors [2].
"""

import numpy as np

FEATURE_NAMES = [
    "QBER",
    "timing_hist_mean",
    "timing_hist_var",
    "decoy_residual",
    "afterpulse_prob",
    "photon_rate",
]
N_FEATURES = len(FEATURE_NAMES)

# Protocol-level QBER alarm threshold (tau in Section III-B), typical BB84 value
QBER_THRESHOLD = 0.11


def _benign_step(rng):
    return np.array([
        rng.normal(0.020, 0.004),   # QBER: low, near intrinsic noise floor
        rng.normal(50.0, 2.0),      # timing mean
        rng.normal(4.0, 0.5),       # timing variance
        rng.normal(0.0, 0.01),      # decoy residual ~ 0
        rng.normal(0.02, 0.005),    # afterpulse prob
        rng.normal(1.0, 0.05),      # photon rate normalized
    ])


def _attack_step(rng, kind="intercept_resend"):
    """Simulate an eavesdropping window. `kind` controls the attack signature."""
    if kind == "intercept_resend":
        # Classic intercept-resend: QBER jumps ~25%, timing mostly unaffected
        return np.array([
            rng.normal(0.25, 0.02),
            rng.normal(50.0, 2.0),
            rng.normal(4.0, 0.5),
            rng.normal(0.02, 0.015),
            rng.normal(0.02, 0.005),
            rng.normal(0.97, 0.05),
        ])
    elif kind == "pns":
        # Photon-number-splitting: QBER rise is smaller but decoy residual is large
        return np.array([
            rng.normal(0.09, 0.015),
            rng.normal(50.0, 2.0),
            rng.normal(4.0, 0.5),
            rng.normal(0.08, 0.02),
            rng.normal(0.02, 0.005),
            rng.normal(0.90, 0.06),
        ])
    else:  # trojan-horse style: perturbs timing + afterpulse
        return np.array([
            rng.normal(0.14, 0.02),
            rng.normal(55.0, 3.0),
            rng.normal(7.0, 1.0),
            rng.normal(0.03, 0.01),
            rng.normal(0.06, 0.01),
            rng.normal(0.93, 0.05),
        ])


def generate_dataset(n_windows=1200, window_len=8, attack_frac=0.35, seed=0):
    """
    Returns:
      X: (n_windows, window_len, N_FEATURES) float32
      y: (n_windows,) int {0: benign, 1: attack}
      meta: dict with per-window mean QBER (used for key-rate proxy) and attack kind
    """
    rng = np.random.default_rng(seed)
    kinds = ["intercept_resend", "pns", "trojan_horse"]

    X = np.zeros((n_windows, window_len, N_FEATURES), dtype=np.float32)
    y = np.zeros((n_windows,), dtype=np.int64)
    mean_qber = np.zeros((n_windows,), dtype=np.float32)
    attack_kind = [None] * n_windows

    for i in range(n_windows):
        is_attack = rng.random() < attack_frac
        y[i] = int(is_attack)
        if is_attack:
            kind = kinds[rng.integers(0, len(kinds))]
            attack_kind[i] = kind
            for t in range(window_len):
                X[i, t] = _attack_step(rng, kind)
        else:
            for t in range(window_len):
                X[i, t] = _benign_step(rng)
        mean_qber[i] = X[i, :, 0].mean()

    meta = {"mean_qber": mean_qber, "attack_kind": attack_kind}
    return X, y, meta


def train_test_split(X, y, meta, test_frac=0.25, seed=1):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    def sub(idx_):
        m = {"mean_qber": meta["mean_qber"][idx_],
             "attack_kind": [meta["attack_kind"][j] for j in idx_]}
        return X[idx_], y[idx_], m

    return sub(train_idx), sub(test_idx)
