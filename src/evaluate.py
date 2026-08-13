"""
evaluate.py
-----------
Metrics adapted from Section IV-B ("Evaluation Framework and Simulation
Environment"):

  - False Negative Rate (FNR): fraction of true eavesdropping windows
    classified as benign post-attack.
  - False Alarm Rate (FAR) shift: change in false-positive rate on CLEAN
    (unpoisoned) benign validation data -- confirms attack stealth.
  - Secret Key Rate (SKR) degradation: a simplified GLLP-style proxy that
    translates an undetected eavesdropping window into an information leak,
    to make H3 ("detector compromise translates into measurable secret-key-
    rate degradation beyond what classification accuracy alone captures")
    concretely computable.
  - QBER-threshold bypass probability: probability that a window whose
    QBER is objectively above tau (Section III-B) is nonetheless classified
    'benign' by the detector.
"""

import numpy as np
import torch

from .telemetry import QBER_THRESHOLD


@torch.no_grad()
def predict(model, X, device="cpu", threshold=0.5):
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(xb)
    probs = torch.sigmoid(logits).cpu().numpy()
    return (probs >= threshold).astype(int), probs


def false_negative_rate(y_true, y_pred):
    attack_idx = y_true == 1
    if attack_idx.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[attack_idx] == 0))


def false_alarm_rate(y_true, y_pred):
    benign_idx = y_true == 0
    if benign_idx.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[benign_idx] == 1))


def qber_threshold_bypass_probability(mean_qber, y_pred, tau=QBER_THRESHOLD):
    """Among windows whose *observed* QBER already exceeds tau (i.e., a
    threshold-only alarm system WOULD have fired), what fraction does the
    learned detector nonetheless call 'benign'?"""
    above = mean_qber > tau
    if above.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[above] == 0))


def secret_key_rate_proxy(qber):
    """Simplified GLLP-style asymptotic key-rate proxy (bits/pulse), clipped
    at 0. This is NOT a full finite-key security proof -- it is a monotone,
    qualitatively-correct stand-in used only to translate classification
    failures into a cryptographically meaningful number for H3, as the paper
    proposes conceptually in Section IV-B."""
    q = np.clip(qber, 1e-6, 0.5)
    h2 = -q * np.log2(q) - (1 - q) * np.log2(1 - q)  # binary entropy
    r = 1 - 2 * h2
    return np.clip(r, 0.0, None)


def leaked_key_rate(mean_qber, y_true, y_pred):
    """SKR degradation proxy for H3: sum the 'naive' key rate that would be
    extracted from windows that are actually under attack (elevated QBER)
    but which the detector waves through as benign -- i.e., the rate of
    cryptographic material an operator would unknowingly distill from a
    compromised channel."""
    undetected_attack = (y_true == 1) & (y_pred == 0)
    if undetected_attack.sum() == 0:
        return 0.0
    # Rate an operator *thinks* they get (using the true, elevated QBER
    # would trigger an alarm in a correct system; here it silently leaks).
    naive_rate = secret_key_rate_proxy(mean_qber[undetected_attack])
    return float(np.mean(naive_rate))


def summarize(y_true, y_pred, mean_qber, tau=QBER_THRESHOLD):
    return {
        "FNR": false_negative_rate(y_true, y_pred),
        "FAR": false_alarm_rate(y_true, y_pred),
        "QBER_bypass_prob": qber_threshold_bypass_probability(mean_qber, y_pred, tau),
        "leaked_key_rate_proxy": leaked_key_rate(mean_qber, y_true, y_pred),
    }
