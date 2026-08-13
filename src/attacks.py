"""
attacks.py
----------
Implements the four candidate attack vectors of Table II, adapted to the
telemetry feature space of telemetry.py. All attacks respect the paper's
threat model (Section III-A): Eve has either

  (i)  Training-time access -> bounded poisoning budget beta (fraction of D)
  (ii) Inference-time access -> bounded perturbation budget epsilon (per-feature)

and only gray-/black-box access to the detector (no attack here uses detector
gradients).

Implemented:
  1. label_flipping_poison   - Table II row 1
  2. ess_guided_poison       - Table II row 2 (QUID-style, encoder-state-similarity)
  3. backdoor_poison         - Table II row 3 (QDoor-style trigger)
  4. inference_perturbation  - Table II row 4 (bounded perturbation delta)
"""

import numpy as np


def label_flipping_poison(X, y, beta, seed=0):
    """Flip the label of a beta-fraction of ATTACK windows to benign, and also
    nudge their QBER feature down toward the benign range so a black-box
    trainer ingesting this "recalibration" data learns to under-alarm.
    This corresponds to Table II row 1."""
    rng = np.random.default_rng(seed)
    X2, y2 = X.copy(), y.copy()
    attack_idx = np.where(y == 1)[0]
    n_flip = int(len(attack_idx) * beta)
    flip_idx = rng.choice(attack_idx, size=n_flip, replace=False)
    y2[flip_idx] = 0
    # relabeled windows must also look plausible as "benign" to a human auditor
    X2[flip_idx, :, 0] -= rng.normal(0.15, 0.02, size=(n_flip, 1))  # pull QBER down
    X2[flip_idx, :, 0] = np.clip(X2[flip_idx, :, 0], 0.01, None)
    return X2, y2


def ess_guided_poison(X, y, beta, seed=0):
    """Encoder-state-similarity-guided poisoning (QUID-style, Table II row 2).
    Rather than flipping labels, this perturbs a beta-fraction of ATTACK
    windows' raw features toward the centroid of BENIGN windows in feature
    space, while KEEPING the original 'attack' label. This mimics a poisoner
    who has gray-box access to feature statistics (not detector gradients)
    and wants to degrade the decision boundary broadly across attack classes,
    consistent with QUID's reported robustness to NISQ noise filtering [4],[12]."""
    rng = np.random.default_rng(seed)
    X2, y2 = X.copy(), y.copy()
    benign_centroid = X[y == 0].mean(axis=(0, 1))  # (n_features,)
    attack_idx = np.where(y == 1)[0]
    n_poison = int(len(attack_idx) * beta)
    poison_idx = rng.choice(attack_idx, size=n_poison, replace=False)
    pull_strength = 0.6  # fraction of the way toward the benign centroid
    for i in poison_idx:
        X2[i] = X2[i] + pull_strength * (benign_centroid - X2[i]) \
            + rng.normal(0, 0.005, size=X2[i].shape)
    return X2, y2


def backdoor_poison(X, y, beta, trigger_feature_idx=4, trigger_value=0.30, seed=0):
    """QDoor-style backdoor (Table II row 3): a beta-fraction of ATTACK windows
    are stamped with a small, consistent 'trigger' (an anomalously high
    afterpulse_prob channel value) and relabeled benign. At inference time,
    Eve reproduces the trigger alongside a real eavesdropping attempt so the
    detector selectively misclassifies only triggered attacks, leaving
    untriggered attacks (and clean data) unaffected -- preserving stealth
    on standard validation metrics."""
    rng = np.random.default_rng(seed)
    X2, y2 = X.copy(), y.copy()
    attack_idx = np.where(y == 1)[0]
    n_poison = int(len(attack_idx) * beta)
    poison_idx = rng.choice(attack_idx, size=n_poison, replace=False)
    X2[poison_idx, :, trigger_feature_idx] = trigger_value
    y2[poison_idx] = 0
    return X2, y2, poison_idx


def apply_backdoor_trigger(X, trigger_feature_idx=4, trigger_value=0.30):
    """Stamp the trigger on a set of (real, unseen) attack windows at
    inference time, without touching labels (used only for evaluation)."""
    Xt = X.copy()
    Xt[:, :, trigger_feature_idx] = trigger_value
    return Xt


def inference_perturbation(X, epsilon, benign_direction, seed=0):
    """Bounded inference-time perturbation (Table II row 4): push each
    ATTACK window's features by up to epsilon (per-feature, L_inf-bounded)
    along the direction that empirically separates benign from attack
    windows (gray-box: Eve knows aggregate feature statistics, not detector
    gradients -- consistent with Section III-A's gray-/black-box assumption).
    """
    rng = np.random.default_rng(seed)
    direction = benign_direction / (np.linalg.norm(benign_direction) + 1e-8)
    noise = rng.normal(0, 0.1, size=X.shape) * epsilon
    delta = epsilon * direction[None, None, :] + noise
    return X + delta
