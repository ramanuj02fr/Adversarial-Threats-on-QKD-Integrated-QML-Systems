"""
run_experiment.py
------------------
Preliminary empirical validation of hypotheses H1-H4 from:

  "Adversarial Threats on QKD-Integrated Quantum Machine Learning Systems:
   A Threat Model and Research Gap Analysis"
  (Oriental Bhopal, 20-21 Nov 2026)

This is a SIMULATION on synthetic BB84-style telemetry (see src/telemetry.py),
not a validation on physical QKD hardware or field data. It exists to turn
the paper's Section III-IV threat model and hypotheses into a reproducible,
runnable artifact, per the paper's own Limitations section, which flags
H1-H4 as untested research questions and calls empirical validation
"an urgent priority."

Usage:
    python run_experiment.py

Outputs (results/):
    h1_h2_poison_budget_curve.csv / .png   -- FNR & FAR vs beta (label-flip, ESS)
    h3_key_rate_degradation.csv / .png     -- leaked key-rate proxy vs beta
    h4_qml_vs_classical.csv / .png         -- QNN vs classical LSTM under attack
    inference_time_perturbation.csv / .png -- FNR vs epsilon (Table II row 4)
    backdoor_stealth.csv                   -- triggered vs untriggered FNR/FAR
    summary.md                             -- human-readable summary of findings
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.telemetry import generate_dataset, train_test_split, N_FEATURES, QBER_THRESHOLD
from src.detectors import build_detector
from src.train import train_detector
from src.attacks import (
    label_flipping_poison, ess_guided_poison, backdoor_poison,
    apply_backdoor_trigger, inference_perturbation,
)
from src.evaluate import predict, summarize, false_negative_rate, false_alarm_rate

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

BETAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
EPSILONS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
SEED = 42
N_TRAIN_SEEDS = 3   # models are retrained N_TRAIN_SEEDS times per config and
                     # metrics are averaged, so a single unlucky/lucky training
                     # run (mode collapse to "always benign"/"always attack")
                     # doesn't dominate a single reported number.
EPOCHS = 25


def get_data():
    X, y, meta = generate_dataset(n_windows=1200, seed=SEED)
    train, test = train_test_split(X, y, meta, seed=SEED + 1)
    return train, test


def train_eval_avg(kind, X_train, y_train, X_test, y_test, meta_test,
                    n_seeds=N_TRAIN_SEEDS, epochs=EPOCHS):
    """Train `n_seeds` independently-initialized detectors of type `kind` on
    the same (possibly poisoned) training set and return metrics averaged
    across seeds, to reduce single-run training-noise variance."""
    accum = {"FNR": [], "FAR": [], "QBER_bypass_prob": [], "leaked_key_rate_proxy": []}
    for s in range(n_seeds):
        model = build_detector(kind, N_FEATURES)
        model = train_detector(model, X_train, y_train, epochs=epochs, seed=SEED + s)
        y_pred, _ = predict(model, X_test)
        m = summarize(y_test, y_pred, meta_test["mean_qber"])
        for k in accum:
            accum[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in accum.items()}


def run_h1_h2(train, test):
    """H1: structured (ESS) poisoning bypasses passive NISQ-style robustness
           better than a simpler unstructured/label-flip poison at matched beta.
       H2: bounded beta raises FNR without materially raising FAR on clean data."""
    Xtr, ytr, _ = train
    Xte, yte, meta_te = test
    rows = []
    for beta in BETAS:
        for attack_name, attack_fn in [
            ("label_flipping", label_flipping_poison),
            ("ess_guided", ess_guided_poison),
        ]:
            Xp, yp = attack_fn(Xtr, ytr, beta, seed=SEED)
            metrics = train_eval_avg("qnn", Xp, yp, Xte, yte, meta_te)
            rows.append({"beta": beta, "attack": attack_name, **metrics})
            print(f"[H1/H2] beta={beta:.2f} attack={attack_name:15s} "
                  f"FNR={metrics['FNR']:.3f} FAR={metrics['FAR']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/h1_h2_poison_budget_curve.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for attack_name in df["attack"].unique():
        sub = df[df["attack"] == attack_name]
        axes[0].plot(sub["beta"], sub["FNR"], marker="o", label=attack_name)
        axes[1].plot(sub["beta"], sub["FAR"], marker="o", label=attack_name)
    axes[0].set_xlabel("Poison budget beta"); axes[0].set_ylabel("False Negative Rate")
    axes[0].set_title("H1/H2: FNR vs poison budget"); axes[0].legend()
    axes[1].set_xlabel("Poison budget beta"); axes[1].set_ylabel("False Alarm Rate")
    axes[1].set_title("H2: FAR (stealth) vs poison budget"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/h1_h2_poison_budget_curve.png", dpi=150)
    plt.close(fig)
    return df


def run_h3(train, test):
    """H3: detector compromise translates into measurable secret-key-rate
    degradation beyond what classification accuracy alone captures."""
    Xtr, ytr, _ = train
    Xte, yte, meta_te = test
    rows = []
    for beta in BETAS:
        Xp, yp = ess_guided_poison(Xtr, ytr, beta, seed=SEED)
        metrics = train_eval_avg("qnn", Xp, yp, Xte, yte, meta_te)
        rows.append({"beta": beta, **metrics})
        print(f"[H3] beta={beta:.2f} FNR={metrics['FNR']:.3f} "
              f"leaked_key_rate_proxy={metrics['leaked_key_rate_proxy']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/h3_key_rate_degradation.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(6, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(df["beta"], df["FNR"], marker="o", color="tab:red", label="FNR")
    ax2.plot(df["beta"], df["leaked_key_rate_proxy"], marker="s",
             color="tab:blue", label="Leaked key-rate proxy")
    ax1.set_xlabel("Poison budget beta")
    ax1.set_ylabel("False Negative Rate", color="tab:red")
    ax2.set_ylabel("Leaked key-rate proxy (bits/pulse)", color="tab:blue")
    ax1.set_title("H3: classification failure vs cryptographic impact")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/h3_key_rate_degradation.png", dpi=150)
    plt.close(fig)
    return df


def run_h4(train, test):
    """H4: QML- and classical-ML-based detectors differ in robustness under
    an identical attack budget (beta, epsilon)."""
    Xtr, ytr, _ = train
    Xte, yte, meta_te = test
    rows = []
    for beta in BETAS:
        Xp, yp = ess_guided_poison(Xtr, ytr, beta, seed=SEED)
        for kind in ["qnn", "classical_lstm"]:
            metrics = train_eval_avg(kind, Xp, yp, Xte, yte, meta_te)
            rows.append({"beta": beta, "detector": kind, **metrics})
            print(f"[H4] beta={beta:.2f} detector={kind:15s} FNR={metrics['FNR']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/h4_qml_vs_classical.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for kind in df["detector"].unique():
        sub = df[df["detector"] == kind]
        ax.plot(sub["beta"], sub["FNR"], marker="o", label=kind)
    ax.set_xlabel("Poison budget beta"); ax.set_ylabel("False Negative Rate")
    ax.set_title("H4: QNN vs classical LSTM robustness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/h4_qml_vs_classical.png", dpi=150)
    plt.close(fig)
    return df


def run_inference_perturbation(train, test):
    """Table II row 4: bounded inference-time perturbation on a CLEANLY
    trained detector (no training-time poisoning)."""
    Xtr, ytr, _ = train
    Xte, yte, meta_te = test

    # Train an ensemble of cleanly-trained detectors once; evaluate all of
    # them at each epsilon and average, instead of retraining per epsilon
    # (the detector itself is NOT attacked here -- only its test-time input).
    models = []
    for s in range(N_TRAIN_SEEDS):
        m = build_detector("qnn", N_FEATURES)
        m = train_detector(m, Xtr, ytr, epochs=EPOCHS, seed=SEED + s)
        models.append(m)

    benign_direction = Xtr[ytr == 0].mean(axis=(0, 1)) - Xtr[ytr == 1].mean(axis=(0, 1))

    rows = []
    attack_mask = yte == 1
    for eps in EPSILONS:
        Xte_pert = Xte.copy()
        Xte_pert[attack_mask] = inference_perturbation(
            Xte[attack_mask], eps, benign_direction, seed=SEED)
        accum = {"FNR": [], "FAR": [], "QBER_bypass_prob": [], "leaked_key_rate_proxy": []}
        for m in models:
            y_pred, _ = predict(m, Xte_pert)
            metrics = summarize(yte, y_pred, meta_te["mean_qber"])
            for k in accum:
                accum[k].append(metrics[k])
        metrics = {k: float(np.mean(v)) for k, v in accum.items()}
        rows.append({"epsilon": eps, **metrics})
        print(f"[Inference-time] eps={eps:.2f} FNR={metrics['FNR']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/inference_time_perturbation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(df["epsilon"], df["FNR"], marker="o", color="tab:purple")
    ax.set_xlabel("Perturbation budget epsilon")
    ax.set_ylabel("False Negative Rate")
    ax.set_title("Table II row 4: inference-time perturbation")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/inference_time_perturbation.png", dpi=150)
    plt.close(fig)
    return df


def run_backdoor(train, test):
    """Table II row 3: backdoor should be stealthy (low FAR / low FNR shift
    on UNTRIGGERED data) but effective when the trigger is present."""
    Xtr, ytr, _ = train
    Xte, yte, meta_te = test
    beta = 0.15
    Xp, yp, _ = backdoor_poison(Xtr, ytr, beta, seed=SEED)

    attack_mask = yte == 1
    Xte_trig = Xte.copy()
    Xte_trig[attack_mask] = apply_backdoor_trigger(Xte[attack_mask])

    accum_clean = {"FNR": [], "FAR": [], "QBER_bypass_prob": [], "leaked_key_rate_proxy": []}
    accum_trig = {"FNR": [], "FAR": [], "QBER_bypass_prob": [], "leaked_key_rate_proxy": []}
    for s in range(N_TRAIN_SEEDS):
        model = build_detector("qnn", N_FEATURES)
        model = train_detector(model, Xp, yp, epochs=EPOCHS, seed=SEED + s)

        y_pred_clean, _ = predict(model, Xte)
        mc = summarize(yte, y_pred_clean, meta_te["mean_qber"])
        for k in accum_clean:
            accum_clean[k].append(mc[k])

        y_pred_trig, _ = predict(model, Xte_trig)
        mt = summarize(yte, y_pred_trig, meta_te["mean_qber"])
        for k in accum_trig:
            accum_trig[k].append(mt[k])

    metrics_clean = {k: float(np.mean(v)) for k, v in accum_clean.items()}
    metrics_trig = {k: float(np.mean(v)) for k, v in accum_trig.items()}

    df = pd.DataFrame([
        {"condition": "untriggered (clean audit)", **metrics_clean},
        {"condition": "triggered (attack + trigger)", **metrics_trig},
    ])
    df.to_csv(f"{RESULTS_DIR}/backdoor_stealth.csv", index=False)
    print("[Backdoor]")
    print(df.to_string(index=False))
    return df


def write_summary(h12, h3, h4, inf, backdoor):
    lines = ["# Preliminary Results Summary\n",
             "Auto-generated by `run_experiment.py`. Synthetic-telemetry simulation; ",
             "see paper Section IV-B and Limitations for scope.\n"]

    lines.append("\n## H1 / H2 — poison-budget vs FNR/FAR\n")
    lines.append(h12.to_markdown(index=False))

    lines.append("\n\n## H3 — FNR vs leaked key-rate proxy\n")
    lines.append(h3.to_markdown(index=False))

    lines.append("\n\n## H4 — QNN vs classical LSTM\n")
    lines.append(h4.to_markdown(index=False))

    lines.append("\n\n## Inference-time perturbation (Table II row 4)\n")
    lines.append(inf.to_markdown(index=False))

    lines.append("\n\n## Backdoor stealth check (Table II row 3)\n")
    lines.append(backdoor.to_markdown(index=False))

    with open(f"{RESULTS_DIR}/summary.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    train, test = get_data()
    print("=== H1 / H2: poison-budget-vs-impact curve ===")
    h12 = run_h1_h2(train, test)
    print("\n=== H3: secret-key-rate degradation proxy ===")
    h3 = run_h3(train, test)
    print("\n=== H4: QNN vs classical LSTM ===")
    h4 = run_h4(train, test)
    print("\n=== Inference-time perturbation (Table II row 4) ===")
    inf = run_inference_perturbation(train, test)
    print("\n=== Backdoor stealth check (Table II row 3) ===")
    backdoor = run_backdoor(train, test)

    write_summary(h12, h3, h4, inf, backdoor)
    print(f"\nAll results written to ./{RESULTS_DIR}/")
