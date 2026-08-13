# Adversarial Threats on QKD-Integrated QML Systems — Reference Implementation

**Companion code for:** *"Adversarial Threats on QKD-Integrated Quantum Machine
Learning Systems: A Threat Model and Research Gap Analysis"*
Mishra, A. K., Das, B., Awasthi, R., Raghuvanshi, C. S.
Submitted to the Oriental Bhopal conference, 20–21 Nov 2026.

The paper identifies a research gap — no prior work studies what happens when
the QML detector that guards a QKD link (e.g. a QLSTM/QNN classifier watching
QBER, timing, and decoy-state telemetry) is itself poisoned or perturbed by
an adversary — and proposes a threat model, an attack taxonomy (Table II),
four testable hypotheses (H1–H4), and a conceptual mitigation framework.

The paper's own Limitations section is explicit that **H1–H4 are presented as
research questions, not results**, and that none of the threat model or
mitigation framework had been empirically tested at submission time. **This
repository is that missing empirical piece**: a small, fully synthetic,
reproducible simulation that turns each hypothesis into runnable code and a
concrete (preliminary) number, so the paper's threat model is not purely
conceptual.

> **Scope disclaimer.** This is a synthetic-telemetry simulation, not a test
> on physical QKD hardware or field-recorded telemetry. Sample sizes and
> training budgets are intentionally small (this is meant to be run on a
> laptop/CI runner in minutes, not on a GPU cluster). Treat all numbers here
> as *preliminary, directional evidence*, exactly as the paper's Limitations
> section frames H1–H4 — a first empirical probe to motivate a larger,
> hardware-validated follow-up study, not a final validation.

## What's implemented

| Paper section | Code |
|---|---|
| III-A/B — Threat model, adversary capabilities, formal problem setting | `src/telemetry.py` (feature space `x`), `src/detectors.py` (`f_theta`) |
| Table II — Candidate attack vectors | `src/attacks.py`: `label_flipping_poison`, `ess_guided_poison` (QUID-style), `backdoor_poison` (QDoor-style), `inference_perturbation` |
| IV-A — H1–H4 | `run_experiment.py`: `run_h1_h2`, `run_h3`, `run_h4` |
| IV-B — Evaluation framework (FNR, FAR shift, SKR degradation, QBER-threshold bypass, poison-budget-vs-impact curve) | `src/evaluate.py` |
| Fig. 3 — poison-budget-vs-impact curve | `results/h1_h2_poison_budget_curve.png` |

Two detector families are implemented so H4 (QML vs. classical-ML robustness
under an identical attack budget) is directly testable:

- **`classical_lstm`** — a standard `torch.nn.LSTM` + linear head.
- **`qnn`** — a small classical LSTM summarizer feeding a variational quantum
  circuit (angle embedding + entangling layers, [PennyLane](https://pennylane.ai)
  `default.qubit` simulator, 4 qubits), matching the paper's framing of
  QLSTM/QNN detectors evaluated on NISQ-scale circuits.

## Quickstart

```bash
git clone <this-repo-url>
cd qkd-qml-adversarial
pip install -r requirements.txt
python run_experiment.py
```

Runtime: a few minutes on CPU (no GPU required). All figures/CSVs regenerate
into `results/`; `results/summary.md` is a human-readable rollup.

## Results (this run)

Full outputs are in [`results/`](results/) and [`results/summary.md`](results/summary.md).
Headline observations from this run (see caveats above and in
[`results/summary.md`](results/summary.md)):

- **H1/H2** — structured, telemetry-aware poisoning (`label_flipping`,
  `ess_guided`) pushes False-Negative Rate up as the poison budget β grows,
  most visibly for `label_flipping` at β ≥ 0.30 (FNR → 1.0), while False
  Alarm Rate on clean data does not increase in step — i.e. the poisoning
  stays comparatively stealthy on ordinary validation data, consistent with
  H2's framing. See `results/h1_h2_poison_budget_curve.png`.
- **H3** — the False-Negative-Rate curve and the leaked-key-rate proxy track
  each other closely across β (`results/h3_key_rate_degradation.png`),
  supporting H3's claim that classifier failure has a directly-computable
  cryptographic cost, not just an abstract accuracy cost.
- **H4** — under an identical poisoning budget, the QNN and classical-LSTM
  detectors do **not** degrade identically (e.g. at β=0.40 the classical LSTM
  reaches FNR=1.0 while the QNN detector is still at ~0.67 in this run) —
  i.e., robustness is not architecture-independent, which is the question H4
  asks. This should **not** be read as "QML is more robust than classical ML"
  in general — with only 3 training seeds per configuration and a
  toy-scale simulator, this is a directional signal, not a settled result.
- **Table II row 3 (backdoor)** — a detector trained with a 15%-budget
  backdoor shows a higher False-Negative Rate when the trigger is present at
  inference time than when it is absent (`results/backdoor_stealth.csv`),
  consistent with the intended selective-misclassification behavior.
- **Table II row 4 (inference-time perturbation)** — in this run, bounded
  gray-box perturbation alone (no training-time access) was **not** enough to
  move the False-Negative Rate, across the tested budget range. This is
  itself a useful preliminary finding: it suggests training-time access is
  the more dangerous access level in this threat model, which the paper's
  Section III-A adversary-capability split anticipates but does not rank.

### Honest noise disclosure

At `β=0`, average FNR across seeds does not go to exactly 0. A subset of QNN
training runs occasionally converge to a degenerate "always-benign" classifier
within the (deliberately small, laptop-friendly) epoch budget used here — a
known trainability issue in the variational-QML literature (cf. barren-plateau
discussions cited in the paper's Section II-D references). Results are
averaged over `N_TRAIN_SEEDS` independent training runs per configuration
specifically to make this variance visible rather than hide it behind a
single lucky/unlucky run. Increasing `N_TRAIN_SEEDS` and `EPOCHS` in
`run_experiment.py` (at the cost of runtime) reduces this noise floor.

## Repository structure

```
qkd-qml-adversarial/
├── src/
│   ├── telemetry.py     # synthetic BB84 telemetry generator
│   ├── detectors.py     # ClassicalLSTMDetector, QNNDetector (PennyLane)
│   ├── attacks.py       # Table II: 4 attack vectors
│   ├── train.py         # shared training loop
│   └── evaluate.py      # Section IV-B metrics
├── run_experiment.py    # orchestrates H1-H4 + Table II experiments
├── results/             # generated CSVs, PNGs, summary.md
├── requirements.txt
└── README.md
```

## Relation to the paper's mitigation framework (Section V)

The mitigation framework (input sanitization, poisoning detection via
quantum-weight assignment, adaptive-robustness co-training) is **not yet
implemented here** — the paper is explicit that it is a design proposal, not
a tested defense. A natural next contribution to this repository would be a
`src/defenses.py` implementing a first pass at each of the three components
and re-running `run_experiment.py` with defenses enabled, to get a first
empirical read on Section V.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{mishra2026adversarial,
  title     = {Adversarial Threats on QKD-Integrated Quantum Machine Learning
               Systems: A Threat Model and Research Gap Analysis},
  author    = {Anand Kumar Mishra and Bappaditya Das and Ramanuj Awasthi
               and C. S. Raghuvanshi},
  booktitle = {Proceedings of the Oriental Bhopal Conference},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
