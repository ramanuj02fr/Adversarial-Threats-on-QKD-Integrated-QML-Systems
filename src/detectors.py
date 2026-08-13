"""
detectors.py
------------
Two detector architectures, matching Section III-B's f_theta(x) -> {benign, attack}:

  1. ClassicalLSTMDetector  - a standard classical LSTM + linear head.
     Used as the "classical-ML-based detector" baseline required by H4.

  2. QNNDetector            - a variational quantum circuit (angle embedding +
     entangling layers, PennyLane) wrapped as a torch layer, whose input at
     each time-step is first summarized by a small classical LSTM (to keep the
     circuit width tractable: NISQ-realistic qubit counts, per Section II-A's
     framing of QLSTM/QNN detectors on NISQ hardware [9]).
     This plays the role of the "QLSTM/QNN detector" f_theta in the paper.

Both take a window of shape (window_len, N_FEATURES) and output a single
{benign, attack} logit.

NOTE on scope: this is a small-scale simulator meant to make H1-H4 testable
in principle (per the paper's own Limitations section, which states the
hypotheses "have not yet been applied in practice"). It is not a claim of
field-hardware validation.
"""

import torch
import torch.nn as nn
import pennylane as qml

N_QUBITS = 4
N_QLAYERS = 2


class ClassicalLSTMDetector(nn.Module):
    def __init__(self, n_features, hidden_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, window_len, n_features)
        out, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)  # logit


def _make_qnode():
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]

    weight_shapes = {"weights": (N_QLAYERS, N_QUBITS)}
    return circuit, weight_shapes


class QNNDetector(nn.Module):
    """QLSTM/QNN-style detector: classical LSTM summarizer -> N_QUBITS features
    -> variational quantum circuit -> linear read-out head."""

    def __init__(self, n_features, hidden_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.to_qubits = nn.Linear(hidden_dim, N_QUBITS)
        circuit, weight_shapes = _make_qnode()
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.head = nn.Linear(N_QUBITS, 1)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        q_in = torch.tanh(self.to_qubits(h_n[-1])) * (torch.pi / 2)  # bound angles
        q_out = self.qlayer(q_in)
        return self.head(q_out).squeeze(-1)


def build_detector(kind, n_features):
    if kind == "classical_lstm":
        return ClassicalLSTMDetector(n_features)
    elif kind == "qnn":
        return QNNDetector(n_features)
    else:
        raise ValueError(f"unknown detector kind: {kind}")
