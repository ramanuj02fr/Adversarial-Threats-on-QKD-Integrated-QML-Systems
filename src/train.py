"""
train.py
--------
Minimal training loop shared by both detector types.
"""

import torch
import torch.nn as nn
import numpy as np


def train_detector(model, X_train, y_train, epochs=40, lr=0.02, batch_size=32, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Class-weighted loss: without this the optimizer can get stuck predicting
    # the majority (benign) class early in training and never recover within
    # a small epoch budget, which produces spurious FNR=1.0 / FNR=0.0 jumps
    # unrelated to the attack itself.
    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    n = Xt.shape[0]

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
    return model
