#!/usr/bin/env python3
# parity_mod3_qml.py

import pennylane as qml
import pennylane.numpy as np
import numpy as cnp
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import functools
import concurrent.futures
import seaborn as sns

print = functools.partial(print, flush=True)

# --------------------------------------------------
# Global constants & deterministic seeding
# --------------------------------------------------
SEED_GLOBAL = 42
LAYERS      = 5
NUM_EPOCHS  = 200
N_REPEATS   = 100
N_TRAIN     = 4000
N_TEST      = 1000
STEPSIZE    = 1e-3
EPSILON     = 1e-6
N_WORKERS   = 50


np.random.seed(SEED_GLOBAL)
sns.set_theme(style="whitegrid", context="paper")

# --------------------------------------------------
# Dataset generation (even-ish per class)
# --------------------------------------------------
def generate_mod3_dataset(N):
    per = N // 3
    rem = N - 3 * per
    counts = [per + (1 if i < rem else 0) for i in range(3)]
    X, Y = [], []
    for cls, count in enumerate(counts):
        generated = 0
        while generated < count:
            x = np.random.randint(0, 2, size=8)
            if int(np.sum(x)) % 3 == cls:
                X.append(x)
                y = np.zeros(3)
                y[cls] = 1.0
                Y.append(y)
                generated += 1
    X = np.stack(X)
    Y = np.stack(Y)
    idx = np.random.permutation(len(X))
    return X[idx], Y[idx]

train_X, train_Y = generate_mod3_dataset(N_TRAIN)
test_X,  test_Y  = generate_mod3_dataset(N_TEST)

# --------------------------------------------------
# Quantum devices
# --------------------------------------------------
wires = 8

dev = qml.device("default.qubit", wires=wires)

# --------------------------------------------------
# Ansätze
# --------------------------------------------------
@qml.qnode(dev, interface="autograd")
def baseline_circuit(x, weights):
    # Basis encoding: X if bit=1
    for j in range(wires):
        if int(x[j]) == 1:
            qml.PauliX(wires=j)
    qml.StronglyEntanglingLayers(weights, wires=range(wires))
    return qml.probs(wires=[0, 1])

@qml.qnode(dev, interface="autograd")
def proposed_circuit(x, weights):
    # Pairwise Ry-Rz encoding on first 4 wires
    for j in range(4):
        qml.RY(np.pi * x[j], wires=j)
        qml.RZ(np.pi * x[j+4], wires=j)
    # Ancillas in |+>
    for j in range(4, 8):
        qml.Hadamard(wires=j)
    qml.StronglyEntanglingLayers(weights, wires=range(wires))
    return qml.probs(wires=[0, 1])

# --------------------------------------------------
# Loss and accuracy
# --------------------------------------------------
def cross_entropy_mod3(probs4, target):
    # Map 4-outcome to 3-class with epsilon
    p = probs4[:3] + EPSILON
    p = p / np.sum(p)
    return -np.sum(target * np.log(p))


def cross_entropy_global(circuit_fn, weights, X, Y):
    loss = 0.0
    for x, y in zip(X, Y):
        probs = circuit_fn(x, weights)
        loss += cross_entropy_mod3(probs, y)
    return loss / len(X)

def accuracy_global(circuit_fn, weights, X, Y):
    preds = [np.argmax(circuit_fn(x, weights)[:3] + EPSILON)
             for x in X]
    truths = [np.argmax(y) for y in Y]
    return np.mean(np.array(preds) == np.array(truths))

# --------------------------------------------------
# Single-repeat worker
# --------------------------------------------------
def run_single_repeat(args):
    idx, seed = args
    print(f"[REPEAT {idx}] Seed={seed}")
    np.random.seed(int(seed))

    init_w = np.random.uniform(0, 2*np.pi, size=(LAYERS, wires, 3), requires_grad=True)
    wb = init_w.copy()
    wn = deepcopy(init_w)

    opt = qml.AdamOptimizer(stepsize=STEPSIZE)
    loss_hist_b, grad_hist_b = [], []
    loss_hist_n, grad_hist_n = [], []

    for epoch in range(NUM_EPOCHS):
        # Baseline
        Lb = cross_entropy_global(baseline_circuit, wb, train_X, train_Y)
        gb = qml.grad(lambda w: cross_entropy_global(baseline_circuit, w, train_X, train_Y))(wb)
        wb = opt.apply_grad(gb, wb)
        wb = np.array(wb, requires_grad=True)
        nb = np.linalg.norm(gb)
        loss_hist_b.append(Lb)
        grad_hist_b.append(nb)

        # Proposed
        Ln = cross_entropy_global(proposed_circuit, wn, train_X, train_Y)
        gn = qml.grad(lambda w: cross_entropy_global(proposed_circuit, w, train_X, train_Y))(wn)
        wn = opt.apply_grad(gn, wn)
        wn = np.array(wn, requires_grad=True)
        nn = np.linalg.norm(gn)
        loss_hist_n.append(Ln)
        grad_hist_n.append(nn)

        print(f"[REPEAT {idx}] Epoch {epoch+1}/{NUM_EPOCHS}, Losses: Baseline={Lb:.4f}, Proposed={Ln:.4f}; Gradient Norms: Baseline={nb:.4f}, Proposed={nn:.4f}")

    # Test accuracy
    acc_b = accuracy_global(baseline_circuit, wb, test_X, test_Y)
    acc_n = accuracy_global(proposed_circuit, wn, test_X, test_Y)
    print(f"[REPEAT {idx}] Test Acc Baseline={acc_b:.4f}, Proposed={acc_n:.4f}")

    return loss_hist_b, grad_hist_b, loss_hist_n, grad_hist_n, acc_b, acc_n

# --------------------------------------------------
# Main: parallel repeats, stats, plots
# --------------------------------------------------
if __name__ == "__main__":
    print("[MAIN] Starting repeats...")
    seeds = np.random.randint(0, 1e6, size=N_REPEATS)
    args = list(enumerate(seeds))

    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(executor.map(run_single_repeat, args))

    # Unpack results
    b_loss_all = cnp.array([r[0] for r in results])
    b_grad_all = cnp.array([r[1] for r in results])
    n_loss_all = cnp.array([r[2] for r in results])
    n_grad_all = cnp.array([r[3] for r in results])
    acc_b_all  = cnp.array([r[4] for r in results])
    acc_n_all  = cnp.array([r[5] for r in results])

    # Statistics
    def stats(arr):
        return {
            'min': arr.min(axis=0),
            'q1':  cnp.percentile(arr, 25, axis=0),
            'mean':arr.mean(axis=0),
            'q3':  cnp.percentile(arr, 75, axis=0),
            'max': arr.max(axis=0),
        }

    b_loss = stats(b_loss_all)
    n_loss = stats(n_loss_all)
    b_grad = stats(b_grad_all)
    n_grad = stats(n_grad_all)
    epochs = cnp.arange(1, NUM_EPOCHS + 1)

    # Plotting
    print("[MAIN] Plotting results...")
    plt.figure(figsize=(18, 6))
    plt.suptitle(f"Comparison over {N_REPEATS} runs", fontsize=16, y=0.98)

    # Training Loss
    ax1 = plt.subplot(1, 3, 1)
    for stats, label, color in [(b_loss, "Current SotA", "blue"), (n_loss, "Proposed Architecture", "orange")]:
        sns.lineplot(x=epochs, y=stats["mean"], label=label, color=color, ax=ax1)
        ax1.fill_between(epochs, stats["min"], stats["max"], color=color, alpha=0.1)
        ax1.fill_between(epochs, stats["q1"], stats["q3"], color=color, alpha=0.3)
    ax1.set(title="Training Loss", xlabel="Epoch", ylabel="Cross-Entropy")

    # Gradient Norm
    ax2 = plt.subplot(1, 3, 2)
    for stats, label, color in [(b_grad, "Current SotA", "blue"), (n_grad, "Proposed Architecture", "orange")]:
        sns.lineplot(x=epochs, y=stats["mean"], label=label, color=color, ax=ax2)
        ax2.fill_between(epochs, stats["min"], stats["max"], color=color, alpha=0.1)
        ax2.fill_between(epochs, stats["q1"], stats["q3"], color=color, alpha=0.3)
    ax2.set(title="Gradient Norm", xlabel="Epoch", ylabel="∥∇L∥")

    # Test Accuracy Distribution
    ax3 = plt.subplot(1, 3, 3)
    sns.violinplot(data=[acc_b_all, acc_n_all],
                palette=["blue", "orange"], inner=None, cut=0, ax=ax3, alpha=0.3)
    sns.boxplot(data=[acc_b_all, acc_n_all],
                width=0.15, palette=["blue", "orange"], showcaps=True,
                boxprops={"zorder": 2}, whiskerprops={"zorder": 2},
                medianprops={"zorder": 3, "color": "black"},
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
                ax=ax3)
    ax3.set(xticks=[0, 1], xticklabels=["Current SotA", "Proposed Architecture"],
            ylabel="Test Accuracy", title="Test Accuracy Distribution")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('parity_mod3_qml_results.png', dpi=600, bbox_inches='tight')
    plt.show()

    # Save CSVs
    df_hist = pd.DataFrame({
        'Epoch': cnp.tile(epochs, N_REPEATS),
        'Current SotA Cross-Entropy': b_loss_all.flatten(),
        'Proposed Architecture Cross-Entropy': n_loss_all.flatten(),
        'Current SotA Gradient Norm': b_grad_all.flatten(),
        'Proposed Architecture Gradient Norm': n_grad_all.flatten(),
    })
    df_hist.to_csv('parity_mod3_loss_grad_history.csv', index=False)

    df_acc = pd.DataFrame({
        'Repeat': cnp.arange(N_REPEATS),
        'Current SotA Accuracy': acc_b_all,
        'Proposed Architecture Accuracy': acc_n_all,
    })
    df_acc.to_csv('parity_mod3_test_accuracy.csv', index=False)

    print("[MAIN] Done.")
