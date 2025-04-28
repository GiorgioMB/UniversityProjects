#!/usr/bin/env python3
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
LAYERS      = 4
NUM_EPOCHS  = 200
N_REPEATS   = 100
N_TRAIN     = 3000
N_TEST      = 1000

np.random.seed(SEED_GLOBAL)
sns.set_theme(style="whitegrid", context="paper")

target_classes = ["GHZ", "W", "Biseparable", "Product"]

# --------------------------------------------------
# State preparation functions
# --------------------------------------------------
def state_ghz(phi):
    vec = np.zeros(8)
    vec[0] = np.cos(phi)
    vec[7] = np.sin(phi)
    return vec


def state_w(_=None):
    vec = np.zeros(8)
    for idx in [1, 2, 4]:
        vec[idx] = 1.0 / np.sqrt(3)
    return vec


def state_bi(theta):
    sub = np.array([np.cos(theta), 0, 0, np.sin(theta)])
    return np.kron(sub, [1.0, 0.0])


def state_prod(theta, phi):
    single = np.array([np.cos(theta), np.sin(theta) * np.exp(1j*phi)])
    vec = single
    for _ in range(2):
        vec = np.kron(vec, single)
    return vec

# --------------------------------------------------
# Generate dataset
# --------------------------------------------------
def generate_dataset(n_per_class):
    X, Y = [], []
    for c, cls in enumerate(target_classes):
        for _ in range(n_per_class):
            if cls == "GHZ":
                phi = np.random.uniform(0, 2*np.pi)
                psi = state_ghz(phi)
            elif cls == "W":
                psi = state_w()
            elif cls == "Biseparable":
                theta = np.random.uniform(0, np.pi/2)
                psi = state_bi(theta)
            else:
                theta = np.random.uniform(0, np.pi/2)
                phi   = np.random.uniform(0, 2*np.pi)
                psi = state_prod(theta, phi)
            X.append(psi)
            one_hot = np.zeros(4)
            one_hot[c] = 1.0
            Y.append(one_hot)
    X = np.stack(X)
    Y = np.stack(Y)
    idx = np.random.permutation(len(X))
    return X[idx], Y[idx]

train_X, train_Y = generate_dataset(N_TRAIN // 4)
test_X,  test_Y  = generate_dataset(N_TEST // 4)

# --------------------------------------------------
# Device and Schur isometry
# --------------------------------------------------
wires = 3

dev = qml.device("default.qubit", wires=wires)
dev2 = qml.device("default.qubit", wires=wires+1)

# Construct Schur transform
b0 = np.eye(8)[0]
b1 = (np.eye(8)[1] + np.eye(8)[2] + np.eye(8)[4]) / np.sqrt(3)
b2 = (np.eye(8)[3] + np.eye(8)[5] + np.eye(8)[6]) / np.sqrt(3)
b3 = np.eye(8)[7]
B = np.stack([b0, b1, b2, b3], axis=1)
U_svd, _, _ = np.linalg.svd(B, full_matrices=True)
Schur_U = U_svd.conj().T

# --------------------------------------------------
# QNode definitions
# --------------------------------------------------
@qml.qnode(dev, interface="autograd")
def baseline_circuit(psi, weights):
    qml.StatePrep(psi, wires=[0,1,2], normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=[0,1,2])
    return qml.probs(wires=[0,1])

@qml.qnode(dev2, interface="autograd")
def proposed_circuit(psi, weights):
    qml.StatePrep(psi, wires=[0,1,2], normalize=True)
    qml.QubitUnitary(Schur_U, wires=[0,1,2])
    qml.Hadamard(wires=3)
    qml.StronglyEntanglingLayers(weights, wires=[0,1,3])
    return qml.probs(wires=[0,1])

# --------------------------------------------------
# Helpers: loss, evaluation
# --------------------------------------------------
def cross_entropy(circuit_fn, weights, X, Y):
    loss = 0.0
    for x, y in zip(X, Y):
        probs = circuit_fn(x, weights)
        loss += -np.sum(y * np.log(probs + 1e-10))
    return loss / len(X)
    

def accuracy(circuit_fn, weights, X, Y):
    acc = 0
    for x, y in zip(X, Y):
        probs = circuit_fn(x, weights)
        pred = np.argmax(probs)
        true = np.argmax(y)
        if pred == true:
            acc += 1
    return acc / len(X)

# --------------------------------------------------
# Single-repeat worker
# --------------------------------------------------
def run_single_repeat(args):
    repeat_id, seed = args
    print(f"[REPEAT {repeat_id}] Seed={seed}")
    np.random.seed(int(seed))

    init_w = np.random.uniform(0, 2*np.pi, size=(LAYERS, wires, 3), requires_grad=True)
    wb = init_w.copy()
    wn = deepcopy(init_w)

    opt = qml.AdamOptimizer(stepsize=1e-3)
    loss_hist_b, grad_hist_b = [], []
    loss_hist_n, grad_hist_n = [], []

    for epoch in range(NUM_EPOCHS):
        # Baseline update
        Lb = cross_entropy(baseline_circuit, wb, train_X, train_Y)
        gb = qml.grad(lambda w: cross_entropy(baseline_circuit, w, train_X, train_Y))(wb)
        wb = opt.apply_grad(gb, wb)
        wb = np.array(wb, requires_grad=True)
        nb = np.linalg.norm(gb)
        loss_hist_b.append(Lb)
        grad_hist_b.append(nb)

        # Proposed update
        Ln = cross_entropy(proposed_circuit, wn, train_X, train_Y)
        gn = qml.grad(lambda w: cross_entropy(proposed_circuit, w, train_X, train_Y))(wn)
        wn = opt.apply_grad(gn, wn)
        wn = np.array(wn, requires_grad=True)
        nn = np.linalg.norm(gn)
        loss_hist_n.append(Ln)
        grad_hist_n.append(nn)

        print(f"[REPEAT {repeat_id}] Epoch {epoch+1}/{NUM_EPOCHS}, Losses: Baseline={Lb:.4f}, New={Ln:.4f}; "
              f"Gradient Norms: Baseline={nb:.4f}, New={nn:.4f}")

    # Test accuracy
    acc_b = accuracy(baseline_circuit, wb, test_X, test_Y)
    acc_n = accuracy(proposed_circuit, wn, test_X, test_Y)
    print(f"[REPEAT {repeat_id}] Test Acc Baseline={acc_b:.4f}, Proposed={acc_n:.4f}")

    return loss_hist_b, grad_hist_b, loss_hist_n, grad_hist_n, acc_b, acc_n

# --------------------------------------------------
# Main: parallel execution, stats & plotting
# --------------------------------------------------
if __name__ == "__main__":
    print("[MAIN] Starting repeats...")
    seeds = np.random.randint(0, 1e6, size=N_REPEATS)
    args = list(enumerate(seeds))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_repeat, args))

    # Aggregate
    b_loss_all = cnp.array([r[0] for r in results])
    b_grad_all = cnp.array([r[1] for r in results])
    n_loss_all = cnp.array([r[2] for r in results])
    n_grad_all = cnp.array([r[3] for r in results])
    acc_b_all  = cnp.array([r[4] for r in results])
    acc_n_all  = cnp.array([r[5] for r in results])

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
    epochs = cnp.arange(1, NUM_EPOCHS+1)

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
    plt.savefig('entanglement_qml_results.png', dpi=600, bbox_inches='tight')
    plt.show()


    # Save CSVs
    df_hist = pd.DataFrame({
        'Epoch': cnp.tile(epochs, N_REPEATS),
        'Current SotA Loss': b_loss_all.flatten(),
        'Proposed Architecture Loss': n_loss_all.flatten(),
        'Current SotA Gradient Norm': b_grad_all.flatten(),
        'Proposed Architecture Gradient Norm': n_grad_all.flatten(),
    })
    df_hist.to_csv('entanglement_loss_grad_history.csv', index=False)

    df_acc = pd.DataFrame({
        'Repeat': cnp.arange(N_REPEATS),
        "Current SotA Accuracy": acc_b_all,
        "Proposed Architecture Accuracy": acc_n_all,
    })
    df_acc.to_csv('entanglement_test_accuracy.csv', index=False)

    print("[MAIN] Done.")
