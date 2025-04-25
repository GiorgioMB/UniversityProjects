#!/usr/bin/env python3
# weierstrass_qml.py

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
N_LAYERS    = 4
NUM_EPOCHS  = 200
N_REPEATS   = 100
N_TRAIN     = 8000
N_TEST      = 2000
N_WORKERS   = 20

a = 0.55
b = 5
N_MAX = 8


np.random.seed(SEED_GLOBAL)


# Precompute normalization constant C = max_x |f(x)| = \n# \sum_{n=0}^{N_MAX} a^n (attained when cos(...) = 1)
C = (1 - a ** (N_MAX + 1)) / (1 - a)
print(f"[INIT] Normalization constant C = {C:.4f}")



# --------------------------------------------------
# Target function definition and Data generation
# --------------------------------------------------
def weierstrass(x):
    """
    Multivariate Weierstrass-type function on x in [0,1]^6.
    """
    total = 0.0
    for n in range(N_MAX + 1):
        total += a ** n * np.prod(np.cos((b ** n) * np.pi * x))
    return total


print("[INIT] Generating data...")
train_X = np.random.uniform(0.0, 1.0, size=(N_TRAIN, 6))
test_X  = np.random.uniform(0.0, 1.0, size=(N_TEST, 6))


# Compute labels and rescale
train_Y = np.array([weierstrass(x) for x in train_X]) / C
test_Y  = np.array([weierstrass(x) for x in test_X])  / C



# --------------------------------------------------
# Device setup
# --------------------------------------------------
n_wires   = 6
dev_base  = qml.device("default.qubit", wires=n_wires)
dev_new   = qml.device("default.qubit", wires=n_wires)



# --------------------------------------------------
# QNode definitions
# --------------------------------------------------
@qml.qnode(dev_base, interface="autograd")
def baseline_circuit(x, weights):
    for j in range(6):
        # Embed all 6 features via Y-rotations
        qml.RY(x[j] * np.pi, wires=j)
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    # Readout on qubit 0
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev_new, interface="autograd")
def new_circuit(x, weights):
    # Embed only first 4 features; leave qubits 4,5 as ancillas
    for j in range(4):
        qml.RY(x[j] * np.pi, wires=j)
        if j < 2:
            qml.RZ(x[j+4] * np.pi, wires=j)
    # Initialize ancillas in superposition
    for w in [4, 5]:
        qml.Hadamard(wires=w)
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    # Readout on qubit 0
    return qml.expval(qml.PauliZ(0))



# --------------------------------------------------
# Loss, gradient and evaluation functions
# --------------------------------------------------
def mse_loss(circuit, weights, X, Y):
    loss = 0.0
    for x, y in zip(X, Y):
        pred = circuit(x, weights)
        loss += (pred - y) ** 2
    return loss / len(X)


# Evaluate test MSE
def evaluate(circuit, weights, X, Y):
    return mse_loss(circuit, weights, X, Y)



# --------------------------------------------------
# Single-repeat worker
# --------------------------------------------------
def run_single_repeat(args):
    repeat_id, seed = args
    print(f"[REPEAT {repeat_id}] Seed = {seed}")
    np.random.seed(int(seed))

    # Initialize global weights and clone
    init_w = np.random.uniform(0, 2 * np.pi,
                               size=(N_LAYERS, n_wires, 3), requires_grad=True)
    wb = init_w.copy()
    wn = deepcopy(init_w)


    opt = qml.AdamOptimizer(stepsize=0.001)
    loss_hist_b = []
    grad_hist_b = []
    loss_hist_n = []
    grad_hist_n = []


    for epoch in range(NUM_EPOCHS):
        # Baseline update
        Lb = mse_loss(baseline_circuit, wb, train_X, train_Y)
        gb = qml.grad(lambda w: mse_loss(baseline_circuit, w, train_X, train_Y))(wb)
        wb = opt.apply_grad(gb, wb)
        wb = np.array(wb, requires_grad=True)
        nb = np.linalg.norm(gb)
        loss_hist_b.append(Lb)
        grad_hist_b.append(nb)


        # New architecture update
        Ln = mse_loss(new_circuit, wn, train_X, train_Y)
        gn = qml.grad(lambda w: mse_loss(new_circuit, w, train_X, train_Y))(wn)
        wn = opt.apply_grad(gn, wn)
        wn = np.array(wn, requires_grad=True)
        nn = np.linalg.norm(gn)
        loss_hist_n.append(Ln)
        grad_hist_n.append(nn)


        print(f"[REPEAT {repeat_id}] Epoch {epoch + 1}/{NUM_EPOCHS}, Losses: Baseline={Lb:.4f}, New={Ln:.4f}; Gradient Norms: Baseline={nb:.4f}, New={nn:.4f}")


    # Evaluate test MSE
    tb = evaluate(baseline_circuit, wb, test_X, test_Y)
    tn = evaluate(new_circuit, wn, test_X, test_Y)
    print(f"[REPEAT {repeat_id}] Test MSE: Baseline={tb:.4f}, New={tn:.4f}")
    return (loss_hist_b, grad_hist_b, loss_hist_n, grad_hist_n, tb, tn)



# --------------------------------------------------
# Main: parallel execution, stats & plotting
# --------------------------------------------------
if __name__ == "__main__":
    print("[MAIN] Starting repeats...")
    seeds = np.random.randint(0, 1e6, size=N_REPEATS)
    args = list(enumerate(seeds))


    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(executor.map(run_single_repeat, args))


    # Aggregate
    baseline_loss_all = cnp.array([r[0] for r in results])
    baseline_grad_all = cnp.array([r[1] for r in results])
    new_loss_all      = cnp.array([r[2] for r in results])
    new_grad_all      = cnp.array([r[3] for r in results])
    test_b_all        = cnp.array([r[4] for r in results])
    test_n_all        = cnp.array([r[5] for r in results])


    # Compute statistics
    def stats(arr):
        return {
            'min': arr.min(axis=0),
            'q1':  cnp.percentile(arr, 25, axis=0),
            'mean':arr.mean(axis=0),
            'q3':  cnp.percentile(arr, 75, axis=0),
            'max': arr.max(axis=0),
        }


    b_loss = stats(baseline_loss_all)
    n_loss = stats(new_loss_all)
    b_grad = stats(baseline_grad_all)
    n_grad = stats(new_grad_all)
    epochs = cnp.arange(1, NUM_EPOCHS + 1)


    # Plot
    print("[MAIN] Plotting results...") 
    plt.figure(figsize=(18, 6))
    plt.suptitle(f"Comparison over {N_REPEATS} runs", fontsize=16, y=0.98)

    # Training Loss
    ax1 = plt.subplot(1, 3, 1)
    for stats, label, col in [(b_loss, "Current SotA", "blue"),
                            (n_loss, "Proposed Architecture", "orange")]:
        sns.lineplot(x=epochs, y=stats["mean"], label=label, color=col, ax=ax1)
        ax1.fill_between(epochs, stats["min"], stats["max"], color=col, alpha=0.1)
        ax1.fill_between(epochs, stats["q1"], stats["q3"], color=col, alpha=0.3)
    ax1.set(title="Training Loss", xlabel="Epoch", ylabel="MSE Loss")

    # Gradient Norm
    ax2 = plt.subplot(1, 3, 2)
    for stats, label, col in [(b_grad, "Current SotA", "blue"),
                            (n_grad, "Proposed Architecture", "orange")]:
        sns.lineplot(x=epochs, y=stats["mean"], label=label, color=col, ax=ax2)
        ax2.fill_between(epochs, stats["min"], stats["max"], color=col, alpha=0.1)
        ax2.fill_between(epochs, stats["q1"], stats["q3"], color=col, alpha=0.3)
    ax2.set(title="Gradient Norm", xlabel="Epoch", ylabel="∥∇L∥")

    # Test MSE Distribution (Violin + Box)
    ax3 = plt.subplot(1, 3, 3)
    sns.violinplot(data=[test_b_all, test_n_all],
                palette=["blue", "orange"], inner=None, cut=0, ax=ax3, alpha=0.3)
    sns.boxplot(data=[test_b_all, test_n_all],
                width=0.15, palette=["blue", "orange"], showcaps=True,
                boxprops={"zorder": 2}, whiskerprops={"zorder": 2},
                medianprops={"zorder": 3, "color": "black"},
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
                ax=ax3)
    ax3.set(xticks=[0, 1], xticklabels=["Current SotA", "Proposed Architecture"],
            ylabel="Test MSE", title="Test MSE Distribution")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("weierstrass_qml_results.png", dpi=600, bbox_inches="tight")
    plt.show()



    # Save CSVs
    df_loss = pd.DataFrame({
        'Epoch': cnp.tile(epochs, N_REPEATS),
        'Current SotA Loss': baseline_loss_all.flatten(),
        'Proposed Architecture Loss': new_loss_all.flatten(),
        'Current SotA Gradient Norm': baseline_grad_all.flatten(),
        'Proposed Architecture Gradient Norm': new_grad_all.flatten(),
    })
    df_loss.to_csv("qml_results_weierstrass.csv", index=False)


    df_test = pd.DataFrame({
        'Repeat': cnp.arange(1, N_REPEATS+1),
        'Current SotA Test MSE': test_b_all,
        'Proposed Architecture Test MSE': test_n_all,
    })
    df_test.to_csv("qml_results_weierstrass_test_mse.csv", index=False)
    print("[MAIN] Done.")
