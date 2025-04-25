#!/usr/bin/env python3
# parallel_qml.py

import pennylane as qml
import pennylane.numpy as np
import numpy as cnp
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import pandas as pd
from joblib import Parallel, delayed
import concurrent.futures
import functools
print = functools.partial(print, flush=True)

# --------------------------------------------------
# Global constants & deterministic seeding
# --------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")
print("[INIT] Setting constants and seeds...")  
SEED_GLOBAL  = 42       
N_LAYERS     = 5
NUM_EPOCHS   = 200
N_REPEATS    = 100
N_TRAIN      = 1000
N_TEST       = 100
np.random.seed(SEED_GLOBAL)



# --------------------------------------------------
# Device setup
# --------------------------------------------------
print("[INIT] Initializing devices...")  
n_wires    = 6
dev_base   = qml.device("default.qubit", wires=n_wires)
dev_new    = qml.device("default.qubit", wires=n_wires)
dev_target = qml.device("default.qubit", wires=n_wires)



# --------------------------------------------------
# One global Haar unitary
# --------------------------------------------------
print("[INIT] Generating Haar unitary...") 
def random_haar_unitary(n):
    # standard QR method
    z = (np.random.normal(size=(n, n)) +
         1j * np.random.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d    = np.diagonal(r)
    ph   = d / np.abs(d)
    return q * ph
Haar_U = random_haar_unitary(2**n_wires)



# --------------------------------------------------
# QNode definitions
# --------------------------------------------------
print("[INIT] Defining QNodes...")  
@qml.qnode(dev_target, interface="autograd")
def target_state_circuit(angle):
    for w in range(n_wires):
        qml.RY(angle, wires=w)
    qml.QubitUnitary(Haar_U, wires=range(n_wires))
    return qml.state()


def target_state(angle):
    return target_state_circuit(angle)


@qml.qnode(dev_base, interface="autograd")
def baseline_channel_circuit(angle, weights):
    for w in range(n_wires):
        qml.RY(angle, wires=w)
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.state()


@qml.qnode(dev_new, interface="autograd")
def new_architecture_channel_circuit(angle, weights):
    for w in range((n_wires//2)+1):
        qml.RY(angle, wires=w) 
    for w in range((n_wires//2)+1, n_wires):
        qml.Hadamard(wires=w)
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.state()



# --------------------------------------------------
# Data generation
# --------------------------------------------------
print("[INIT] Generating training/test angles...") 
train_angles = np.random.uniform(0, np.pi, N_TRAIN)
test_angles  = np.random.uniform(0, np.pi, N_TEST)
assert np.unique(train_angles).size == N_TRAIN
assert np.unique(test_angles).size == N_TEST
assert np.intersect1d(train_angles, test_angles).size == 0


print("[INIT] Evaluating training/test target states...") 
train_states = {theta: target_state(theta) for theta in train_angles}
test_states  = {theta: target_state(theta) for theta in test_angles}
_repeat_seeds = np.random.randint(0, 1000000, size=N_REPEATS)



# --------------------------------------------------
# Helpers: fidelity, cost & eval
# --------------------------------------------------
def fidelity(s1, s2):
    inner = qml.math.sum(qml.math.conj(s1) * s2)
    return qml.math.real(inner * qml.math.conj(inner))


def cost_function_serial(circuit_fn, weights, angles):
    total_loss = 0.0
    for theta in angles:
        out_state = circuit_fn(theta, weights)
        total_loss += 1 - fidelity(train_states[theta], out_state)
    return total_loss / len(angles)


def evaluate_architecture_serial(circuit_fn, weights, angles):
    fids = []
    for theta in angles:
        out = circuit_fn(theta, weights)
        fids.append(fidelity(test_states[theta], out))
    return np.mean(fids)


def compute_stats(arr):
    return {
        "min":  arr.min(axis=0),
        "q1":   cnp.percentile(arr, 25, axis=0),
        "mean": arr.mean(axis=0),
        "q3":   cnp.percentile(arr, 75, axis=0),
        "max":  arr.max(axis=0),
    }



# --------------------------------------------------
# Single‐repeat worker
# --------------------------------------------------
def run_single_repeat(args):
    repeat_id, seed = args
    print(f"[REPEAT {repeat_id}] Starting with seed {seed}")  # [PRINT]

    np.random.seed(int(seed))

    w0 = np.random.uniform(0, 2*np.pi, size=(N_LAYERS, n_wires, 3), requires_grad=True)
    wb = w0.copy()
    wn = deepcopy(w0)

    opt = qml.AdamOptimizer(stepsize=0.001)

    loss_hist_b = []
    grad_hist_b = []
    loss_hist_n = []
    grad_hist_n = []

    for epoch in range(NUM_EPOCHS):
        Lb = cost_function_serial(baseline_channel_circuit, wb, train_angles)
        gb = qml.grad(lambda w: cost_function_serial(baseline_channel_circuit, w, train_angles))(wb)
        wb = opt.apply_grad(gb, wb)
        wb = np.array(wb, requires_grad=True)
        nb = np.linalg.norm(gb)
        loss_hist_b.append(Lb)
        grad_hist_b.append(nb)


        Ln = cost_function_serial(new_architecture_channel_circuit, wn, train_angles)
        gn = qml.grad(lambda w: cost_function_serial(new_architecture_channel_circuit, w, train_angles))(wn)
        wn = opt.apply_grad(gn, wn)
        wn = np.array(wn, requires_grad=True)
        nn = np.linalg.norm(gn)
        loss_hist_n.append(Ln)
        grad_hist_n.append(nn)
        print(f"[REPEAT {repeat_id}] Epoch {epoch + 1}/{NUM_EPOCHS}, Losses: Baseline={Lb:.4f}, New={Ln:.4f}; Gradient Norms: Baseline={nb:.4f}, New={nn:.4f}")


    fb = evaluate_architecture_serial(baseline_channel_circuit, wb, test_angles)
    fn = evaluate_architecture_serial(new_architecture_channel_circuit, wn, test_angles)
    print(f"[REPEAT {repeat_id}] Completed — Fids: Baseline={fb:.4f}, New={fn:.4f}")  
    return (loss_hist_b, grad_hist_b, loss_hist_n, grad_hist_n, fb, fn)


# --------------------------------------------------
# Main: dispatch repeats in parallel, then plot + save
# --------------------------------------------------
if __name__ == "__main__":
    print("[MAIN] Starting parallel training across repeats...") 
    args_list = list(enumerate(_repeat_seeds))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_repeat, args_list))
    print("[MAIN] All repeats completed. Aggregating results...") 

    # unpack
    baseline_loss_all = []
    baseline_grad_all = []
    new_loss_all      = []
    new_grad_all      = []
    fid_baseline_all  = []
    fid_new_all       = []

    for lb, gb, ln, gn, fb, fn in results:
        baseline_loss_all.append(lb)
        baseline_grad_all.append(gb)
        new_loss_all.append(ln)
        new_grad_all.append(gn)
        fid_baseline_all.append(fb)
        fid_new_all.append(fn)

    # to arrays
    baseline_loss_all = np.array(baseline_loss_all)
    baseline_grad_all = np.array(baseline_grad_all)
    new_loss_all      = np.array(new_loss_all)
    new_grad_all      = np.array(new_grad_all)

    # aggregate stats
    b_loss = compute_stats(baseline_loss_all)
    n_loss = compute_stats(new_loss_all)
    b_grad = compute_stats(baseline_grad_all)
    n_grad = compute_stats(new_grad_all)
    epochs = cnp.arange(1, NUM_EPOCHS + 1)

    print("[MAIN] Plotting results...") 
    plt.figure(figsize=(18, 6))
    plt.suptitle(f"Comparison over {N_REPEATS} runs", fontsize=16, y=0.98)

    # Loss
    ax1 = plt.subplot(1, 3, 1)
    for stats, label, col in [(b_loss, "Current SotA", "blue"),
                              (n_loss, "Proposed Architecture", "orange")]:
        sns.lineplot(x=epochs, y=stats["mean"], label=label, color=col, ax=ax1)
        ax1.fill_between(epochs, stats["min"], stats["max"], color=col, alpha=0.1)
        ax1.fill_between(epochs, stats["q1"], stats["q3"], color=col, alpha=0.3)
    ax1.set(title="Training Loss", xlabel="Epoch", ylabel="Cost (1−Fidelity)")

    # Grad norm
    ax2 = plt.subplot(1, 3, 2)
    for stats, label, col in [(b_grad, "Current SotA", "blue"),
                              (n_grad, "Proposed Architecture", "orange")]:
        sns.lineplot(x=epochs, y=stats["mean"], label=label, color=col, ax=ax2)
        ax2.fill_between(epochs, stats["min"], stats["max"], color=col, alpha=0.1)
        ax2.fill_between(epochs, stats["q1"], stats["q3"], color=col, alpha=0.3)
    ax2.set(title="Gradient Norm", xlabel="Epoch", ylabel="∥∇L∥")

    # Fidelity distribution
    ax3 = plt.subplot(1, 3, 3)
    sns.violinplot(data=[fid_baseline_all, fid_new_all],
                   palette=["blue", "orange"], inner=None, cut=0, ax=ax3, alpha=0.3)
    sns.boxplot(data=[fid_baseline_all, fid_new_all],
                width=0.15, palette=["blue", "orange"], showcaps=True,
                boxprops={"zorder": 2}, whiskerprops={"zorder": 2},
                medianprops={"zorder": 3, "color": "black"},
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
                ax=ax3)
    ax3.set(xticks=[0, 1], xticklabels=["Current SotA", "Proposed Architecture"],
            ylabel="Test Fidelity", title="Fidelity Distribution")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("qml_results.png", dpi=600, bbox_inches="tight")
    plt.show()

    # Save results
    df_all = pd.DataFrame({
        "Epoch":    cnp.tile(epochs, N_REPEATS),
        "Current SotA Loss": baseline_loss_all.flatten(),
        "Proposed Architecture Loss": new_loss_all.flatten(),
        "Current SotA Gradient Norm":   baseline_grad_all.flatten(),
        "Proposed Architecture Gradient Norm":   new_grad_all.flatten(),
    })
    df_all.to_csv("qml_results_haar.csv", index=False)

    df_fid = pd.DataFrame({
        "Repeat":               cnp.arange(1, N_REPEATS + 1),
        "Current SotA Fidelity":    fid_baseline_all,
        "Proposed Architecture Fidelity":    fid_new_all,
    })
    df_fid.to_csv("qml_results_haar_fidelities.csv", index=False)
    print("[MAIN] Done.")
