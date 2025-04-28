#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import optax

# --------------------------------------------------
# Global constants & deterministic seeding
# --------------------------------------------------
print("[INIT] Setting constants and seeds...")  
SEED_GLOBAL = 42
LAYERS      = 5
NUM_EPOCHS  = 200
N_REPEATS   = 100
N_TRAIN     = 4000
N_TEST      = 1000
STEPSIZE    = 1e-3
EPSILON     = 1e-6
WIRES       = 8



# --------------------------------------------------
# Dataset generation (pure NumPy)
# --------------------------------------------------
def generate_mod3_dataset(N, seed):
    rng = np.random.RandomState(seed)
    per = N // 3
    rem = N - 3 * per
    counts = [per + (1 if i < rem else 0) for i in range(3)]
    X, Y = [], []
    for cls, count in enumerate(counts):
        gen = 0
        while gen < count:
            x = rng.randint(0, 2, size=WIRES)
            if int(x.sum()) % 3 == cls:
                X.append(x)
                y = np.zeros(3)
                y[cls] = 1.0
                Y.append(y)
                gen += 1
    X = np.stack(X); Y = np.stack(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


print("[INIT] Generating training and test datasets...") 
train_X_np, train_Y_np = generate_mod3_dataset(N_TRAIN, SEED_GLOBAL)
test_X_np,  test_Y_np  = generate_mod3_dataset(N_TEST,  SEED_GLOBAL + 1)


print("[INIT] Moving to JAX training and test datasets...")
train_X = jnp.array(train_X_np)
train_Y = jnp.array(train_Y_np)
test_X  = jnp.array(test_X_np)
test_Y  = jnp.array(test_Y_np)



# --------------------------------------------------
# Quantum devices & QNodes with JAX interface
# --------------------------------------------------
print("[INIT] Defining QNodes...")  
dev = qml.device("default.qubit", wires=WIRES)


@qml.qnode(dev, interface="jax", diff_method="backprop")
def baseline_circuit(x, weights):
    for j in range(WIRES):
        qml.RX(jnp.pi * x[j], wires=j)
    qml.StronglyEntanglingLayers(weights, wires=range(WIRES))
    return qml.probs(wires=[0, 1])


@qml.qnode(dev, interface="jax", diff_method="backprop")
def proposed_circuit(x, weights):
    # Pairwise Ry-Rz encoding on first 4 wires
    for j in range(4):
        qml.RY(jnp.pi * x[j],   wires=j)
        qml.RZ(jnp.pi * x[j+4], wires=j)
    # Ancillas in |+>
    for j in range(4, WIRES):
        qml.Hadamard(wires=j)
    qml.StronglyEntanglingLayers(weights, wires=range(WIRES))
    return qml.probs(wires=[0, 1])


batched_baseline = jax.vmap(baseline_circuit,  in_axes=(0, None))
batched_proposed = jax.vmap(proposed_circuit, in_axes=(0, None))



# --------------------------------------------------
# Loss, gradient, and accuracy
# --------------------------------------------------
def cross_entropy(probs, targets):
    p = probs[:, :3] + EPSILON
    p = p / jnp.sum(p, axis=1, keepdims=True)
    return -jnp.mean(jnp.sum(targets * jnp.log(p), axis=1))


def loss_and_grads(weights, X, Y, circuit_fn):
    probs = circuit_fn(X, weights)
    loss  = cross_entropy(probs, Y)
    grads = jax.grad(lambda w: cross_entropy(circuit_fn(X, w), Y))(weights)
    return loss, grads


print("[INIT] Defining loss and gradient functions...")
baseline_step  = jax.jit(lambda w, X, Y: loss_and_grads(w, X, Y, batched_baseline))
proposed_step = jax.jit(lambda w, X, Y: loss_and_grads(w, X, Y, batched_proposed))


def accuracy(weights, X, Y, circuit_fn):
    probs = circuit_fn(X, weights)
    preds = jnp.argmax(probs[:, :3], axis=1)
    truths= jnp.argmax(Y, axis=1)
    return jnp.mean(preds == truths)



# --------------------------------------------------
# Single-run training (returns histories + accuracies)
# --------------------------------------------------
def train_one_repeat(seed, dict_map):
    key = jax.random.PRNGKey(int(seed))
    seed_number = dict_map[seed]
    wb = jax.random.uniform(key, (LAYERS, WIRES, 3), minval=0, maxval=2*jnp.pi)
    wn = wb.copy()

    # Optax optimizer setup
    tx = optax.adam(STEPSIZE)
    opt_state_b = tx.init(wb)
    opt_state_n = tx.init(wn)

    loss_hist_b, grad_hist_b = [], []
    loss_hist_n, grad_hist_n = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        lb, gb = baseline_step(jax.device_get(wb), train_X, train_Y)
        updates_b, opt_state_b = tx.update(gb, opt_state_b, params=wb)
        wb = optax.apply_updates(wb, updates_b)
        nb = jnp.linalg.norm(gb)

        ln, gn = proposed_step(jax.device_get(wn), train_X, train_Y)
        updates_n, opt_state_n = tx.update(gn, opt_state_n, params=wn)
        wn = optax.apply_updates(wn, updates_n)
        nn = jnp.linalg.norm(gn)

        loss_hist_b.append(float(lb)); grad_hist_b.append(float(nb))
        loss_hist_n.append(float(ln)); grad_hist_n.append(float(nn))

        print(f"[REPEAT {seed_number}] Epoch {epoch}/{NUM_EPOCHS}, Losses: Baseline={lb:.4f}, New={ln:.4f}; Gradient Norms: Baseline={nb:.4f}, New={nn:.4f}")


    acc_b = float(accuracy(wb, test_X, test_Y, batched_baseline))
    acc_n = float(accuracy(wn, test_X, test_Y, batched_proposed))
    print(f"[REPEAT {seed_number}] Completed — Accuracy: Baseline={acc_b:.4f}, New={acc_n:.4f}")  

    return loss_hist_b, grad_hist_b, loss_hist_n, grad_hist_n, acc_b, acc_n



# --------------------------------------------------
# Helper for summary stats
# --------------------------------------------------
def stats(all_runs):
    arr = np.array(all_runs)
    return {"min": np.min(arr, axis=0),
            "q1":  np.percentile(arr, 25, axis=0),
            "mean":np.mean(arr, axis=0),
            "q3":  np.percentile(arr, 75, axis=0),
            "max": np.max(arr, axis=0)}



# --------------------------------------------------
# Main: sequential repeats, stats, plots, CSVs
# --------------------------------------------------
if __name__ == "__main__":
    print("[MAIN] Starting training across repeats...") 
    seeds   = np.random.randint(0, 1_000_000, size=N_REPEATS)
    map_to_enumerate = {s: i for i, s in enumerate(seeds)}
    results = [train_one_repeat(s, map_to_enumerate) for s in seeds]

    # Unpack
    b_loss_all = np.array([r[0] for r in results])
    b_grad_all = np.array([r[1] for r in results])
    n_loss_all = np.array([r[2] for r in results])
    n_grad_all = np.array([r[3] for r in results])
    acc_b_all  = np.array([r[4] for r in results])
    acc_n_all  = np.array([r[5] for r in results])

    # Compute stats
    b_loss = stats(b_loss_all)
    n_loss = stats(n_loss_all)
    b_grad = stats(b_grad_all)
    n_grad = stats(n_grad_all)
    epochs = np.arange(1, NUM_EPOCHS + 1)

    # Plotting
    print("[MAIN] Plotting results...") 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Comparison over {N_REPEATS} runs", fontsize=16, y=0.98)

    # Training Loss
    for stats, label, color in [(b_loss, "Current SotA", "blue"), (n_loss, "Proposed Architecture", "orange")]:
        axes[0].plot(epochs, stats["mean"], label=label, color=color)
        axes[0].fill_between(epochs, stats["min"], stats["max"], color=color, alpha=0.1)
        axes[0].fill_between(epochs, stats["q1"], stats["q3"], color=color, alpha=0.3)
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Cross-Entropy")
    axes[0].legend()

    # Gradient Norm
    for stats, label, color in [(b_grad, "Current SotA", "blue"), (n_grad, "Proposed Architecture", "orange")]:
        axes[1].plot(epochs, stats["mean"], label=label, color=color)
        axes[1].fill_between(epochs, stats["min"], stats["max"], color=color, alpha=0.1)
        axes[1].fill_between(epochs, stats["q1"], stats["q3"], color=color, alpha=0.3)
    axes[1].set(title="Gradient Norm", xlabel="Epoch", ylabel="∥∇L∥")
    axes[1].legend()

    # Test Accuracy Distribution
    sns.violinplot(data=[acc_b_all, acc_n_all],
                palette=["blue", "orange"], inner=None, cut=0, ax=axes[2], alpha=0.3)
    sns.boxplot(data=[acc_b_all, acc_n_all],
                width=0.15, palette=["blue", "orange"], showcaps=True,
                boxprops={"zorder": 2}, whiskerprops={"zorder": 2},
                medianprops={"zorder": 3, "color": "black"},
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
                ax=axes[2])
    axes[2].set(xticks=[0, 1], xticklabels=["Current SotA", "Proposed Architecture"],
                title="Test Accuracy Distribution", ylabel="Accuracy")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("parity_mod3_qml_jax_results.png", dpi=600, bbox_inches="tight")
    plt.show()

    # Save CSVs
    df_hist = pd.DataFrame({
        "Epoch":             np.tile(epochs, N_REPEATS),
        "Baseline Loss":     b_loss_all.flatten(),
        "Proposed Loss":     n_loss_all.flatten(),
        "Baseline GradNorm": b_grad_all.flatten(),
        "Proposed GradNorm": n_grad_all.flatten(),
    })
    df_hist.to_csv("parity_mod3_loss_grad_history.csv", index=False)

    df_acc = pd.DataFrame({
        "Repeat":            np.arange(N_REPEATS),
        "Baseline Accuracy": acc_b_all,
        "Proposed Accuracy": acc_n_all,
    })
    df_acc.to_csv("parity_mod3_test_accuracy.csv", index=False)

    print("[MAIN] Done.")
