#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import optax
import functools
print = functools.partial(print, flush=True)

# --------------------------------------------------
# Global constants & deterministic seeding
# --------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

SEED_GLOBAL = 42
LAYERS      = 2
NUM_EPOCHS  = 200
N_REPEATS   = 100
STEPSIZE    = 1e-3
WIRES       = 6
N_TRAIN     = 2000
N_TEST      = 500


# Weierstrass parameters
A = 0.55
B = 5
N_MAX = 8

# --------------------------------------------------
# Dataset generation
# --------------------------------------------------
def generate_weierstrass(n_samples, seed):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, WIRES))

    @jax.jit
    def weier(x):
        total = 0.0
        for n in range(N_MAX + 1):
            total += A**n * jnp.prod(jnp.cos((B**n) * jnp.pi * x))
        return total

    C = (1 - A ** (N_MAX + 1)) / (1 - A)
    Y = jnp.array([weier(x) for x in X]) / C
    return jnp.array(X), Y

# --------------------------------------------------
# QNode definitions & batching
# --------------------------------------------------
dev = qml.device("default.mixed", wires=WIRES, p = 0.05)

@qml.qnode(dev, interface="jax", diff_method="backprop")
def baseline_circuit(x, weights):
    for j in range(4):
        qml.RY(x[j] * jnp.pi, wires=j)
        if j < 2:
            qml.RZ(x[j+4] * jnp.pi, wires=j)
    qml.StronglyEntanglingLayers(weights, wires=range(WIRES))
    for w in range(WIRES):
        qml.DepolarizingChannel(p, wires=w)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="jax", diff_method="backprop")
def proposed_circuit(x, weights, p = 0.05):
    for j in range(4):
        qml.RY(x[j] * jnp.pi, wires=j)
        if j < 2:
            qml.RZ(x[j+4] * jnp.pi, wires=j)
    for w in [4, 5]:
        qml.Hadamard(wires=w)
    qml.StronglyEntanglingLayers(weights, wires=range(WIRES))
    for w in range(WIRES):
        qml.DepolarizingChannel(p, wires=w)
    return qml.expval(qml.PauliZ(0))

batched_baseline = jax.vmap(baseline_circuit, in_axes=(0, None))
batched_proposed = jax.vmap(proposed_circuit, in_axes=(0, None))
print("[INIT] QNode definitions complete.")

# --------------------------------------------------
# Loss helper
# --------------------------------------------------
def mse_loss(preds, targets):
    return jnp.mean((preds - targets)**2)

# --------------------------------------------------
# Trainer builder
# --------------------------------------------------
opt = optax.adam(STEPSIZE)

def make_trainer(circuit_fn):
    def init_params(key):
        return jax.random.uniform(key, (LAYERS, WIRES, 3), minval=0.0, maxval=2*jnp.pi)

    @jax.jit
    def train(keys, X, Y):
        def single_run(key):
            params = init_params(key)
            state = opt.init(params)

            def step(carry, _):
                p, s = carry
                preds = circuit_fn(X, p)
                loss = mse_loss(preds, Y)
                grads = jax.grad(lambda w: mse_loss(circuit_fn(X, w), Y))(p)
                updates, s = opt.update(grads, s, p)
                p = optax.apply_updates(p, updates)
                gn = jnp.linalg.norm(grads)
                return (p, s), (loss, gn)

            (_, _), (loss_hist, grad_hist) = jax.lax.scan(
                step,
                (params, state),
                None,
                length=NUM_EPOCHS
            )
            return loss_hist, grad_hist, params

        loss_h, grad_h, finals = jax.vmap(single_run)(keys)
        return loss_h, grad_h, finals
    return train

# --------------------------------------------------
# Summary statistics
# --------------------------------------------------
def stats(arr):
    return {
    'min':   np.min(arr, axis=0),
    'q1':    np.percentile(arr, 25, axis=0),
    'mean':  np.mean(arr, axis=0),
    'q3':    np.percentile(arr, 75, axis=0),
    'max':   np.max(arr, axis=0),
}

# --------------------------------------------------
# Main script
# --------------------------------------------------
if __name__ == "__main__":
    print("[INIT] Generating datasets...")
    train_X, train_Y = generate_weierstrass(N_TRAIN, SEED_GLOBAL)
    test_X, test_Y   = generate_weierstrass(N_TEST,  SEED_GLOBAL + 1)
    print(f"[INIT] Data shapes: {train_X.shape}, {train_Y.shape}; {test_X.shape}, {test_Y.shape}")

    print("[INIT] Preparing trainers...")
    baseline_train = make_trainer(batched_baseline)
    proposed_train = make_trainer(batched_proposed)

    keys = jax.random.split(jax.random.PRNGKey(SEED_GLOBAL), N_REPEATS)

    print("[MAIN] Training baseline architecture...")
    loss_b_h, grad_b_h, fin_b = baseline_train(keys, train_X, train_Y)
    print("[MAIN] Training proposed architecture...")
    loss_n_h, grad_n_h, fin_n = proposed_train(keys, train_X, train_Y)

    # to NumPy
    loss_b_np = np.array(loss_b_h)
    grad_b_np = np.array(grad_b_h)
    loss_n_np = np.array(loss_n_h)
    grad_n_np = np.array(grad_n_h)
    pred_test_b = np.array(batched_baseline(test_X, fin_b))
    pred_test_b = pred_test_b.reshape(N_REPEATS, N_TEST)
    pred_test_n = np.array(batched_proposed(test_X, fin_n))
    pred_test_n = pred_test_n.reshape(N_REPEATS, N_TEST)
    test_b_all = np.array([mse_loss(pred_test_b[i], test_Y) for i in range(N_REPEATS)])
    test_n_all = np.array([mse_loss(pred_test_n[i], test_Y) for i in range(N_REPEATS)])

    b_loss = stats(loss_b_np)
    n_loss = stats(loss_n_np)
    b_grad = stats(grad_b_np)
    n_grad = stats(grad_n_np)
    epochs = np.arange(1, NUM_EPOCHS+1)

    print("[MAIN] Plotting results...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Comparison over {N_REPEATS} runs", fontsize=16)

    # --- Training Loss ---
    for stats_d, label, col in [
        (b_loss, "Current SotA", "blue"),
        (n_loss, "Proposed Architecture", "orange"),
    ]:
        sns.lineplot(
            x=epochs, y=stats_d['mean'],
            label=label, color=col, ax=ax1
        )
        ax1.fill_between(epochs, stats_d['min'], stats_d['max'], color=col, alpha=0.1)
        ax1.fill_between(epochs, stats_d['q1'], stats_d['q3'], color=col, alpha=0.3)
    ax1.set(
        title="Training Loss",
        xlabel="Epoch",
        ylabel="MSE Loss"
    )

    # --- Gradient Norm ---
    for stats_d, label, col in [
        (b_grad, "Current SotA", "blue"),
        (n_grad, "Proposed Architecture", "orange"),
    ]:
        sns.lineplot(
            x=epochs, y=stats_d['mean'],
            label=label, color=col, ax=ax2
        )
        ax2.fill_between(epochs, stats_d['min'], stats_d['max'], color=col, alpha=0.1)
        ax2.fill_between(epochs, stats_d['q1'], stats_d['q3'], color=col, alpha=0.3)
    ax2.set(
        title="Gradient Norm",
        xlabel="Epoch",
        ylabel="∥∇L∥"
    )

    # --- Test MSE Distribution ---
    # Use saturation to lighten the violin colors, then bump alpha on the patches if desired
    vp = sns.violinplot(
        data=[test_b_all, test_n_all],
        palette=["blue", "orange"],
        inner=None, cut=0,
        saturation=0.3,
        ax=ax3
    )
    # If you really need 30% transparency on the violins themselves:
    for body in vp.collections:
        body.set_alpha(0.3)

    sns.boxplot(
        data=[test_b_all, test_n_all],
        width=0.15,
        palette=["blue", "orange"],
        showcaps=True,
        boxprops={"zorder": 2},
        whiskerprops={"zorder": 2},
        medianprops={"zorder": 3, "color": "black"},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
        ax=ax3
    )
    ax3.set(
        xticks=[0, 1],
        xticklabels=["Current SotA", "Proposed Architecture"],
        ylabel="Test MSE",
        title="Test MSE Distribution"
    )

    # Make room for the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig("weierstrass_qml_results_noisy.png", dpi=600, bbox_inches="tight")
    plt.show()

    # Save CSVs
    df_loss = pd.DataFrame({
        'Epoch': np.tile(epochs, N_REPEATS),
        'Current SotA Loss': loss_b_np.flatten(),
        'Proposed Architecture Loss': loss_n_np.flatten(),
        'Current SotA Gradient Norm': grad_b_np.flatten(),
        'Proposed Architecture Gradient Norm': grad_n_np.flatten(),
    })
    df_loss.to_csv("qml_results_weierstrass_0_H.csv", index=False)

    df_test = pd.DataFrame({
        'Repeat': np.arange(1, N_REPEATS+1),
        'Current SotA Test MSE': test_b_all,
        'Proposed Architecture Test MSE': test_n_all,
    })
    df_test.to_csv("qml_results_weierstrass_test_mse_noisy.csv", index=False)

    print("[MAIN] Done.")
