#!/usr/bin/env python3
import functools
import warnings
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import pandas as pd
import pennylane as qml
import seaborn as sns
from qiskit.quantum_info import random_clifford
print = functools.partial(print, flush=True)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")



# --------------------------------------------------
# Global constants & deterministic seeding
# --------------------------------------------------
SEED_GLOBAL = 42
N_LAYERS: int = 5
NUM_EPOCHS: int = 200
N_REPEATS: int = 100
CHUNK_SIZE: int = 50         
N_TRAIN: int = 1000
N_TEST: int = 100
STEPSIZE: float = 1e-3
sns.set_theme(style="whitegrid", context="paper")



# --------------------------------------------------
# Device setup
# --------------------------------------------------
N_WIRES = 6
DEV = qml.device("default.qubit", wires=N_WIRES)



# --------------------------------------------------
# One global Clifford unitary (Haar-random in the Clifford group)
# --------------------------------------------------
print("[INIT] Generating Clifford unitary…")
CLIFFORD_OP = random_clifford(N_WIRES, seed=SEED_GLOBAL)
CLIFF_U_NP = CLIFFORD_OP.to_matrix()
CLIFF_U = jnp.array(CLIFF_U_NP)   



# --------------------------------------------------
# QNode definitions (JAX interface)
# --------------------------------------------------
print("[INIT] Defining QNodes…")


@qml.qnode(DEV, interface="jax", diff_method="backprop")
def target_state_circuit(angle: float):
    """Produces the target state |ψ(θ)⟩ ≔ U_C · RY(θ)^{⊗n}|0⟩"""
    for w in range(N_WIRES):
        qml.RY(angle, wires=w)
    qml.QubitUnitary(CLIFF_U, wires=range(N_WIRES))
    return qml.state()



@qml.qnode(DEV, interface="jax", diff_method="backprop")
def baseline_circuit(angle: float, weights):
    for w in range((N_WIRES // 2) + 1):
        qml.RY(angle, wires=w)
    qml.StronglyEntanglingLayers(weights, wires=range(N_WIRES))
    return qml.state()



@qml.qnode(DEV, interface="jax", diff_method="backprop")
def proposed_circuit(angle: float, weights):
    for w in range((N_WIRES // 2) + 1):
        qml.RY(angle, wires=w)
    for w in range((N_WIRES // 2) + 1, N_WIRES):
        qml.Hadamard(wires=w)
    qml.StronglyEntanglingLayers(weights, wires=range(N_WIRES))
    return qml.state()



# --------------------------------------------------
# Batched / JIT-compiled kernels (angle dimension is mapped, weights shared)
# --------------------------------------------------
print("[INIT] Creating vmapped kernels…")
vm_target = jax.jit(jax.vmap(target_state_circuit, in_axes=(0,)))
vm_baseline = jax.jit(jax.vmap(baseline_circuit, in_axes=(0, None)))
vm_proposed = jax.jit(jax.vmap(proposed_circuit, in_axes=(0, None)))



# --------------------------------------------------
# Helpers: fidelity, loss, statistics
# --------------------------------------------------
@jax.jit
def fidelity(psi, phi):
    """F(|ψ⟩,|φ⟩) = |⟨ψ|φ⟩|²  – compatible with batching."""
    inner = jnp.vdot(psi, phi)
    return jnp.real(inner * jnp.conj(inner))


@jax.jit
def loss_fn(weights, circuit_fn, ref_states, angles):
    preds = circuit_fn(angles, weights)            # (M, 64)
    fids = jax.vmap(fidelity)(ref_states, preds)   # (M,)
    return jnp.mean(1.0 - fids)


def stat_summary(arr):
    """Return dict with min, q1, mean, q3, max along axis 0."""
    return {
        "min": jnp.min(arr, axis=1),
        "q1": jnp.percentile(arr, 25, axis=1),
        "mean": jnp.mean(arr, axis=1),
        "q3": jnp.percentile(arr, 75, axis=1),
        "max": jnp.max(arr, axis=1),
    }



# --------------------------------------------------
# Chunked trainer builder (vectorised over repeats)
# --------------------------------------------------
OPTIMIZER = optax.adam(STEPSIZE)


def make_trainer(circuit_fn, ref_states, angles):
    """Return a jit-compiled function that trains CHUNK_SIZE repeats in parallel."""


    def init_params(rngs):  # rngs shape (CHUNK_SIZE, 2)
        def one_key(k):
            return jax.random.uniform(
                k, (N_LAYERS, N_WIRES, 3), minval=0.0, maxval=2 * jnp.pi
            )


        params = jax.vmap(one_key)(rngs)
        opt_states = jax.vmap(OPTIMIZER.init)(params)
        return params, opt_states


    @jax.jit
    def step(params, opt_states):
        # Compute losses & grads for every repeat in the chunk
        losses, grads = jax.vmap(
            lambda p: jax.value_and_grad(loss_fn)(p, circuit_fn, ref_states, angles)
        )(params)
        updates, new_opt = jax.vmap(OPTIMIZER.update)(grads, opt_states, params)
        new_params = jax.vmap(optax.apply_updates)(params, updates)
        grad_norms = jnp.linalg.norm(jnp.reshape(grads, (grads.shape[0], -1)), axis=1)
        return new_params, new_opt, losses, grad_norms


    def train_chunk(rng_chunk):
        params, opt_states = init_params(rng_chunk)


        def epoch_body(carry, _):
            p, o = carry
            p2, o2, L, G = step(p, o)
            return (p2, o2), (L, G)


        (final_params, _), (loss_hist, grad_hist) = jax.lax.scan(
            epoch_body, (params, opt_states), None, length=NUM_EPOCHS
        )
        return final_params, loss_hist, grad_hist


    return train_chunk



# --------------------------------------------------
# Main: data generation, training loop, evaluation, plotting & export
# --------------------------------------------------
if __name__ == "__main__":
    # -----------------------------
    # Training / test data
    # -----------------------------
    print("[INIT] Generating training / test angles…")
    np_rng = jax.random.PRNGKey(SEED_GLOBAL)
    train_angles = jax.random.uniform(np_rng, (N_TRAIN,), minval=0.0, maxval=jnp.pi)
    np_rng, _ = jax.random.split(np_rng)


    # Ensure test-set angles are disjoint from train set
    while True:
        test_angles = jax.random.uniform(np_rng, (N_TEST,), minval=0.0, maxval=jnp.pi)
        if len(jnp.intersect1d(train_angles, test_angles)) == 0:
            break


    # Pre-compute target states (no gradients needed ⇒ stop_gradient)
    print("[INIT] Computing target states…")
    train_states = jax.lax.stop_gradient(vm_target(train_angles))  # (N_TRAIN, 64)
    test_states = jax.lax.stop_gradient(vm_target(test_angles))



    # -----------------------------
    # Prepare trainers
    # -----------------------------
    print("[INIT] Preparing trainers…")
    train_baseline = make_trainer(vm_baseline, train_states, train_angles)
    train_proposed = make_trainer(vm_proposed, train_states, train_angles)



    # -----------------------------
    # Parallel training over repeats
    # -----------------------------
    print("[MAIN] Starting training over repeats…")
    master_key = jax.random.PRNGKey(SEED_GLOBAL)
    all_keys = jax.random.split(master_key, N_REPEATS)


    loss_b_chunks, grad_b_chunks, final_b_chunks = [], [], []
    loss_p_chunks, grad_p_chunks, final_p_chunks = [], [], []


    for idx in range(0, N_REPEATS, CHUNK_SIZE):
        chunk_id = idx // CHUNK_SIZE + 1
        rng_chunk = all_keys[idx : idx + CHUNK_SIZE]
        print(
            f"[MAIN] Chunk {chunk_id}/{(N_REPEATS + CHUNK_SIZE - 1)//CHUNK_SIZE}…"
        )


        # Baseline
        params_b, loss_b, grad_b = train_baseline(rng_chunk)
        # Proposed
        params_p, loss_p, grad_p = train_proposed(rng_chunk)


        loss_b_chunks.append(jnp.array(loss_b))  # (EPOCHS, CHUNK_SIZE)
        grad_b_chunks.append(jnp.array(grad_b))
        final_b_chunks.append(jnp.array(params_b))  # (CHUNK_SIZE, …)


        loss_p_chunks.append(jnp.array(loss_p))
        grad_p_chunks.append(jnp.array(grad_p))
        final_p_chunks.append(jnp.array(params_p))


    print("[MAIN] Training complete – stacking results…")


    # -----------------------------
    # Stack histories & final params
    # -----------------------------
    loss_b_all = jnp.concatenate([x.T for x in loss_b_chunks], axis=0)  # (R, E)
    grad_b_all = jnp.concatenate([x.T for x in grad_b_chunks], axis=0)
    loss_p_all = jnp.concatenate([x.T for x in loss_p_chunks], axis=0)
    grad_p_all = jnp.concatenate([x.T for x in grad_p_chunks], axis=0)


    final_b_all = jnp.concatenate(final_b_chunks, axis=0)  # (N_REPEATS, …)
    final_p_all = jnp.concatenate(final_p_chunks, axis=0)


    # -----------------------------
    # Evaluation on test set
    # -----------------------------
    print("[MAIN] Evaluating fidelities on test set…")


    @jax.jit
    def evaluate_all(params, circuit_fn):
        def fid_one(p):
            preds = circuit_fn(test_angles, p)
            fids = jax.vmap(fidelity)(test_states, preds)
            return jnp.mean(fids)

        return jax.vmap(fid_one)(params)


    fid_b_all = jnp.array(evaluate_all(final_b_all, vm_baseline))
    fid_p_all = jnp.array(evaluate_all(final_p_all, vm_proposed))


    # -----------------------------
    # Statistics & plotting
    # -----------------------------
    print("[MAIN] Computing statistics & preparing plots…")
    epochs = jnp.arange(1, NUM_EPOCHS + 1)
    b_loss_stats = stat_summary(loss_b_all)
    p_loss_stats = stat_summary(loss_p_all)


    b_grad_stats = stat_summary(grad_b_all)
    p_grad_stats = stat_summary(grad_p_all)


    plt.figure(figsize=(18, 6))
    plt.suptitle(f"Comparison over {N_REPEATS} repeats", fontsize=16, y=0.98)


    # Loss history
    ax1 = plt.subplot(1, 3, 1)
    for stats, label, col in [
        (b_loss_stats, "Current SotA", "blue"),
        (p_loss_stats, "Proposed Architecture", "orange"),
    ]:
        ax1.plot(epochs, stats["mean"], label=label, color=col)
        ax1.fill_between(epochs, stats["min"], stats["max"], color=col, alpha=0.1)
        ax1.fill_between(epochs, stats["q1"], stats["q3"], color=col, alpha=0.3)
    ax1.set(title="Training Loss", xlabel="Epoch", ylabel="1 – Fidelity")
    ax1.legend()


    # Gradient norm history
    ax2 = plt.subplot(1, 3, 2)
    for stats, label, col in [
        (b_grad_stats, "Current SotA", "blue"),
        (p_grad_stats, "Proposed Architecture", "orange"),
    ]:
        ax2.plot(epochs, stats["mean"], label=label, color=col)
        ax2.fill_between(epochs, stats["min"], stats["max"], color=col, alpha=0.1)
        ax2.fill_between(epochs, stats["q1"], stats["q3"], color=col, alpha=0.3)
    ax2.set(title="Gradient Norm", xlabel="Epoch", ylabel="∥∇L∥")
    ax2.legend()


    # Fidelity violin/box
    ax3 = plt.subplot(1, 3, 3)
    sns.violinplot(
        data=[fid_b_all, fid_p_all],
        palette=["blue", "orange"],
        inner=None,
        cut=0,
        ax=ax3,
        alpha=0.3,
    )
    sns.boxplot(
        data=[fid_b_all, fid_p_all],
        width=0.15,
        palette=["blue", "orange"],
        showcaps=True,
        boxprops={"zorder": 2},
        whiskerprops={"zorder": 2},
        medianprops={"zorder": 3, "color": "black"},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
        ax=ax3,
    )
    ax3.set(
        xticks=[0, 1],
        xticklabels=["Current SotA", "Proposed Architecture"],
        ylabel="Test Fidelity",
        title="Fidelity Distribution",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("clifford_qml_jax_results.png", dpi=600, bbox_inches="tight")
    plt.show()


    # -----------------------------
    # CSV exports
    # -----------------------------
    print("[MAIN] Saving CSV files…")


    df_hist = pd.DataFrame(
        {
            "Epoch": jnp.tile(epochs, N_REPEATS),
            "Current SotA Loss": loss_b_all.flatten(),
            "Proposed Architecture Loss": loss_p_all.flatten(),
            "Current SotA Gradient Norm": grad_b_all.flatten(),
            "Proposed Architecture Gradient Norm": grad_p_all.flatten(),
        }
    )
    df_hist.to_csv("qml_clifford_loss_grad_history_jax_0_H.csv", index=False)


    df_fid = pd.DataFrame(
        {
            "Repeat": jnp.arange(N_REPEATS),
            "Current SotA Fidelity": fid_b_all,
            "Proposed Architecture Fidelity": fid_p_all,
        }
    )
    df_fid.to_csv("qml_clifford_test_fidelity_jax_0_H.csv", index=False)
    print("[MAIN] Done.")
