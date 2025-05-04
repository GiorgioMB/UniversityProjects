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
LAYERS      = 5
NUM_EPOCHS  = 200
N_REPEATS   = 100
CHUNK_SIZE  = 10  
N_TRAIN     = 2600
N_TEST      = 650
STEPSIZE    = 1e-3
EPSILON     = 1e-6
WIRES       = 14
N_MODULUS   = 89 * 61  # Any semiprime in 2^14 would work 



# --------------------------------------------------
# Dataset generation
# --------------------------------------------------
def generate_quadratic_res_dataset(n_samples, seed, invalid_set= None):
    rng = np.random.RandomState(seed)
    residues = set((z * z) % N_MODULUS for z in range(N_MODULUS))


    invalid_codes = set()
    if invalid_set is not None:
        for item in invalid_set:
            arr = np.asarray(item)
            if arr.ndim == 1 and arr.size == WIRES:
                code = int(arr.dot(2**np.arange(WIRES)))
            else:
                code = int(item)
            invalid_codes.add(code)


    total_codes = 1 << WIRES  # 2**WIRES
    all_codes = np.arange(total_codes)


    mods      = all_codes % N_MODULUS
    is_res    = np.isin(mods, list(residues))
    is_valid  = ~np.isin(all_codes, list(invalid_codes))


    res_codes = all_codes[is_res    & is_valid]
    non_codes = all_codes[~is_res   & is_valid]


    per_class = n_samples // 2
    if len(res_codes) < per_class:
        raise ValueError(f"Only {len(res_codes)} valid residues, need {per_class}")
    if len(non_codes) < per_class:
        raise ValueError(f"Only {len(non_codes)} valid non-residues, need {per_class}")


    pick_res = rng.choice(res_codes,    per_class, replace=False)
    pick_non = rng.choice(non_codes,    per_class, replace=False)


    codes    = np.concatenate([pick_res, pick_non])
    labels   = np.vstack([
        np.tile([0.0, 1.0], (per_class, 1)),  
        np.tile([1.0, 0.0], (per_class, 1)),  
    ])


    bits = ((codes[:, None] >> np.arange(WIRES)) & 1).astype(int)


    perm = rng.permutation(n_samples)
    return bits[perm], labels[perm]

# --------------------------------------------------
# QNode definitions & batching
# --------------------------------------------------
dev = qml.device("default.qubit", wires=WIRES)


@qml.qnode(dev, interface="jax", diff_method="backprop")
def baseline_circuit(x, weights):
    for j in range(WIRES):
        qml.RX(jnp.pi * x[j], wires=j)


    qml.StronglyEntanglingLayers(weights, wires=range(WIRES))
    return qml.probs(wires=[0])

@qml.qnode(dev, interface="jax", diff_method="backprop")
def proposed_circuit(x, weights):
    half = WIRES // 2
    for j in range(half):
        qml.RY(jnp.pi * x[j], wires=j)
        qml.RZ(jnp.pi * x[j+half], wires=j)
    

    for j in range(half, WIRES):
        qml.Hadamard(wires=j)


    qml.StronglyEntanglingLayers(weights, wires=range(WIRES))
    return qml.probs(wires=[0])


batched_baseline = jax.vmap(baseline_circuit, in_axes=(0, None))
batched_proposed = jax.vmap(proposed_circuit, in_axes=(0, None))
print("[INIT] QNode definitions complete.")



# --------------------------------------------------
# Loss and accuracy helpers
# --------------------------------------------------
def binary_cross_entropy(probs, targets):
    p = probs + EPSILON
    p = p / jnp.sum(p, axis=1, keepdims=True)
    return -jnp.mean(jnp.sum(targets * jnp.log(p), axis=1))


def accuracy_all(params, circuit_fn, X, Y):


    def acc_one(w):
        probs = jax.vmap(lambda x: circuit_fn(x, w))(X)
        preds = jnp.argmax(probs, axis=1)
        truths = jnp.argmax(Y, axis=1)
        return jnp.mean(preds == truths)
    

    return jax.vmap(acc_one)(params)



# --------------------------------------------------
# Trainer builder for a chunk of repeats
# --------------------------------------------------
opt = optax.adam(STEPSIZE)


def make_chunked_trainer(circuit_fn):


    def init_all(keys):
        params0 = jax.vmap(lambda k: jax.random.uniform(
            k,
            (LAYERS, WIRES, 3),
            minval=0.0,
            maxval=2 * jnp.pi
        ))(keys)
        opt_state0 = jax.vmap(opt.init)(params0)
        return params0, opt_state0


    def train_chunk(keys, train_X, train_Y):
        # keys: shape (CHUNK_SIZE,)
        params0, opt0 = init_all(keys)


        def one_step(param, state):
            probs = circuit_fn(train_X, param)
            loss = binary_cross_entropy(probs, train_Y)
            grads = jax.grad(lambda p: binary_cross_entropy(
                circuit_fn(train_X, p), train_Y))(param)
            updates, new_state = opt.update(grads, state, param)
            new_param = optax.apply_updates(param, updates)
            gn = jnp.linalg.norm(grads)
            return new_param, new_state, loss, gn


        def epoch_step(carry, _):
            params, opt_states = carry
            new_params, new_states, losses, grad_norms = jax.vmap(
                one_step, in_axes=(0,0)
            )(params, opt_states)
            return (new_params, new_states), (losses, grad_norms)


        (final_params, _), (loss_hist, grad_hist) = jax.lax.scan(
            epoch_step,
            (params0, opt0),
            None,
            length=NUM_EPOCHS
        )


        return final_params, loss_hist, grad_hist


    return jax.jit(train_chunk)



# --------------------------------------------------
# Summary statistics helper
# --------------------------------------------------
def stats(arr):
    return {
        "min":  np.min(arr, axis=1),
        "q1":   np.percentile(arr, 25, axis=1),
        "mean": np.mean(arr, axis=1),
        "q3":   np.percentile(arr, 75, axis=1),
        "max":  np.max(arr, axis=1),
    }



# --------------------------------------------------
# Main script
# --------------------------------------------------
if __name__ == "__main__":
    print("[INIT] Generating datasets...")
    train_X_np, train_Y_np = generate_quadratic_res_dataset(N_TRAIN, SEED_GLOBAL)
    print("[INIT] Train set generated.")
    test_X_np,  test_Y_np  = generate_quadratic_res_dataset(N_TEST,  SEED_GLOBAL + 1, invalid_set=train_X_np)
    print("[INIT] Test set generated.")


    print("[INIT] Dataset sanity checks...")
    train_rows = set(map(tuple, train_X_np))
    assert all(tuple(row) not in train_rows for row in test_X_np), \
    "Test set contains training samples!"
    uniq_train = np.unique(train_X_np, axis=0)
    assert len(uniq_train) == N_TRAIN, \
        f"Training set contains duplicates!  Got only {len(uniq_train)} unique rows."
    uniq_test = np.unique(test_X_np, axis=0)
    assert len(uniq_test) == N_TEST, \
        f"Test set contains duplicates!  Got only {len(uniq_test)} unique rows."
    print("[INIT] Dataset sanity checks passed.")
    

    print("[INIT] Data generated, number of samples:")
    print(f"  Train: {train_X_np.shape[0]}, of which {np.sum(train_Y_np[:, 1]):.0f} are 1 and {np.sum(train_Y_np[:, 0]):.0f} are 0")
    print(f"  Test:  {test_X_np.shape[0]}, of which {np.sum(test_Y_np[:, 1]):.0f} are 1 and {np.sum(test_Y_np[:, 0]):.0f} are 0")
    

    print("[INIT] Moving data to JAX...")
    train_X = jnp.array(train_X_np)
    train_Y = jnp.array(train_Y_np)
    test_X  = jnp.array(test_X_np)
    test_Y  = jnp.array(test_Y_np)


    print("[INIT] Preparing trainers...")
    baseline_trainer = make_chunked_trainer(batched_baseline)
    proposed_trainer = make_chunked_trainer(batched_proposed)


    master_key = jax.random.PRNGKey(SEED_GLOBAL)
    all_keys = jax.random.split(master_key, N_REPEATS)


    loss_b_chunks, loss_n_chunks = [], []
    grad_b_chunks, grad_n_chunks = [], []
    final_b_chunks, final_n_chunks = [], []

    print("[MAIN] Starting chunked training...")
    for idx in range(0, N_REPEATS, CHUNK_SIZE):
        ck = all_keys[idx:idx+CHUNK_SIZE]
        print(f"[MAIN] Training chunk {idx//CHUNK_SIZE + 1}/{(N_REPEATS//CHUNK_SIZE)}...")
        # Baseline chunk
        fb, lb, gb = baseline_trainer(ck, train_X, train_Y)
        # Proposed chunk
        fn, ln, gn = proposed_trainer(ck, train_X, train_Y)

        # Move to NumPy
        lb_np = np.array(lb)
        gb_np = np.array(gb)
        fb_np = np.array(fb)
        ln_np = np.array(ln)
        gn_np = np.array(gn)
        fn_np = np.array(fn)

        # Store
        loss_b_chunks.append(lb_np)
        grad_b_chunks.append(gb_np)
        final_b_chunks.append(fb_np)
        loss_n_chunks.append(ln_np)
        grad_n_chunks.append(gn_np)
        final_n_chunks.append(fn_np)

        print(f"[MAIN] Chunk {idx//CHUNK_SIZE + 1} done.")


    loss_b = np.concatenate(loss_b_chunks, axis=1)
    grad_b = np.concatenate(grad_b_chunks, axis=1)
    final_b = np.concatenate(final_b_chunks, axis=0)

    loss_n = np.concatenate(loss_n_chunks, axis=1)
    grad_n = np.concatenate(grad_n_chunks, axis=1)
    final_n = np.concatenate(final_n_chunks, axis=0)


    print("[MAIN] Computing summary statistics...")
    b_loss_stats = stats(loss_b)
    n_loss_stats = stats(loss_n)
    b_grad_stats = stats(grad_b)
    n_grad_stats = stats(grad_n)
    epochs = np.arange(1, NUM_EPOCHS + 1)


    print("[MAIN] Computing test accuracies...")
    acc_b_all = np.array(accuracy_all(final_b, baseline_circuit, test_X, test_Y))
    acc_n_all = np.array(accuracy_all(final_n, proposed_circuit, test_X, test_Y))


    print("[MAIN] Plotting results...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Comparison over {N_REPEATS} repeats", fontsize=16, y=0.98)


    for stats_dict, label, col in [(b_loss_stats, "Current SotA", "blue"),
                                   (n_loss_stats, "Proposed Architecture", "orange")]:
        axes[0].plot(epochs, stats_dict["mean"], label=label, color=col)
        axes[0].fill_between(epochs, stats_dict["min"], stats_dict["max"], color=col, alpha=0.1)
        axes[0].fill_between(epochs, stats_dict["q1"], stats_dict["q3"], color=col, alpha=0.3)
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Cross-Entropy")
    axes[0].legend()


    for stats_dict, label, col in [(b_grad_stats, "Current SotA", "blue"),
                                   (n_grad_stats, "Proposed Architecture", "orange")]:
        axes[1].plot(epochs, stats_dict["mean"], label=label, color=col)
        axes[1].fill_between(epochs, stats_dict["min"], stats_dict["max"], color=col, alpha=0.1)
        axes[1].fill_between(epochs, stats_dict["q1"], stats_dict["q3"], color=col, alpha=0.3)
    axes[1].set(title="Gradient Norm", xlabel="Epoch", ylabel="∥∇L∥")
    axes[1].legend()


    sns.violinplot(data=[acc_b_all, acc_n_all], palette=["blue", "orange"], inner=None, cut=0, ax=axes[2], alpha=0.3)
    sns.boxplot(data=[acc_b_all, acc_n_all], width=0.15,
                palette=["blue", "orange"], showcaps=True,
                boxprops={"zorder":2}, whiskerprops={"zorder":2},
                medianprops={"zorder":3, "color":"black"},
                flierprops={"marker":"o","markersize":4,"alpha":0.6},
                ax=axes[2])
    axes[2].set(xticks=[0,1], xticklabels=["Current SotA","Proposed Architecture"],
                title="Test Accuracy Distribution", ylabel="Accuracy")


    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("quadratic_residues_qml_results.png", dpi=600, bbox_inches="tight")
    plt.show()


    print("[MAIN] Saving CSVs...")
    df_hist = pd.DataFrame({
        "Epoch":                         np.tile(epochs, N_REPEATS),
        "Current SotA Loss":             loss_b.flatten(),
        "Proposed Architecture Loss":     loss_n.flatten(),
        "Current SotA Gradient Norm":    grad_b.flatten(),
        "Proposed Architecture Gradient Norm": grad_n.flatten(),
    })
    df_hist.to_csv("quadratic_residues_loss_grad_history.csv", index=False)


    df_acc = pd.DataFrame({
        "Repeat":                         np.arange(N_REPEATS),
        "Current SotA Accuracy":         acc_b_all,
        "Proposed Architecture Accuracy": acc_n_all,
    })
    df_acc.to_csv("quadratic_residues_test_accuracy.csv", index=False)
    print("[MAIN] Done.")
