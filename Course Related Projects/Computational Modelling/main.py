"""
Evolving a small VQC in PennyLane (JAX interface) for the digits 0-vs-1 toy task,
with optional QAOA layers and automatic visualisation.
"""
from __future__ import annotations

import argparse, json, math, random, shutil, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import qaoa  
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Global hyper‑parameters (override via CLI)
# ──────────────────────────────────────────────────────────────────────────────
HP = dict(
    n_qubits=6,
    population=50,
    generations=50,
    tournament=3,
    max_init_depth=8,
    p_cx=0.7,
    p_mut=0.3,
    optimiser_steps=30,           # inner param optimisation (gradient-free)
    optimiser_sigma=0.15,
    complexity_penalty=1e-2,
    seed=42,
    shots=None,                   # None -> analytic mode
    use_qaoa=False,               # allow genomes to insert 1-layer QAOA blocks
    results_dir="results",
)

rng = np.random.default_rng(HP["seed"])
random.seed(HP["seed"])

GATE_SET = ("RX", "RY", "RZ", "CNOT")  # base gate library


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Data utilities 
# ──────────────────────────────────────────────────────────────────────────────
def load_dataset():
    """Returns train/test splits with L2-normalised feature vectors (length 64)."""
    X, y = load_digits(n_class=2, return_X_y=True)
    X = MinMaxScaler().fit_transform(X)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return train_test_split(
        X.astype("float32"), y.astype("int8"), test_size=0.2, random_state=HP["seed"]
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Genome / individual abstractions
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Gene:
    gate: str
    targets: Tuple[int, ...]
    theta: float | None = None      # Only for parametrised single‑qubit rotations

    # QAOA convenience ctor ----------------------------------------------------
    @staticmethod
    def qaoa_layer():
        # A single p=1 layer: free parameters gamma, beta
        return Gene("QAOA", tuple(), None)


Genome = List[Gene]


@dataclass
class Individual:
    genome: Genome
    fitness: float | None = None
    best_thetas: Dict[int, float] = field(default_factory=dict)  # idx->theta

    # -------------------------------------------------------------------------
    def depth(self) -> int:
        """Number of primitive gates (QAOA counts as 2*n_qubits for penalty)."""
        d = 0
        for g in self.genome:
            if g.gate == "QAOA":
                d += 2 * HP["n_qubits"]
            else:
                d += 1
        return d

    # -------------------------------------------------------------------------
    def copy(self):
        return Individual([Gene(**vars(g)) for g in self.genome])

    # ────────────────────────────────────────────────────────────────────
    #   Circuit factory
    # ────────────────────────────────────────────────────────────────────
    def make_qnode(self, dev, Xvec, theta_overrides: Dict[int, float] | None = None):
        thetas = {
            idx: theta_overrides.get(idx, g.theta)
            for idx, g in enumerate(self.genome)
            if g.theta is not None
        }

        @qml.qnode(dev, interface="jax", diff_method="parameter-shift")
        def circuit():
            qml.templates.AmplitudeEmbedding(Xvec, wires=range(HP["n_qubits"]), normalize=False)

            for idx, g in enumerate(self.genome):
                if g.gate in {"RX", "RY", "RZ"}:
                    angle = thetas[idx]
                    getattr(qml, g.gate)(angle, wires=g.targets[0])
                elif g.gate == "CNOT":
                    qml.CNOT(wires=g.targets)
                elif g.gate == "QAOA":
                    # Simple MaxCut on a ring as a generic entangling pattern
                    graph = [(i, (i + 1) % HP["n_qubits"]) for i in range(HP["n_qubits"])]
                    gamma, beta = thetas[idx], thetas[idx + 1]  # assume stored consecutively
                    qaoa.max_cut(graph, gamma, beta)
                else:
                    raise ValueError(f"Unknown gate {g.gate}")

            return qml.expval(qml.PauliZ(0))

        return circuit

    # ────────────────────────────────────────────────────────────────────
    #   Parameter fine‑tuning (hill climb in theta-space)
    # ────────────────────────────────────────────────────────────────────
    def tune_parameters(self, dev, X_train, y_train):
        # build theta vector in the order of appearance
        theta_vec = jnp.array(
            [g.theta for g in self.genome if g.theta is not None], dtype=jnp.float32
        )

        # helper to map back to dict -------------------------------------------------
        def vec_to_dict(vec):
            out, k = {}, 0
            for idx, g in enumerate(self.genome):
                if g.theta is not None:
                    out[idx] = vec[k]
                    k += 1
            return out

        # negative log‑likelihood + lambda * depth
        @jax.jit
        def loss_fn(vec):
            theta_map = vec_to_dict(vec)
            nll = 0.0
            for x, y in zip(X_train, y_train):
                zexp = self.make_qnode(dev, x, theta_map)()
                # Probabilities
                p = 0.5 * (1 + zexp) if y == 0 else 0.5 * (1 - zexp)
                nll = nll - jnp.log(jnp.clip(p, 1e-9, 1.0))
            nll = nll / len(X_train)
            return nll + HP["complexity_penalty"] * self.depth()

        best_vec = theta_vec
        best_loss = float(loss_fn(best_vec))

        for _ in range(HP["optimiser_steps"]):
            cand = best_vec + HP["optimiser_sigma"] * jax.random.normal(
                jax.random.PRNGKey(rng.integers(0, 2**32)), best_vec.shape
            )
            cand_loss = float(loss_fn(cand))
            if cand_loss < best_loss:
                best_loss, best_vec = cand_loss, cand

        # persist
        self.best_thetas = {k: float(v) for k, v in vec_to_dict(best_vec).items()}
        self.fitness     = best_loss


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Evolutionary operators 
# ──────────────────────────────────────────────────────────────────────────────
def random_gene() -> Gene:
    if HP["use_qaoa"] and rng.random() < 0.1:
        gamma = rng.uniform(0, 2 * math.pi)
        beta = rng.uniform(0, math.pi)
        return Gene("QAOA", (), gamma)  # beta will be appended as a separate Gene
    g = random.choice(GATE_SET)
    if g == "CNOT":
        c, t = rng.choice(HP["n_qubits"], size=2, replace=False)
        return Gene("CNOT", (int(c), int(t)))
    else:
        q = random.randrange(HP["n_qubits"])
        return Gene(g, (q,), rng.uniform(0, 2 * math.pi))


def seed_genome():
    genome: Genome = [random_gene() for _ in range(rng.integers(1, HP["max_init_depth"]))]
    # expand QAOA placeholders into two-gene blocks (gamma, beta)
    expanded = []
    for g in genome:
        if g.gate == "QAOA":
            expanded.extend([g, Gene("QAOA", (), rng.uniform(0, math.pi))])  # beta gene
        else:
            expanded.append(g)
    return expanded


def crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    if rng.random() > HP["p_cx"] or len(p1.genome) < 2 or len(p2.genome) < 2:
        return p1.copy(), p2.copy()
    cx1, cx2 = rng.integers(1, len(p1.genome)), rng.integers(1, len(p2.genome))
    return (
        Individual(p1.genome[:cx1] + p2.genome[cx2:]),
        Individual(p2.genome[:cx2] + p1.genome[cx1:]),
    )


def mutate(ind: Individual):
    if rng.random() > HP["p_mut"]:
        return
    op = random.choice(("insert", "delete", "swap", "point"))
    g = ind.genome
    if op == "insert" or not g:
        pos = rng.integers(0, len(g) + 1)
        ind.genome.insert(pos, random_gene())
    elif op == "delete" and len(g) > 1:
        ind.genome.pop(rng.integers(0, len(g)))
    elif op == "swap" and len(g) > 1:
        i, j = rng.choice(len(g), size=2, replace=False)
        g[i], g[j] = g[j], g[i]
    else:  # point mutation
        target = random.choice(g)
        if target.gate == "CNOT":
            c, t = rng.choice(HP["n_qubits"], size=2, replace=False)
            target.targets = (int(c), int(t))
        elif target.gate in {"RX", "RY", "RZ"}:
            if rng.random() < 0.5:
                target.theta = (target.theta + rng.normal() * 0.3) % (2 * math.pi)
            else:
                target.targets = (random.randrange(HP["n_qubits"]),)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  EA main loop 
# ──────────────────────────────────────────────────────────────────────────────
def evolutionary_search(dev, X_train, y_train, **HP):
    pop = [Individual(seed_genome()) for _ in range(HP["population"])]
    history = {
        "gen": [],
        "best_loss": [],
        "best_depth": [],
        "best_train_acc": [],
    }

    def acc(ind, X, y):
        correct = 0
        for x, label in zip(X, y):
            zexp = ind.make_qnode(dev, x, ind.best_thetas)()
            p0 = 0.5 * (1 + zexp)
            pred = 0 if p0 >= 0.5 else 1
            correct += pred == label
        return correct / len(X)

    for gen in range(HP["generations"]):
        print(f"\nGeneration {gen + 1}/{HP['generations']}:")
        # (i) evaluate / fine-tune
        for ind in pop:
            if ind.fitness is None:
                ind.tune_parameters(dev, X_train, y_train)

        pop.sort(key=lambda ind: ind.fitness)
        best = pop[0]
        train_acc = acc(best, X_train, y_train)

        print(
            f"Gen {gen+1:02d} ┃ loss={best.fitness:.4f}  depth={best.depth():2d}  "
            f"train-acc={train_acc*100:5.1f}%"
        )

        history["gen"].append(gen)
        history["best_loss"].append(best.fitness)
        history["best_depth"].append(best.depth())
        history["best_train_acc"].append(train_acc)

        if best.fitness < 0.01:  # early stop
            break

        # (ii) selection → tournament
        parents = []
        while len(parents) < HP["population"]:
            contenders = rng.choice(pop, HP["tournament"], replace=False)
            parents.append(min(contenders, key=lambda ind: ind.fitness))

        # (iii) reproduction
        offspring = []
        for p1, p2 in zip(parents[::2], parents[1::2]):
            c1, c2 = crossover(p1, p2)
            mutate(c1)
            mutate(c2)
            offspring.extend([c1, c2])
        if gen != HP["generations"] - 1:  # last gen is not replaced
            pop = offspring
            ##keep the best individual from the previous generation
            pop.append(best.copy())

    pop.sort(key=lambda ind: ind.fitness)
    return pop[0], history


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────
def plot_history(hist, save_dir: Path):
    plt.figure()
    plt.plot(hist["gen"], hist["best_loss"])
    plt.xlabel("Generation")
    plt.ylabel("Best loss")
    plt.title("Evolution progress")
    plt.tight_layout()
    plt.savefig(save_dir / "loss_vs_generation.pdf")

    plt.figure()
    plt.plot(hist["gen"], np.array(hist["best_train_acc"]) * 100)
    plt.xlabel("Generation")
    plt.ylabel("Best train accuracy (%)")
    plt.tight_layout()
    plt.savefig(save_dir / "train_acc_vs_generation.pdf")

    plt.figure()
    plt.plot(hist["gen"], hist["best_depth"])
    plt.xlabel("Generation")
    plt.ylabel("Depth (gate count)")
    plt.tight_layout()
    plt.savefig(save_dir / "depth_vs_generation.pdf")


def plot_gate_histogram(champion: Individual, save_dir: Path):
    from collections import Counter

    counts = Counter(g.gate for g in champion.genome)
    labels, freqs = zip(*sorted(counts.items()))
    plt.figure()
    plt.bar(labels, freqs)
    plt.xlabel("Gate type")
    plt.ylabel("Frequency in champion genome")
    plt.tight_layout()
    plt.savefig(save_dir / "gate_histogram.pdf")


def save_circuit_diagram(champion: Individual, save_dir: Path):
    dev_tmp = qml.device("default.qubit", wires=HP["n_qubits"])
    dummy_x = jnp.ones(2**HP["n_qubits"]) / jnp.sqrt(2**HP["n_qubits"])
    fig, _ = qml.draw_mpl(champion.make_qnode(dev_tmp, dummy_x, champion.best_thetas))()
    fig.savefig(save_dir / "champion_circuit.pdf", bbox_inches="tight")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  CLI entry point 
# ──────────────────────────────────────────────────────────────────────────────
def main(**overrides):
    """
    Run an evolutionary search for a variational circuit and log results.

    Any kwarg in *overrides* shadows the default hyper-parameters in HP.
    """

    # ------------------------------------------------------------------ #
    # 1. Merge the user-supplied overrides into the module-level HP dict #
    # ------------------------------------------------------------------ #
    args = {**HP, **overrides}                     # merged view
    for k in ("generations", "population", "shots"):
        if k in args and args[k] is not None: 
            HP[k] = args[k]

    if args.get("use_qaoa"):
        HP["use_qaoa"] = True                      # respect CLI flag

    # ----------------------- #
    # 2. Results directory    #
    # ----------------------- #
    out_dir = Path(HP["results_dir"])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)      # make intermediate dirs too

    # ----------------------- #
    # 3. Data and quantum dev #
    # ----------------------- #
    X_train, X_test, y_train, y_test = load_dataset()
    dev = qml.device(
        "default.qubit",
        wires=HP["n_qubits"],
        shots=HP["shots"]    
    )

    # ----------------------- #
    # 4. Evolutionary search  #
    # ----------------------- #
    t0 = time.time()
    champion, history = evolutionary_search(dev, X_train, y_train, **HP)
    runtime = time.time() - t0

    # ----------------------- #
    # 5. Evaluation           #
    # ----------------------- #
    def accuracy(ind, X, y):
        correct = 0
        for x, label in zip(X, y):
            zexp = ind.make_qnode(dev, x, ind.best_thetas)()
            p0 = 0.5 * (1 + zexp)
            pred = 0 if p0 >= 0.5 else 1
            correct += int(pred == label)
        return correct / len(X)

    test_acc = accuracy(champion, X_test, y_test)
    print(f"\nChampion test accuracy: {test_acc*100:.2f}% -- depth={champion.depth()}")

    # ----------------------- #
    # 6. Visualisation        #
    # ----------------------- #
    plot_history(history, out_dir)
    plot_gate_histogram(champion, out_dir)
    save_circuit_diagram(champion, out_dir)

    # ----------------------- #
    # 7. Metadata dump        #
    # ----------------------- #
    summary = {
        "HP": HP,
        "runtime_sec": runtime,
        "test_accuracy": test_acc,
        "depth": champion.depth(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {out_dir.absolute()}")

if __name__ == "__main__":
    # Toggle this to True if you want to parse CLI occasionally
    USE_CLI = False

    if USE_CLI:
        import argparse
        parser = argparse.ArgumentParser("Evolutionary VQC (PennyLane-JAX)")
        for k, v in HP.items():
            arg_type = type(v) if v is not None else str
            flag = f"--{k.replace('_', '-')}"
            parser.add_argument(flag, type=arg_type, default=v)
        args = vars(parser.parse_args())
        main(**args)              # args is already a dict
    else:
        main()                    # pure-python usage
