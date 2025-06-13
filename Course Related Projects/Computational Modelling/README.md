# Evolutionary Design of Quantum Circuit Architectures for NISQ-Era ML  

> **TL;DR** An evolutionary algorithm (“circuit DNA”) automatically breeds compact, high-accuracy quantum classifiers.  
> Six-qubit toy task (MNIST 0 vs 1) $\rightarrow$ **90.3 %** test accuracy with a **4-gate** circuit.  

---

## Project Overview
This project explores **automated quantum‐circuit architecture search** in the Noisy Intermediate-Scale Quantum (NISQ) regime.  
Instead of hand-crafting an ansatz, we **evolve** a population of candidate circuits—simultaneously optimising their _structure_ (gate sequence) and _parameters_ (rotation angles)—to solve a classical machine-learning task.

The codebase accompanies the attached research manuscript:

> **“Evolutionary Design of Quantum Circuit Architectures for NISQ-Era Machine Learning”**  
> (G. Micaletto, 2025)

While the experiment targets a small 0-vs-1 digit classifier, it serves as a reusable template for larger data sets, other cost functions, or hardware back-ends.

---

## Key Contributions & Features

| Area | What’s new / improved |
|------|-----------------------|
| **Evolutionary Search Engine** | <ul><li>Variable-length genomes drawn from a minimal gate set **{ RX, RY, RZ, CNOT }** (optionally one-layer QAOA).</li><li>Rich genetic operators: insertion, deletion, swap, point mutation; single-point crossover with adaptive depth control.</li><li>**Complexity-aware fitness** = negative log-likelihood + gate-depth penalty.</li></ul> |
| **Parameter Fine-Tuning** | <ul><li>Gradient-free **Gaussian hill-climber** inside each individual for rapid local optimisation.</li><li>Seamless analytic-mode or shot-based simulation via PennyLane device abstraction.</li></ul> |
| **Reproducible Research** | Deterministic seeds (`numpy` + `random`) and full **JSON summary** of hyper-parameters + metrics. |
| **Visualisation Suite** | Instant **PDF plots** (loss, accuracy, depth), gate-usage histogram, and **circuit diagram** of the champion. |
| **CLI-ready** | One-file script with argparse toggle; every hyper-parameter overridable from the command line. |

---
## Directory Structure

```
evo-vqc-project/
├── main.py              # Main single-file implementation
├── manuscript/             # Final form of the research paper
│   └── Evolutionary_Design_of_QCA.pdf
├── results/                # Auto-generated artefacts per run
│   ├── loss_vs_generation.pdf
│   ├── train_acc_vs_generation.pdf
│   ├── depth_vs_generation.pdf
│   ├── gate_histogram.pdf
│   ├── champion_circuit.pdf
│   └── summary.json
└── README.md

```

---

## Quick Start

### 1 — Install dependencies

```bash
# Recommended: Python ≥ 3.10 in a virtualenv/conda
pip install pennylane pennylane-qiskit jax jaxlib \
            numpy matplotlib scikit-learn tqdm
```
### 2 — Run the experiment
```bash
python main.py               # uses defaults (50 × 50 generations)
```
Common overrides:
```bash
# Use shots instead of analytic mode
python main.py --shots 1024

# Speed/run-time trade-off
python main.py --population 30 --generations 30

# Allow 1-layer QAOA blocks in the search space
python main.py --use-qaoa
```
Note: the flage `USE_CLI` has to be manually set to `True` to use overrides

All outputs land in `results/` and are overwritten each run.


## Sample Results

```json
{
  "test_accuracy": 0.9028,
  "depth": 4,
  "runtime_sec": 16480,
  "n_qubits": 6,
  "population": 50,
  "generations": 50,
  "...": "see results/summary.json for full dump"
}
```

Generated champion circuit (drawn on six wires):

![champion_circuit.pdf](results/champion_circuit.pdf)

---

## Workflow

1. **Data Prep** `sklearn.load_digits` → normalise & L2 scale (0/1 classes only).
2. **Population Init** Random genomes, depth ≤ 8.
3. **Inner Loop per Generation**

   1. **Parameter Tuning** (Gaussian hill-climb).
   2. **Fitness Evaluation** (NLL + depth penalty).
   3. **Tournament Selection** (k = 3).
   4. **Crossover & Mutation** to spawn offspring.
   5. **Elitism** (keep best).
4. **Early Stop** if loss < 0.01 or after `--generations`.
5. **Visualise & Dump** champion + history.

---

## Extending the Framework

| Want to…                     | Change                                                                                                   |
| ---------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Support real hardware**    | Swap `qml.device("default.qubit", …)` for `qml.device("qiskit.ibmq", …)` (or similar) and set `--shots`. |
| **Use a different data set** | Re-implement `load_dataset()` to return `X_train, X_test, y_train, y_test`.                              |
| **Add new gates**            | Append to `GATE_SET` and extend `Individual.make_qnode()`.                                               |
| **Try deeper QAOA**          | Increase block insertion probability or encode $p>1$ layers.                                             |

