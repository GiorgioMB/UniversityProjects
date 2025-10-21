# Quantum Experiments: State Learning, Entanglement Classification, Arithmetic Benchmarks, and Rugged Regression
A compact, **reproducible** suite of quantum ML experiments built on **PennyLane**, with **JAX/Optax** used where appropriate for speed and clean autodiff. 
Each experiment is a single, self-contained script designed for straightforward replication and ablation. 
Results are written to **CSV** for downstream plotting and statistical analysis.

This repository currently includes **five** experiments:

1. **Quantum State Learning** (Clifford/Haar targets): `Quantum State Learning.py`  
2. **Quantum State Learning – Appendix variants**: `Quantum State Learning Appendix.py`  
3. **Entanglement Class Classification** (multi-class): `Entanglement Class Classification.py`  
4. **Quadratic Residues mod N** (arithmetic classification with JAX/Optax): `Quadratic Residues mod N.py`  
5. **Weierstrass Function Estimation** (rugged regression): `Weierstrass Function Estimation.py`


Across all experiments we benchmark two circuit families under matched training budgets: a **baseline** variational ansatz (PennyLane’s `StronglyEntanglingLayers`) and a **proposed** ansatz. Inputs are embedded via explicitly specified feature maps; in some tasks these maps are **non-isometric** (i.e., lossy in the sense that the classical feature map $x \mapsto U(x)\lvert 0\rangle$ does not preserve pairwise distances or injectivity), or employ restricted data re-uploading, which can disadvantage the proposed model when compared at equal depth/parameter count. The proposed ansatz also includes $k$ **ancillary qubits** initialized in $\lvert +\rangle^{\otimes k}$ (via Hadamards) that are entangled with the data register and not directly measured; these act as latent degrees of freedom, effectively enlarging the variational family while leaving measurement semantics unchanged.

For each run we log, at every optimization step (t): the **empirical training loss** $\mathcal{L}^{(t)}$ (task-specific: negative fidelity surrogate or cross-entropy for classification; mean-squared error for regression), and the **parameter-space gradient norms** $\lVert\nabla_{\theta}\mathcal{L}^{(t)}\rVert_2$ for both circuit families (to study optimization dynamics and plateauing). Task metrics are reported on held-out data: **state fidelity** $F(\psi_\theta,\phi_\star)=|\langle \psi_\theta \mid \phi_\star\rangle|^2$ for state learning; **accuracy** for classification; and **test MSE** for regression. 

---

## Environment & Installation

**Python:** 3.10+ recommended.

### Minimal installation (CPU)
```bash
pip install pennylane numpy scipy matplotlib pandas seaborn joblib
# For state learning (Clifford targets):
pip install qiskit
# For arithmetic task (JAX/Optax):
pip install "jax[cpu]" optax
```

**Notes**
- JAX is **only required** by `Quadratic Residues mod N.py`. All other scripts run with PennyLane’s `autograd` interface.
- Qiskit is used to generate **random Clifford** targets in state-learning experiments.

---

## Common conventions

- **Random seeds** are set (`SEED_GLOBAL = 42`) for reproducibility.
- **Training/validation splits** and **repeat counts** are controlled via top-of-file constants.
- **Logging:** CSVs are written in the working directory with clear, task-specific names (see below).
- **Two-architecture comparison:** every script logs both **Current SotA** (baseline) and **Proposed Architecture** metrics side by side.
- **Gradient norms**: recorded for both architectures at each optimization step for plateau/instability diagnostics.

---

## Files 
- `Quantum State Learning.py`: **Goal.** Fit a parametrized circuit to a target **pure state**. Targets include **Clifford** (from Qiskit) or **Haar-like** selections (see Appendix script).
- `Quantum State Learning Appendix.py`: **Goal.** Variants/ablations for state-learning experiments, including **Haar**-style targets and alternative circuit depths.
- `Entanglement Class Classification.py`: **Goal.** Supervised classification of entanglement families (e.g., Separable/GHZ/W); variational classifier with explicit data encoding.
- `Quadratic Residues mod N.py`: **Goal.** Binary classification: decide whether integers are **quadratic residues modulo N** using variational circuits. Also includes a Weierstrass helper for data generation experiments inside this script.
- `Weierstrass Function Estimation.py`: **Goal.** Regress a multivariate **Weierstrass-type** function—useful to probe **optimization stability** and **barren plateaus** (non-smooth, multi-scale target).

