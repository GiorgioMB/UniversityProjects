# --------------------------------------------------
# Imports and Seeding
# --------------------------------------------------
import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
sns.set_theme(style="whitegrid", context="paper")
np.random.seed(42)  # global seed to have reproducible randomness



# --------------------------------------------------
# Device Setup
# --------------------------------------------------
n_wires = 2  
dev_baseline = qml.device("default.qubit", wires=n_wires)
dev_new = qml.device("default.qubit", wires=n_wires)
dev_target = qml.device("default.qubit", wires=n_wires)



# --------------------------------------------------
# Known Quantum Channel (Target) Definition
# --------------------------------------------------
theta_channel = np.pi / 4

def random_haar_unitary(n):
    """
    Generate an n × n unitary matrix sampled from the Haar measure.

    This function uses the standard method of generating a Haar-random unitary:
    it constructs a complex matrix with i.i.d. standard normal entries (real and imaginary parts),
    performs a QR decomposition, and adjusts the phases to ensure uniformity
    with respect to the Haar measure on U(n).

    Parameters
    ----------
    n : int
        Dimension of the unitary matrix to generate.

    Returns
    -------
    ndarray
        An n × n unitary matrix sampled from the Haar measure.
    """
    z = (np.random.normal(size=(n, n)) + 1j * np.random.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    q = q * ph
    return q
Haar_U = random_haar_unitary(4)  # Haar-random unitary for 2-qubit circuit


@qml.qnode(dev_target, interface="autograd")
def target_state_circuit(angle):
    """
    Construct a 2-qubit quantum circuit preparing a Haar-perturbed RY-rotated state.

    The circuit starts from the initial state |00⟩. Each qubit undergoes an RY rotation
    with the same angle, followed by the application of a fixed Haar-random 2-qubit unitary
    U ∈ U(4). The resulting state is:

        .. math::

            |\psi(\theta)\rangle = U \cdot \left( RY(\theta) \otimes RY(\theta) \right) |00\rangle

    where U is sampled once via `random_haar_unitary(4)` and fixed across calls.

    Parameters
    ----------
    angle : float
        Rotation angle θ for both RY gates.

    Returns
    -------
    ndarray
        The final statevector as a NumPy array of shape (4,).
    """
    qml.RY(angle, wires=0)
    qml.RY(angle, wires=1)
    qml.QubitUnitary(Haar_U, wires=[0, 1])
    return qml.state()


def target_state(angle):
    """Helper function to obtain the target state for a given input angle."""
    return target_state_circuit(angle)



# --------------------------------------------------
# Architecture Definitions
# --------------------------------------------------
@qml.qnode(dev_baseline, interface="autograd")
def baseline_channel_circuit(angle, weights):
    """
    Prepare a 2-qubit quantum state using the baseline encoding architecture.

    This circuit encodes the input parameter `angle` into both qubits using RY rotations. 
    After data encoding, it applies a series of L entangling layers using the 
    `StronglyEntanglingLayers` template, which alternates parameterized single-qubit 
    rotations with CNOT gates.

    Mathematically, the prepared state is:

        .. math::

            |\psi_{\text{base}}(\theta, W)\rangle = \mathcal{U}_{\text{ent}}(W) \cdot 
            \left( RY(\theta) \otimes RY(\theta) \right) |00\rangle

    where:
    - :math:`\theta` is the input angle,
    - :math:`W` denotes the parameters for the entangling layers,
    - :math:`\mathcal{U}_{\text{ent}}(W)` represents the L-layer strongly entangling ansatz.

    Parameters
    ----------
    angle : float
        Input angle θ to be encoded via RY rotations on both qubits.
    weights : array[float]
        Trainable parameters for `StronglyEntanglingLayers`, shaped according to 
        (L, n_wires, 3).

    Returns
    -------
    ndarray
        The final statevector as a NumPy array of shape (4,).
    """
    for wire in range(n_wires):
        qml.RY(angle, wires=wire)    
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.state()


@qml.qnode(dev_new, interface="autograd")
def new_architecture_channel_circuit(angle, weights):
    """
    Prepare a 2-qubit quantum state using the new asymmetric encoding architecture.

    This circuit encodes the input parameter `angle` only on qubit 0 (the data qubit)
    using an RY rotation. Qubit 1 (the auxiliary qubit) is initialized in the |+⟩ state
    using a Hadamard gate. The encoded state then undergoes L layers of parameterized
    operations via the `StronglyEntanglingLayers` template.

    Mathematically, the state prepared is:

        .. math::

            |\psi_{\text{new}}(\theta, W)\rangle = \mathcal{U}_{\text{ent}}(W) \cdot 
            \left( RY(\theta) \otimes H \right) |00\rangle

    where:
    - :math:`RY(\theta)` encodes the data on qubit 0,
    - :math:`H` prepares qubit 1 in the |+⟩ state,
    - :math:`\mathcal{U}_{\text{ent}}(W)` is the L-layer strongly entangling template.

    This architecture biases the expressivity toward asymmetric roles of data and
    auxiliary qubits, potentially improving generalization.

    Parameters
    ----------
    angle : float
        Input angle θ to be encoded via RY on the data qubit (qubit 0).
    weights : array[float]
        Trainable parameters for `StronglyEntanglingLayers`, shaped according to 
        (L, n_wires, 3).

    Returns
    -------
    ndarray
        The final statevector as a NumPy array of shape (4,).
    """
    qml.RY(angle, wires=0)
    qml.Hadamard(wires=1)
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.state()



# --------------------------------------------------
# Cost Functions and Test Function Definitions
# --------------------------------------------------
def fidelity(state1, state2):
    """
    Compute the fidelity between two pure quantum states.

    Fidelity is defined as:

        .. math::

            F(|\psi\rangle, |\phi\rangle) = |\langle \psi | \phi \rangle|^2

    For pure states represented as complex vectors, this is computed as:

        .. math::

            F = \left( \mathrm{Re}(\langle \psi | \phi \rangle) \right)^2 + 
                \left( \mathrm{Im}(\langle \psi | \phi \rangle) \right)^2

    This implementation assumes that both `state1` and `state2` are normalized vectors
    representing pure quantum states in the same Hilbert space.

    Parameters
    ----------
    state1 : array[complex]
        Complex statevector of the first pure quantum state.
    state2 : array[complex]
        Complex statevector of the second pure quantum state.

    Returns
    -------
    float
        Fidelity between the two states, in [0, 1].
    """
    inner = qml.math.sum(qml.math.conj(state1) * state2)
    return qml.math.square(qml.math.real(inner)) + qml.math.square(qml.math.imag(inner))


def cost_function(circuit_fn, weights, angles):
    """
    Compute the average fidelity-based loss over a set of training angles.

    For each input angle, this function compares the output of a quantum circuit
    to a precomputed target state using the fidelity. The cost is defined as:

        .. math::

            \text{Loss}(\theta, W) = 1 - \frac{1}{N} \sum_{i=1}^N 
                F\left( |\psi_{\text{target}}(\theta_i)\rangle, 
                        |\psi_{\text{circuit}}(\theta_i, W)\rangle \right)

    where:
    - :math:`\theta_i` are the input angles,
    - :math:`W` are the variational parameters,
    - :math:`F` is the fidelity function.

    Parameters
    ----------
    circuit_fn : callable
        A quantum function (QNode) taking `(angle, weights)` and returning a statevector.
    weights : array[float]
        Trainable parameters used inside the quantum circuit.
    angles : list[float]
        Input data points (e.g., encoded via RY gates).

    Returns
    -------
    float
        The mean fidelity loss (1 - fidelity) across all training angles.
    """
    global train_states
    total_todo = len(angles)
    loss = 0
    currently_done = 0
    for angle in angles:
        output_state = circuit_fn(angle, weights)
        t_state = train_states[angle]
        loss += 1 - fidelity(t_state, output_state)
        currently_done += 1
        print(f"Progress ({(currently_done/total_todo)*100:.2f}%)", end="\r")
    return loss / len(angles)


def evaluate_architecture(circuit_fn, weights, angles):
    """
    Evaluate the average fidelity of a quantum circuit on a test dataset.

    For each angle in the input list, the function computes the fidelity between
    the circuit's output and the precomputed target state. This provides a direct
    performance metric for comparing different circuit architectures.

    Parameters
    ----------
    circuit_fn : callable
        A quantum function (QNode) taking `(angle, weights)` and returning a statevector.
    weights : array[float]
        Trained parameters used in the circuit.
    angles : list[float]
        List of test angles used to evaluate generalization.

    Returns
    -------
    float
        Mean fidelity between the circuit outputs and the target test states.
    """
    global test_states
    fidelities = []
    currently_done = 0
    total_todo = len(angles)
    for angle in angles:
        output_state = circuit_fn(angle, weights)
        t_state = test_states[angle]
        fidelities.append(fidelity(t_state, output_state))
        currently_done += 1
        print(f"Progress ({(currently_done/total_todo)*100:.2f}%)", end="\r")
    return np.mean(fidelities)



# --------------------------------------------------
# Initialize Training Parameters and Data
# --------------------------------------------------
N_layers = 4
num_epochs = 200
opt = qml.AdamOptimizer(stepsize=0.001)
N_repeats = 100
N_train = 1000
N_test = 100


# Containers to store per-experiment data:
baseline_loss_all = []
baseline_grad_all = []
new_loss_all = []
new_grad_all = []
test_fidelity_baseline = []
test_fidelity_new = []


# Generate random training and test samples
print("Beginning training and test states generation...")
train_angles = np.random.uniform(0, np.pi, N_train)
test_angles = np.random.uniform(0, np.pi, N_test)
assert np.unique(train_angles).size == N_train, "Training angles are not unique!"
assert np.unique(test_angles).size == N_test, "Test angles are not unique!"
train_states = {train_angle: target_state(train_angle) for train_angle in train_angles}
test_states = {test_angle: target_state(test_angle) for test_angle in test_angles}
print("Training and test states generated.")



# --------------------------------------------------
# Training Loop
# --------------------------------------------------
for repeat in range(N_repeats):
    print(f"Repeat {repeat + 1}/{N_repeats}...")
    weights_baseline = np.random.uniform(0, 2*np.pi, size=(N_layers, n_wires, 3), requires_grad=True)
    weights_new = deepcopy(weights_baseline)  # Initialize new architecture weights to the same as baseline


    # Lists to store training history for each repeat
    loss_history_baseline = []
    grad_norm_history_baseline = []
    loss_history_new = []
    grad_norm_history_new = []
    

    # Training loop for both architectures
    for epoch in range(num_epochs):
        # --- Baseline architecture ---
        loss_val_baseline = cost_function(baseline_channel_circuit, weights_baseline, train_angles)
        grad_baseline = qml.grad(lambda w: cost_function(baseline_channel_circuit, w, train_angles))(weights_baseline)
        weights_baseline = opt.apply_grad(grad_baseline, weights_baseline)
        weights_baseline = np.array(weights_baseline, requires_grad=True)
        loss_history_baseline.append(loss_val_baseline)
        grad_norm_history_baseline.append(np.linalg.norm(grad_baseline))

        
        # --- New architecture ---
        loss_val_new = cost_function(new_architecture_channel_circuit, weights_new, train_angles)
        grad_new = qml.grad(lambda w: cost_function(new_architecture_channel_circuit, w, train_angles))(weights_new)
        weights_new = opt.apply_grad(grad_new, weights_new)
        weights_new = np.array(weights_new, requires_grad=True)
        loss_history_new.append(loss_val_new)
        grad_norm_history_new.append(np.linalg.norm(grad_new))

        
        print({' '*13},f"Epoch {epoch+1:03d}: Baseline Architecture: Loss = {loss_val_baseline:.4f}, Grad Norm = {grad_norm_history_baseline[-1]:.4f} New Architecture: Loss = {loss_val_new:.4f}, Grad Norm = {grad_norm_history_new[-1]:.4f}", end="\r")
        del loss_val_baseline, grad_baseline, loss_val_new, grad_new

    
    fid_baseline = evaluate_architecture(baseline_channel_circuit, weights_baseline, test_angles)
    fid_new = evaluate_architecture(new_architecture_channel_circuit, weights_new, test_angles)
    print(f"\nTest Fidelity Baseline: {fid_baseline:.4f}, Test Fidelity New Architecture: {fid_new:.4f}")
    baseline_loss_all.append(loss_history_baseline)
    baseline_grad_all.append(grad_norm_history_baseline)
    new_loss_all.append(loss_history_new)
    new_grad_all.append(grad_norm_history_new)
    test_fidelity_baseline.append(fid_baseline)
    test_fidelity_new.append(fid_new)
    print(f"Repeat {repeat + 1}/{N_repeats} completed.\n")


baseline_loss_all = np.array(baseline_loss_all)
baseline_grad_all = np.array(baseline_grad_all)
new_loss_all = np.array(new_loss_all)
new_grad_all = np.array(new_grad_all)



# --------------------------------------------------
# Aggregate Training Statistics (per epoch)
# --------------------------------------------------
baseline_loss_min = baseline_loss_all.min(axis=0)
baseline_loss_mean = baseline_loss_all.mean(axis=0)
baseline_loss_max = baseline_loss_all.max(axis=0)


baseline_grad_min = baseline_grad_all.min(axis=0)
baseline_grad_mean = baseline_grad_all.mean(axis=0)
baseline_grad_max = baseline_grad_all.max(axis=0)


new_loss_min = new_loss_all.min(axis=0)
new_loss_mean = new_loss_all.mean(axis=0)
new_loss_max = new_loss_all.max(axis=0)


new_grad_min = new_grad_all.min(axis=0)
new_grad_mean = new_grad_all.mean(axis=0)
new_grad_max = new_grad_all.max(axis=0)


epochs = range(1, num_epochs + 1)



# --------------------------------------------------
# Plot Training Loss for Both Architectures
# --------------------------------------------------
plt.figure(figsize=(14, 6))


# Loss Plot
plt.subplot(1, 2, 1)
sns.lineplot(x=epochs, y=baseline_loss_mean, label="Baseline Mean Loss", color="blue")
plt.fill_between(epochs, baseline_loss_min, baseline_loss_max, color="blue", alpha=0.2)
sns.lineplot(x=epochs, y=new_loss_mean, label="New Arch. Mean Loss", color="orange")
plt.fill_between(epochs, new_loss_min, new_loss_max, color="orange", alpha=0.2)
plt.xlabel("Epoch")
plt.ylabel("Cost (1 - Fidelity)")
plt.title("Training Loss Over Epochs (Aggregated over 100 Repetitions)")
plt.legend()


# Gradient Norm Plot
plt.subplot(1, 2, 2)
sns.lineplot(x=epochs, y=baseline_grad_mean, label="Baseline Mean Grad Norm", color="blue")
plt.fill_between(epochs, baseline_grad_min, baseline_grad_max, color="blue", alpha=0.2)
sns.lineplot(x=epochs, y=new_grad_mean, label="New Arch. Mean Grad Norm", color="orange")
plt.fill_between(epochs, new_grad_min, new_grad_max, color="orange", alpha=0.2)
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm Over Epochs (Aggregated over 100 Repetitions)")
plt.legend()


plt.tight_layout()
plt.show()



# --------------------------------------------------
# Plot Box Plot of Test Fidelities
# --------------------------------------------------
plt.figure(figsize=(6, 6))
sns.boxplot(data=[test_fidelity_baseline, test_fidelity_new], palette=["blue", "orange"])
plt.xticks([0, 1], ["Baseline", "New Architecture"])
plt.ylabel("Test Fidelity")
plt.title("Box Plot of Test Fidelities Across 100 Repetitions")
plt.show()
