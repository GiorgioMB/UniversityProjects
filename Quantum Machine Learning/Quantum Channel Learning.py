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



# --------------------------------------------------
# Known Quantum Channel (Target) Definition
# --------------------------------------------------
theta_channel = np.pi / 4


@qml.qnode(dev_new, interface="autograd")
def target_state_circuit(angle):
    """
    Prepares the target state for a given input angle.
    Encoding: RY(angle) on data qubit and Hadamard on auxiliary,
    followed by the known channel: RY(theta_channel) on the data qubit.
    """
    qml.RY(angle, wires=0)
    qml.Hadamard(wires=1)
    qml.RY(angle, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 0])
    qml.RX(-theta_channel, wires=1)
    qml.RZ(-angle, wires=1)
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
    Baseline circuit: Both qubits encode the input.
    We encode the input angle on each qubit using RY rotations.
    Then L layers of parameterized single-qubit rotations and a CNOT entangling gate are applied.
    """
    for wire in range(n_wires):
        qml.RY(angle, wires=wire)    
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.state()


@qml.qnode(dev_new, interface="autograd")
def new_architecture_channel_circuit(angle, weights):
    """
    New architecture circuit: Only the data qubit encodes the input.
    Data qubit: encoded via RY(angle); auxiliary qubit: prepared in |+> via Hadamard.
    Then L layers of parameterized rotations on both qubits with an entangling CNOT are applied.
    """
    # Encoding step.
    qml.RY(angle, wires=0)
    qml.Hadamard(wires=1)
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.state()



# --------------------------------------------------
# Cost Functions and Test Function Definitions
# --------------------------------------------------
def fidelity(state1, state2):
    """
    Computes fidelity between two pure states.
    Fidelity = |<state1|state2>|^2 = (Re(inner))^2 + (Im(inner))^2,
    where inner = sum(conj(state1) * state2)
    """
    inner = qml.math.sum(qml.math.conj(state1) * state2)
    return qml.math.square(qml.math.real(inner)) + qml.math.square(qml.math.imag(inner))


def cost_function_baseline(weights, angles):
    """Cost for the baseline architecture: average (1 - fidelity) over training angles."""
    loss = 0
    for angle in angles:
        output_state = baseline_channel_circuit(angle, weights)
        t_state = target_state(angle)
        loss += 1 - fidelity(t_state, output_state)
    return loss / len(angles)


def cost_function_new(weights, angles):
    """Cost for the new architecture: average (1 - fidelity) over training angles."""
    loss = 0
    for angle in angles:
        output_state = new_architecture_channel_circuit(angle, weights)
        t_state = target_state(angle)
        loss += 1 - fidelity(t_state, output_state)
    return loss / len(angles)


def evaluate_architecture(circuit_fn, weights, angles):
    """Evaluate average fidelity over a list of angles."""
    fidelities = []
    for angle in angles:
        output_state = circuit_fn(angle, weights)
        t_state = target_state(angle)
        fidelities.append(fidelity(t_state, output_state))
    return np.mean(fidelities)



# --------------------------------------------------
# Initialize Training Parameters and Data
# --------------------------------------------------
N_layers = 4  
num_epochs = 200
opt = qml.AdamOptimizer(stepsize=0.005)
N_repeats = 100
N_train = 20
N_test = 10


# Containers to store per-experiment data:
baseline_loss_all = []
baseline_grad_all = []
new_loss_all = []
new_grad_all = []
test_fidelity_baseline = []
test_fidelity_new = []


# Generate random training and test angles
train_angles = np.random.uniform(0, np.pi, N_train)
test_angles = np.random.uniform(0, np.pi, N_test)



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
        weights_baseline, loss_val_baseline = opt.step_and_cost(
            lambda w: cost_function_baseline(w, train_angles), weights_baseline
        )
        loss_history_baseline.append(loss_val_baseline)
        grad_baseline = qml.grad(lambda w: cost_function_baseline(w, train_angles))(weights_baseline)
        grad_norm_baseline = np.linalg.norm(grad_baseline)
        grad_norm_history_baseline.append(grad_norm_baseline)


        weights_new, loss_val_new = opt.step_and_cost(
            lambda w: cost_function_new(w, train_angles), weights_new
        )
        loss_history_new.append(loss_val_new)
        grad_new = qml.grad(lambda w: cost_function_new(w, train_angles))(weights_new)
        grad_norm_new = np.linalg.norm(grad_new)
        grad_norm_history_new.append(grad_norm_new)
        print(f"Epoch {epoch+1:03d}: Baseline Architecture: Loss = {loss_val_baseline:.4f}, Grad Norm = {grad_norm_baseline:.4f} New Architecture: Loss = {loss_val_new:.4f}, Grad Norm = {grad_norm_new:.4f}", end="\r")

    
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
