#%%
# --------------------------------------------------
# Imports and Seeding
# --------------------------------------------------
import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# --------------------------------------------------
# Device Setup
# --------------------------------------------------
# We'll use 2 wires: one for data and one auxiliary.
n_wires = 2  
dev_baseline = qml.device("default.qubit", wires=n_wires)
dev_new = qml.device("default.qubit", wires=n_wires)

# --------------------------------------------------
# Define the Known Quantum Channel (Target)
# --------------------------------------------------
# For this example, the channel applies a fixed RY rotation (π/4) on the data qubit.
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
    qml.RY(theta_channel, wires=0)
    return qml.state()

def target_state(angle):
    """Helper function to obtain the target state for a given input angle."""
    return target_state_circuit(angle)

# --------------------------------------------------
# Baseline Architecture Circuit
# --------------------------------------------------
@qml.qnode(dev_baseline, interface="autograd")
def baseline_channel_circuit(angle, weights):
    """
    Baseline circuit: Both qubits encode the input.
    We encode the input angle on each qubit using RY rotations.
    Then L layers of parameterized single-qubit rotations and a CNOT entangling gate are applied.
    """
    # Encoding: apply RY(angle) on both qubits.
    for wire in range(n_wires):
        qml.RY(angle, wires=wire)
    
    # Variational layers.
    L_layers = weights.shape[0]
    for layer in range(L_layers):
        for wire in range(n_wires):
            qml.RZ(weights[layer, wire, 0], wires=wire)
            qml.RY(weights[layer, wire, 1], wires=wire)
            qml.RX(weights[layer, wire, 2], wires=wire)
        qml.CNOT(wires=[0, 1])
    return qml.state()

# --------------------------------------------------
# New Architecture Circuit
# --------------------------------------------------
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
    
    # Variational layers.
    L_layers = weights.shape[0]
    for layer in range(L_layers):
        for wire in range(n_wires):
            qml.RZ(weights[layer, wire, 0], wires=wire)
            qml.RY(weights[layer, wire, 1], wires=wire)
            qml.RX(weights[layer, wire, 2], wires=wire)
        qml.CNOT(wires=[0, 1])
    return qml.state()

# --------------------------------------------------
# Differentiable Fidelity Function
# --------------------------------------------------
def fidelity(state1, state2):
    """
    Computes fidelity between two pure states without using np.vdot.
    Fidelity = |<state1|state2>|^2 = (Re(inner))^2 + (Im(inner))^2,
    where inner = sum(conj(state1) * state2)
    """
    inner = qml.math.sum(qml.math.conj(state1) * state2)
    return qml.math.square(qml.math.real(inner)) + qml.math.square(qml.math.imag(inner))

# --------------------------------------------------
# Cost Functions for Both Architectures
# --------------------------------------------------
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

# --------------------------------------------------
# Generate Training and Test Data
# --------------------------------------------------
# Here, the "data" are just input angles used to encode quantum states.
N_train = 20
train_angles = np.random.uniform(0, np.pi, N_train)

N_test = 10
test_angles = np.random.uniform(0, np.pi, N_test)

# --------------------------------------------------
# Initialize Weights and Optimizers
# --------------------------------------------------
L_layers = 2  # Number of variational layers
# weights shape: (L_layers, n_wires, 3) for three rotation angles per qubit.
weights_baseline = np.random.uniform(0, 2*np.pi, size=(L_layers, n_wires, 3), requires_grad=True)
weights_new = np.random.uniform(0, 2*np.pi, size=(L_layers, n_wires, 3), requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.01)

# --------------------------------------------------
# Training Loop for Baseline Architecture
# --------------------------------------------------
num_epochs = 200
loss_history_baseline = []
grad_norm_history_baseline = []

print("Training baseline architecture (all qubits encode data)")
for epoch in range(num_epochs):
    weights_baseline, loss_val = opt.step_and_cost(lambda w: cost_function_baseline(w, train_angles), weights_baseline)
    loss_history_baseline.append(loss_val)
    
    # Compute gradient magnitude.
    grad = qml.grad(lambda w: cost_function_baseline(w, train_angles))(weights_baseline)
    grad_norm = np.linalg.norm(grad)
    grad_norm_history_baseline.append(grad_norm)
    print(f"Epoch {epoch+1:03d}: Loss = {loss_val:.6f}, Grad Norm = {grad_norm:.6f}", end="\r")

# --------------------------------------------------
# Training Loop for New Architecture
# --------------------------------------------------
loss_history_new = []
grad_norm_history_new = []

print("\nTraining new architecture (partial encoding + auxiliary)")
for epoch in range(num_epochs):
    weights_new, loss_val = opt.step_and_cost(lambda w: cost_function_new(w, train_angles), weights_new)
    loss_history_new.append(loss_val)
    
    # Compute gradient magnitude.
    grad = qml.grad(lambda w: cost_function_new(w, train_angles))(weights_new)
    grad_norm = np.linalg.norm(grad)
    grad_norm_history_new.append(grad_norm)
    
    print(f"Epoch {epoch+1:03d}: Loss = {loss_val:.6f}, Grad Norm = {grad_norm:.6f}", end="\r")

# --------------------------------------------------
# Evaluation on Test Set
# --------------------------------------------------
def evaluate_architecture(circuit_fn, weights, angles):
    fidelities = []
    for angle in angles:
        output_state = circuit_fn(angle, weights)
        t_state = target_state(angle)
        fidelities.append(fidelity(t_state, output_state))
    return np.mean(fidelities)

avg_test_fid_baseline = evaluate_architecture(baseline_channel_circuit, weights_baseline, test_angles)
avg_test_fid_new = evaluate_architecture(new_architecture_channel_circuit, weights_new, test_angles)

print(f"\nBaseline Test Fidelity: {avg_test_fid_baseline:.4f}")
print(f"New Architecture Test Fidelity: {avg_test_fid_new:.4f}")

# --------------------------------------------------
# Plot Training Loss and Gradient Norms
# --------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history_baseline, label="Baseline")
plt.plot(loss_history_new, label="New Architecture")
plt.xlabel("Epoch")
plt.ylabel("Cost (1 - Fidelity)")
plt.title("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(grad_norm_history_baseline, label="Baseline")
plt.plot(grad_norm_history_new, label="New Architecture")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Magnitude")
plt.legend()

plt.tight_layout()
plt.show()
