#%%
# --------------------------------------------------
# Imports and Seeding
# --------------------------------------------------
import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --------------------------------------------------
# Parameters and Data Setup
# --------------------------------------------------
# For the baseline: all 3 qubits encode the state (Hilbert space dim 8).
# For the new architecture: first 2 qubits encode the state (dim 4) and the 3rd is auxiliary.
n_wires_baseline = 3      # full device for baseline
n_data_qubits = 2         # data qubits for new architecture
n_wires_new = n_wires_baseline  # total qubits (data + auxiliary) remain 3

labels = [0, 1, 2]
num_train = 150
num_test = 50
train_labels = np.random.choice(labels, size=num_train)
test_labels  = np.random.choice(labels, size=num_test)



# --------------------------------------------------
# State Preparation Functions (Encoding)
# --------------------------------------------------
# For the baseline, we encode a state vector in 8 dimensions.
def get_amplitude_state(label):
    state = np.zeros(8)
    if label == 0:
        state[0] = 1.0  # |000>
    elif label == 1:
        # A state with support on |000> and |001>
        state[0] = 1/np.sqrt(2)
        state[1] = 1/np.sqrt(2)
    elif label == 2:
        # A state with support on |000>, |001>
        state[0] = 1/np.sqrt(3)
        state[1] = np.sqrt(2/3)
    return state

# For the new architecture, we encode only on the first two qubits (4-dim state).
def get_amplitude_state_new(label):
    state = np.zeros(4)
    if label == 0:
        state[0] = 1.0  # |00>
    elif label == 1:
        state[0] = 1/np.sqrt(2)
        state[1] = 1/np.sqrt(2)
    elif label == 2:
        state[0] = 1/np.sqrt(3)
        state[1] = np.sqrt(2/3)
    return state



# --------------------------------------------------
# Devices and Circuit Definitions
# --------------------------------------------------
dev_baseline = qml.device("default.qubit", wires=n_wires_baseline)
dev_new = qml.device("default.qubit", wires=n_wires_new)

# Note on Measurement:
# We will measure the first two wires (data qubits) to obtain a 4-element probability vector.
# Then we aggregate outcomes into 3 classes as follows:
#   - Outcome "00" (index 0) -> class 0.
#   - Outcome "01" (index 1) -> class 1.
#   - Outcomes "10" and "11" (indices 2 and 3) -> class 2.
def aggregate_probs(probs):
    p0 = probs[0]
    p1 = probs[1]
    p2 = probs[2] + probs[3]
    return np.array([p0, p1, p2])

@qml.qnode(dev_baseline, interface="autograd")
def baseline_circuit(label, weights):
    state = get_amplitude_state(label)
    qml.AmplitudeEmbedding(state, wires=range(n_wires_baseline), normalize=True)
    L = weights.shape[0]
    for layer in range(L):
        for wire in range(n_wires_baseline):
            qml.RZ(weights[layer, wire, 0], wires=wire)
            qml.RY(weights[layer, wire, 1], wires=wire)
            qml.RX(weights[layer, wire, 2], wires=wire)
        for wire in range(n_wires_baseline):
            qml.CNOT(wires=[wire, (wire+1) % n_wires_baseline])
    return qml.probs(wires=[0,1])

@qml.qnode(dev_new, interface="autograd")
def new_architecture_circuit(label, weights):
    state = get_amplitude_state_new(label)
    qml.AmplitudeEmbedding(state, wires=range(n_data_qubits), normalize=True)
    qml.Hadamard(wires=n_data_qubits)
    L = weights.shape[0]
    for layer in range(L):
        for wire in range(n_wires_new):
            qml.RZ(weights[layer, wire, 0], wires=wire)
            qml.RY(weights[layer, wire, 1], wires=wire)
            qml.RX(weights[layer, wire, 2], wires=wire)
        for wire in range(n_wires_new):
            qml.CNOT(wires=[wire, (wire+1) % n_wires_new])
    return qml.probs(wires=[0,1])



# --------------------------------------------------
# Cost Functions and Loss Definitions
# --------------------------------------------------
def cross_entropy_loss(pred, target):
    return -np.sum(target * np.log(pred + 1e-10))

def cost_baseline(weights, labels_arr):
    loss = 0
    for lbl in labels_arr:
        probs = baseline_circuit(lbl, weights)
        probs = aggregate_probs(probs)
        if lbl == 0:
            target = np.array([1, 0, 0])
        elif lbl == 1:
            target = np.array([0, 1, 0])
        elif lbl == 2:
            target = np.array([0, 0, 1])
        loss += cross_entropy_loss(probs, target)
    return loss / len(labels_arr)

def cost_new(weights, labels_arr):
    loss = 0
    for lbl in labels_arr:
        probs = new_architecture_circuit(lbl, weights)
        probs = aggregate_probs(probs)
        if lbl == 0:
            target = np.array([1, 0, 0])
        elif lbl == 1:
            target = np.array([0, 1, 0])
        elif lbl == 2:
            target = np.array([0, 0, 1])
        loss += cross_entropy_loss(probs, target)
    return loss / len(labels_arr)



# --------------------------------------------------
# Training Setup
# --------------------------------------------------
num_epochs = 200   ##Note: this will take a while, testing on my local machine I used only 10 epochs
L_layers = 2 

weights_baseline = np.random.uniform(0, 2*np.pi, size=(L_layers, n_wires_baseline, 3), requires_grad=True)
weights_new = np.random.uniform(0, 2*np.pi, size=(L_layers, n_wires_new, 3), requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.01)

loss_history_baseline = []
grad_norm_history_baseline = []

loss_history_new = []
grad_norm_history_new = []



# --------------------------------------------------
# Training Loop for Baseline Circuit (all qubits encode data)
# --------------------------------------------------
print("Training baseline circuit (full encoding)")
for epoch in range(num_epochs):
    weights_baseline, loss_val = opt.step_and_cost(lambda w: cost_baseline(w, train_labels), weights_baseline)
    loss_history_baseline.append(loss_val)
    grad = qml.grad(lambda w: cost_baseline(w, train_labels))(weights_baseline)
    grad_norm = np.linalg.norm(grad)
    grad_norm_history_baseline.append(grad_norm)
    print(f"Epoch {epoch+1:03d}: Loss = {loss_val:.4f}, Grad Norm = {grad_norm:.4f}", end="\r")



# --------------------------------------------------
# Training Loop for New Architecture Circuit (partial encoding + auxiliary)
# --------------------------------------------------
print("\nTraining new architecture circuit (partial encoding + auxiliary)")
for epoch in range(num_epochs):
    weights_new, loss_val = opt.step_and_cost(lambda w: cost_new(w, train_labels), weights_new)
    loss_history_new.append(loss_val)
    grad = qml.grad(lambda w: cost_new(w, train_labels))(weights_new)
    grad_norm = np.linalg.norm(grad)
    grad_norm_history_new.append(grad_norm)
    print(f"Epoch {epoch+1:03d}: Loss = {loss_val:.4f}, Grad Norm = {grad_norm:.4f}", end="\r")



# --------------------------------------------------
# Evaluation on Test Set
# --------------------------------------------------
def predict(circuit_fn, weights, lbl):
    probs = circuit_fn(lbl, weights)
    probs = aggregate_probs(probs)
    return np.argmax(probs)

preds_baseline = [predict(baseline_circuit, weights_baseline, lbl) for lbl in test_labels]
acc_baseline = np.mean(np.array(preds_baseline) == test_labels)

preds_new = [predict(new_architecture_circuit, weights_new, lbl) for lbl in test_labels]
acc_new = np.mean(np.array(preds_new) == test_labels)

print(f"\nBaseline Test Accuracy: {acc_baseline * 100:.2f}%")
print(f"New Architecture Test Accuracy: {acc_new * 100:.2f}%")



# --------------------------------------------------
# Plotting Loss and Gradient Norms
# --------------------------------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(loss_history_baseline, label="Baseline Circuit")
plt.plot(loss_history_new, label="New Architecture Circuit")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(grad_norm_history_baseline, label="Baseline Circuit")
plt.plot(grad_norm_history_new, label="New Architecture Circuit")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.legend()
plt.tight_layout()
plt.show()
