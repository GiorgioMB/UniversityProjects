# --------------------------------------------------
# Imports and Seeding
# --------------------------------------------------
import pennylane as qml
import pennylane.numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
np.random.seed(42)



# --------------------------------------------------
# Data Preparation
# --------------------------------------------------
digits = load_digits()
mask = (digits.target == 0) | (digits.target == 1)
X = digits.data[mask]  
y = digits.target[mask]

def normalize(x):
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x

X_baseline = np.array([normalize(x) for x in X])  
X_new = np.array([normalize(x[:32]) for x in X])    

Xb_train, Xb_test, y_train, y_test = train_test_split(X_baseline, y, test_size=0.2, random_state=42)
Xn_train, Xn_test, _, _ = train_test_split(X_new, y, test_size=0.2, random_state=42)



# --------------------------------------------------
# Device and Circuit Definitions
# --------------------------------------------------
n_wires = 6  
dev_baseline = qml.device("default.qubit", wires=n_wires)
dev_new = qml.device("default.qubit", wires=n_wires)

# Baseline Circuit: amplitude embedding on all qubits
@qml.qnode(dev_baseline, interface="autograd")
def baseline_circuit(x, weights):
    qml.AmplitudeEmbedding(x, wires=range(n_wires), normalize=True)
    L = weights.shape[0]
    for layer in range(L):
        for wire in range(n_wires):
            qml.RZ(weights[layer, wire, 0], wires=wire)
            qml.RY(weights[layer, wire, 1], wires=wire)
            qml.RX(weights[layer, wire, 2], wires=wire)
        for wire in range(n_wires):
            qml.CNOT(wires=[wire, (wire+1) % n_wires])
    return qml.probs(wires=0)


# New Architecture Circuit: amplitude embedding on first 5 qubits; auxiliary qubit (wire 5) in |+>
@qml.qnode(dev_new, interface="autograd")
def new_architecture_circuit(x, weights):
    qml.AmplitudeEmbedding(x, wires=range(n_wires - 1), normalize=True)
    qml.Hadamard(wires=n_wires - 1)
    L = weights.shape[0]
    for layer in range(L):
        for wire in range(n_wires):
            qml.RZ(weights[layer, wire, 0], wires=wire)
            qml.RY(weights[layer, wire, 1], wires=wire)
            qml.RX(weights[layer, wire, 2], wires=wire)
        for wire in range(n_wires):
            qml.CNOT(wires=[wire, (wire+1) % n_wires])
    return qml.probs(wires=0)




# --------------------------------------------------
# Cost Functions and Loss Definitions
# --------------------------------------------------
def cross_entropy_loss(probs, target):
    return -np.sum(target * np.log(probs + 1e-10))

def cost_baseline(weights, X, y):
    loss = 0
    for x_val, label in zip(X, y):
        probs = baseline_circuit(x_val, weights)
        target = np.array([1, 0]) if label == 0 else np.array([0, 1])
        loss += cross_entropy_loss(probs, target)
    return loss / len(X)

def cost_new(weights, X, y):
    loss = 0
    for x_val, label in zip(X, y):
        probs = new_architecture_circuit(x_val, weights)
        target = np.array([1, 0]) if label == 0 else np.array([0, 1])
        loss += cross_entropy_loss(probs, target)
    return loss / len(X)



# --------------------------------------------------
# Training Setup
# --------------------------------------------------
num_epochs = 200 ##Note: this will take a while, testing on my local machine I used only 10 epochs
L_layers = 2 

weights_baseline = np.random.uniform(0, 2*np.pi, size=(L_layers, n_wires, 3), requires_grad=True)
weights_new = np.random.uniform(0, 2*np.pi, size=(L_layers, n_wires, 3), requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.01)

loss_history_baseline = []
grad_norm_history_baseline = []

loss_history_new = []
grad_norm_history_new = []



# --------------------------------------------------
# Training Loop for Baseline Circuit (all qubits encode data)
# --------------------------------------------------
print("Training baseline circuit (all qubits used for encoding)")
for epoch in range(num_epochs):
    weights_baseline, loss_val = opt.step_and_cost(lambda w: cost_baseline(w, Xb_train, y_train), weights_baseline)
    loss_history_baseline.append(loss_val)
    
    grad = qml.grad(lambda w: cost_baseline(w, Xb_train, y_train))(weights_baseline)
    grad_norm = np.linalg.norm(grad)
    grad_norm_history_baseline.append(grad_norm)
    
    print(f"Epoch {epoch+1:03d}: Loss = {loss_val:.4f}, Grad Norm = {grad_norm:.4f}", end="\r")



# --------------------------------------------------
# Training Loop for New Architecture Circuit (partial encoding + auxiliary)
# --------------------------------------------------
print("\nTraining new architecture circuit (partial encoding + auxiliary)")
for epoch in range(num_epochs):
    weights_new, loss_val = opt.step_and_cost(lambda w: cost_new(w, Xn_train, y_train), weights_new)
    loss_history_new.append(loss_val)
    grad = qml.grad(lambda w: cost_new(w, Xn_train, y_train))(weights_new)
    grad_norm = np.linalg.norm(grad)
    grad_norm_history_new.append(grad_norm)

    print(f"Epoch {epoch+1:03d}: Loss = {loss_val:.4f}, Grad Norm = {grad_norm:.4f}", end="\r")



# --------------------------------------------------
# Evaluation on Test Set
# --------------------------------------------------
def predict(circuit_fn, weights, x):
    probs = circuit_fn(x, weights)
    return 0 if probs[0] >= 0.5 else 1

preds_baseline = [predict(baseline_circuit, weights_baseline, x_val) for x_val in Xb_test]
acc_baseline = np.mean(np.array(preds_baseline) == y_test)

preds_new = [predict(new_architecture_circuit, weights_new, x_val) for x_val in Xn_test]
acc_new = np.mean(np.array(preds_new) == y_test)

print(f"\nBaseline Test Accuracy: {acc_baseline * 100:.2f}%")
print(f"New Architecture Test Accuracy: {acc_new * 100:.2f}%")



# --------------------------------------------------
# Plotting Loss and Gradient Norms
# --------------------------------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history_baseline, label="Baseline Circuit")
plt.plot(loss_history_new, label="New Architecture Circuit")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(grad_norm_history_baseline, label="Baseline Circuit")
plt.plot(grad_norm_history_new, label="New Architecture Circuit")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.legend()
