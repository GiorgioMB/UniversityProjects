# -------------------------------------------------------------------------
# Imports and Seeding
# -------------------------------------------------------------------------
import pennylane as qml
import pennylane.numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
np.random.seed(42)



# -------------------------------------------------------------------------
# Device and Circuit Definitions
# -------------------------------------------------------------------------
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



# -------------------------------------------------------------------------
# Cost Functions, Loss Function and Test Function Definition
# -------------------------------------------------------------------------
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


def predict(circuit_fn, weights, x):
    probs = circuit_fn(x, weights)
    return 0 if probs[0] >= 0.5 else 1



# -------------------------------------------------------------------------
# Data Preparation
# -------------------------------------------------------------------------
def load_and_prepare():
    seed = 42
    digits = load_digits()
    mask = (digits.target == 0) | (digits.target == 1)
    X = digits.data[mask]  
    y = digits.target[mask]

    def normalize(x):
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x

    X_baseline = np.array([normalize(x) for x in X])  
    X_new = np.array([normalize(x[:32]) for x in X])    

    Xb_train, Xb_test, y_train, y_test = train_test_split(X_baseline, y, test_size=0.2, random_state=seed)
    Xn_train, Xn_test, _, _ = train_test_split(X_new, y, test_size=0.2, random_state=seed)
    return Xb_train, Xb_test, y_train, y_test, Xn_train, Xn_test



# -------------------------------------------------------------------------
# Training Setup
# -------------------------------------------------------------------------
num_epochs = 200 ##Note: this will take a while, testing on my local machine I used only 10 epochs
N_layers = 3
N_repeats = 100


total_losses_baseline = []
total_losses_new = []
total_grad_norms_baseline = []
total_grad_norms_new = []
accuracies_baseline = []
accuracies_new = []


for repeat in range(N_repeats):
    print(f"Repeat {repeat + 1}/{N_repeats}...")
    Xb_train, Xb_test, y_train, y_test, Xn_train, Xn_test = load_and_prepare()
    weights_baseline = np.random.uniform(0, 2*np.pi, size=(N_layers, n_wires, 3), requires_grad=True)
    weights_new = np.random.uniform(0, 2*np.pi, size=(N_layers, n_wires, 3), requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=0.01)
    loss_history_baseline = []
    grad_norm_history_baseline = []
    
    loss_history_new = []
    grad_norm_history_new = []

    # -------------------------------------------------------------------------
    # Training Loop for Baseline Circuit (all qubits encode data)
    # -------------------------------------------------------------------------
    print("Training baseline circuit (all qubits used for encoding)")
    for epoch in range(num_epochs):
        weights_baseline, loss_val = opt.step_and_cost(lambda w: cost_baseline(w, Xb_train, y_train), weights_baseline)
        loss_history_baseline.append(loss_val)
        
        grad = qml.grad(lambda w: cost_baseline(w, Xb_train, y_train))(weights_baseline)
        grad_norm = np.linalg.norm(grad)
        grad_norm_history_baseline.append(grad_norm)
        
        print(f"Epoch {epoch+1:03d}: Loss = {loss_val:.4f}, Grad Norm = {grad_norm:.4f}", end="\r")


    # -------------------------------------------------------------------------
    # Training Loop for New Architecture Circuit (partial encoding + auxiliary)
    # -------------------------------------------------------------------------
    print("\nTraining new architecture circuit (partial encoding + auxiliary)")
    for epoch in range(num_epochs):
        weights_new, loss_val = opt.step_and_cost(lambda w: cost_new(w, Xn_train, y_train), weights_new)
        loss_history_new.append(loss_val)
        grad = qml.grad(lambda w: cost_new(w, Xn_train, y_train))(weights_new)
        grad_norm = np.linalg.norm(grad)
        grad_norm_history_new.append(grad_norm)

        print(f"Epoch {epoch+1:03d}: Loss = {loss_val:.4f}, Grad Norm = {grad_norm:.4f}", end="\r")


    # -------------------------------------------------------------------------
    # Evaluation on Test Set
    # -------------------------------------------------------------------------
    preds_baseline = [predict(baseline_circuit, weights_baseline, x_val) for x_val in Xb_test]
    acc_baseline = np.mean(np.array(preds_baseline) == y_test)

    preds_new = [predict(new_architecture_circuit, weights_new, x_val) for x_val in Xn_test]
    acc_new = np.mean(np.array(preds_new) == y_test)

    print(f"\nBaseline Test Accuracy: {acc_baseline * 100:.2f}%")
    print(f"New Architecture Test Accuracy: {acc_new * 100:.2f}%")

    accuracies_baseline.append(acc_baseline)
    accuracies_new.append(acc_new)

    total_losses_baseline.append(loss_history_baseline)
    total_losses_new.append(loss_history_new)
    total_grad_norms_baseline.append(grad_norm_history_baseline)
    total_grad_norms_new.append(grad_norm_history_new)
    print(f"\nRepeat {repeat + 1}/{N_repeats} completed.\n")



# -------------------------------------------------------------------------
# Preparing Data For Plotting
# -------------------------------------------------------------------------
losses_baseline_array = np.array(total_losses_baseline)
losses_new_array = np.array(total_losses_new)

grads_baseline_array = np.array(total_grad_norms_baseline)
grads_new_array = np.array(total_grad_norms_new)

mean_loss_baseline = np.mean(losses_baseline_array, axis=0)
min_loss_baseline = np.min(losses_baseline_array, axis=0)
max_loss_baseline = np.max(losses_baseline_array, axis=0)

mean_loss_new = np.mean(losses_new_array, axis=0)
min_loss_new = np.min(losses_new_array, axis=0)
max_loss_new = np.max(losses_new_array, axis=0)

mean_grad_baseline = np.mean(grads_baseline_array, axis=0)
min_grad_baseline = np.min(grads_baseline_array, axis=0)
max_grad_baseline = np.max(grads_baseline_array, axis=0)

mean_grad_new = np.mean(grads_new_array, axis=0)
min_grad_new = np.min(grads_new_array, axis=0)
max_grad_new = np.max(grads_new_array, axis=0)



# -------------------------------------------------------------------------
# Plotting Loss and Gradient Norms
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 4))

# Plot Losses
plt.subplot(1, 2, 1)
epochs = np.arange(num_epochs)

plt.plot(epochs, mean_loss_baseline, label="Baseline (mean)", linewidth=2)
plt.fill_between(epochs, min_loss_baseline, max_loss_baseline, alpha=0.3, label="Baseline (min-max)")

plt.plot(epochs, mean_loss_new, label="New Arch (mean)", linewidth=2)
plt.fill_between(epochs, min_loss_new, max_loss_new, alpha=0.3, label="New Arch (min-max)")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

# Plot Gradient Norms
plt.subplot(1, 2, 2)
plt.plot(epochs, mean_grad_baseline, label="Baseline (mean)", linewidth=2)
plt.fill_between(epochs, min_grad_baseline, max_grad_baseline, alpha=0.3, label="Baseline (min-max)")

plt.plot(epochs, mean_grad_new, label="New Arch (mean)", linewidth=2)
plt.fill_between(epochs, min_grad_new, max_grad_new, alpha=0.3, label="New Arch (min-max)")

plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm over Epochs")
plt.legend()

plt.tight_layout()
plt.show()



# -------------------------------------------------------------------------
# Box plots of Test Accuracies
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.boxplot([accuracies_baseline, accuracies_new], labels=["Baseline", "New Architecture"])
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Distribution")
plt.ylim(0, 1)
plt.grid()
plt.show()
