#%%
# --------------------------------------------------
# Imports and Seeding
# --------------------------------------------------
import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)



# --------------------------------------------------
# Data Generation: Function Regression with a Damped Oscillator
# --------------------------------------------------
def damped_oscillator(x, beta=5.0):
    """
    Defines a damped oscillator function.
    Here we use the sum of the input vector as the effective time parameter.
    """
    t = np.sum(x)
    return np.exp(-t) * np.cos(beta * t)

d_baseline = 6
d_new = 5

N = 200  
X_baseline = np.random.uniform(0, 0.2, (N, d_baseline))
X_new = np.random.uniform(0, 0.2, (N, d_new))

y_baseline = np.array([damped_oscillator(x) for x in X_baseline])
y_new = np.array([damped_oscillator(x) for x in X_new])

Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_baseline, y_baseline, test_size=0.2, random_state=42)
Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)



# --------------------------------------------------
# Device and Circuit Definitions
# --------------------------------------------------
n_wires_baseline = 6
n_wires_new = 6

dev_baseline = qml.device("default.qubit", wires=n_wires_baseline)
dev_new = qml.device("default.qubit", wires=n_wires_new)

@qml.qnode(dev_baseline, interface="autograd")
def baseline_circuit(x, weights):
    for i in range(d_baseline):
        qml.RY(x[i], wires=i)
    L = weights.shape[0]
    for layer in range(L):
        for wire in range(n_wires_baseline):
            qml.RZ(weights[layer, wire, 0], wires=wire)
            qml.RY(weights[layer, wire, 1], wires=wire)
            qml.RX(weights[layer, wire, 2], wires=wire)
        for wire in range(n_wires_baseline):
            qml.CNOT(wires=[wire, (wire+1) % n_wires_baseline])
    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev_new, interface="autograd")
def new_architecture_circuit(x, weights):
    for i in range(d_new):
        qml.RY(x[i], wires=i)
    qml.Hadamard(wires=n_wires_new - 1)
    L = weights.shape[0]
    for layer in range(L):
        for wire in range(n_wires_new):
            qml.RZ(weights[layer, wire, 0], wires=wire)
            qml.RY(weights[layer, wire, 1], wires=wire)
            qml.RX(weights[layer, wire, 2], wires=wire)
        for wire in range(n_wires_new):
            qml.CNOT(wires=[wire, (wire+1) % n_wires_new])
    return qml.expval(qml.PauliZ(wires=0))



# --------------------------------------------------
# Cost Function and Loss Definition (MSE)
# --------------------------------------------------
def mse_loss(pred, target):
    return (pred - target)**2

def cost_baseline(weights, X, y):
    loss = 0
    for x_val, target in zip(X, y):
        pred = baseline_circuit(x_val, weights)
        loss += mse_loss(pred, target)
    return loss / len(X)

def cost_new(weights, X, y):
    loss = 0
    for x_val, target in zip(X, y):
        pred = new_architecture_circuit(x_val, weights)
        loss += mse_loss(pred, target)
    return loss / len(X)


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
print("Training baseline circuit (rotational embedding on all qubits)")
for epoch in range(num_epochs):
    weights_baseline, loss_val = opt.step_and_cost(lambda w: cost_baseline(w, Xb_train, yb_train), weights_baseline)
    loss_history_baseline.append(loss_val)
    
    grad = qml.grad(lambda w: cost_baseline(w, Xb_train, yb_train))(weights_baseline)
    grad_norm = np.linalg.norm(grad)
    grad_norm_history_baseline.append(grad_norm)
    
    print(f"Epoch {epoch+1:03d}: Loss = {loss_val:.4f}, Grad Norm = {grad_norm:.4f}", end="\r")



# --------------------------------------------------
# Training Loop for New Architecture Circuit (partial encoding + auxiliary)
# --------------------------------------------------
print("\nTraining new architecture circuit (rotational embedding with auxiliary qubit)")
for epoch in range(num_epochs):
    weights_new, loss_val = opt.step_and_cost(lambda w: cost_new(w, Xn_train, yn_train), weights_new)
    loss_history_new.append(loss_val)
    
    grad = qml.grad(lambda w: cost_new(w, Xn_train, yn_train))(weights_new)
    grad_norm = np.linalg.norm(grad)
    grad_norm_history_new.append(grad_norm)
    
    print(f"Epoch {epoch+1:03d}: Loss = {loss_val:.4f}, Grad Norm = {grad_norm:.4f}", end="\r")



# --------------------------------------------------
# Evaluation on Test Set
# --------------------------------------------------
def predict_regression(circuit_fn, weights, x):
    return circuit_fn(x, weights)

preds_baseline = [predict_regression(baseline_circuit, weights_baseline, x_val) for x_val in Xb_test]
mse_baseline = np.mean((np.array(preds_baseline) - yb_test)**2)

preds_new = [predict_regression(new_architecture_circuit, weights_new, x_val) for x_val in Xn_test]
mse_new = np.mean((np.array(preds_new) - yn_test)**2)

print(f"\nBaseline Test MSE: {mse_baseline:.4f}")
print(f"New Architecture Test MSE: {mse_new:.4f}")



# --------------------------------------------------
# Plotting Loss and Gradient Norms
# --------------------------------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history_baseline, label="Baseline Circuit")
plt.plot(loss_history_new, label="New Architecture Circuit")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(grad_norm_history_baseline, label="Baseline Circuit")
plt.plot(grad_norm_history_new, label="New Architecture Circuit")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.legend()
plt.show()
