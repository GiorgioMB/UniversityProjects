from sklearn.datasets import load_iris
import numpy as np
from pennylane import numpy as qnp
from pennylane.measurements import ExpectationMP
import matplotlib.pyplot as plt
from torch import nn
import pennylane as qml
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
torch.manual_seed(62101)
np.random.seed(62101)
qnp.random.seed(62101)

data = load_iris()
X = data.data
y = data.target
X = X[y != 2]
y = y[y != 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=62101)
X_train = qnp.array(X_train)
X_test = qnp.array(X_test)
y_train = qnp.array(y_train)
y_test = qnp.array(y_test)
num_qubits = np.log2(X_train.shape[1]).astype(int)
dev = qml.device("default.qubit", wires=num_qubits)

## Define the quantum circuit
def quantum_circuit(features:qnp.ndarray, params:qnp.ndarray, qubit_to_sample: int) -> ExpectationMP:
    """
    Returns the expectation value of the Pauli-Z operator on the qubit specified by qubit_to_sample
    - features (qnp.ndarray): Iqnput features to the quantum circuit
    - params (qnp.ndarray): Parameters of the quantum circuit
    - qubit_to_sample (int): Qubit to sample from the quantum circuit
    """
    if qubit_to_sample >= num_qubits:
        raise ValueError("The qubit to sample must be less than the number of qubits")
    qml.AmplitudeEmbedding(features, wires=range(num_qubits), normalize=True)
    qml.StronglyEntanglingLayers(params, wires=range(num_qubits))
    return qml.expval(qml.PauliZ(qubit_to_sample))

@qml.qnode(dev)
def cost_circuit(features:qnp.ndarray, params:qnp.ndarray, testing:bool = False) -> float:
    """
    Executes the quantum circuit and returns the expectation value of the Pauli-Z operator on the first qubit or a random qubit, depending on the testing parameter
    
    Arguments:
    - features (qnp.ndarray): Iqnput features to the quantum circuit
    - params (qnp.ndarray): Parameters of the quantum circuit
    - testing (bool): Whether the model is being tested or not (ie if only one qubit should be used)
    """
    if testing: qubit_to_sample = 0
    else: qubit_to_sample = qnp.random.randint(num_qubits)
    ##Uncomment the line below to see the qubit being sampled
    ##print(f"Sampling for qubit: {qubit_to_sample + 1}")
    return quantum_circuit(features, params, qubit_to_sample)

@qml.qnode(dev)
def draw_circuit(features:qnp.ndarray, params:qnp.ndarray) -> float:
    """
    Executes the quantum circuit and returns the expectation value of the Pauli-Z operator on the first qubit, for visualization purposes
    
    Arguments:
    - features (qnp.ndarray): Iqnput features to the quantum circuit
    - params (qnp.ndarray): Parameters of the quantum circuit
    """
    qubit_to_sample = 0
    return quantum_circuit(features, params, qubit_to_sample)


## Define the activation function and cost function
def sigmoid(x:qnp.ndarray) -> qnp.ndarray:
    """
    Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    Maps any real value into the range (0, 1), suitable for binary classification.
    """
    return 1 / (1 + qnp.exp(-x))

def cost_function(params:qnp.ndarray, features:qnp.ndarray, labels:qnp.ndarray, not_random:bool) -> float:
    """
    Binary cross-entropy cost function for classification

    Arguments:
    - params (qnp.ndarray): Parameters of the quantum circuit
    - features (qnp.ndarray): Image data to train
    - labels (qnp.ndarray): Labels corresponding to the image data
    - not_random (bool): Whether to sample the same qubit each time or not
    """
    loss = 0.0
    for f, label in zip(features, labels):
        logits = cost_circuit(f, params, not_random)  # ensure cost_circuit outputs logits
        prediction = sigmoid(logits)       # apply sigmoid to convert logits to probabilities
        loss += -label * qnp.log(prediction + 1e-9) - (1 - label) * qnp.log(1 - prediction + 1e-9)  # 1e-9 for numerical stability
    loss = loss / len(features)
    return loss


## Define the training and testing functions for the quantum model
optimizer = qml.AdamOptimizer(stepsize=0.01, beta1=0.9, beta2=0.999, eps=1e-8) ## Setting the optimizer to have the same behaviour between torch and pennylane
def train_quantum_model(features:qnp.ndarray, labels:qnp.ndarray, params:qnp.ndarray, epochs=10, deterministic:bool = False) -> qnp.ndarray:
    """
    Trains the quantum model using the cost function and optimizer

    Arguments:
    - features (qnp.ndarray): Feature matrix to train
    - labels (qnp.ndarray): Labels corresponding to the image data
    - params (qnp.ndarray): Parameters of the quantum circuit
    - epochs (int): Number of epochs to train the model
    - deterministic (bool): Whether to sample the same qubit each time or not
    """
    for epoch in range(epochs):
        params, cost = optimizer.step_and_cost(lambda p: cost_function(p, features, labels, deterministic), params)
        print(f"Epoch {epoch+1}: Cost = {cost:.4f}")
    return params

def test_quantum_model(features:qnp.ndarray, labels:qnp.ndarray, params:qnp.ndarray, override: bool = False) -> float:
    """
    Tests the quantum model and returns the accuracy

    Arguments:
    - features (qnp.ndarray): Feature matrix to test
    - labels (qnp.ndarray): Labels corresponding to the image data
    - params (qnp.ndarray): Parameters of the quantum circuit
    - override (bool): Whether to override the testing setting and sample random qubits
    """
    raw_logits = [cost_circuit(f, params, testing = (not override)) for f in features]
    accuracy = qnp.mean([(qnp.sign(pred) != lab) for pred, lab in zip(raw_logits, labels)])
    return accuracy


## Define the training and testing functions for the classical model
def train_classical_model(model:nn.Module, features:np.ndarray, labels:np.ndarray, epochs:int=10) -> nn.Module:
    """
    Trains the classical model using the mean squared error loss

    Arguments:
    - model (torch.nn.Module): Model to train
    - features (qnp.ndarray): Feature matrix to train
    - labels (qnp.ndarray): Labels corresponding to the image data
    - epochs (int): Number of epochs to train the model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas = (0.9, 0.999), eps = 1e-8)
    features = torch.tensor(features.numpy(), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item() / len(features):.4f}")
    return model

def test_classical_model(model: nn.Module, features:np.ndarray, labels:np.ndarray) -> float:
    """
    Tests the classical model and returns the accuracy

    Arguments:
    - model (torch.nn.Module): Trained model
    - features (qnp.ndarray): Feature matrix to test
    - labels (qnp.ndarray): Labels corresponding to the image data
    """
    features = torch.tensor(features.numpy(), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        predictions = torch.argmax(outputs, dim=1)
    accuracy = torch.mean((predictions == torch.tensor(labels)).float())
    return accuracy

## Define the function to generate adversarial examples
def generate_pgd_adversarial_example_classical(model:nn.Module, features:qnp.ndarray, labels:qnp.ndarray, epsilon:float=2, alpha:float=0.01, num_iter:int=50) -> qnp.ndarray:
    """
    Generate adversarial examples using Projected Gradient Descent for the classical model
    The adversarial examples are generated by perturbing the input features using the following formula:
    x' = x + alpha * sign(grad(loss, x))

    Arguments:
    - model (torch.nn.Module): Trained model
    - features (qnp.ndarray): Feature matrix to generate adversarial examples
    - labels (qnp.ndarray): Labels corresponding to the image data
    - epsilon (float): Maximum perturbation allowed (default: 0.2)
    - alpha (float): Step size for the perturbation (default: 0.01)
    - num_iter (int): Number of iterations for the PGD algorithm (default: 30)
    """
    features = torch.tensor(features.numpy(), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    original_features = features.clone()
    features.requires_grad = True
    for i in range(num_iter):
        optimizer = torch.optim.Adam([features], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        optimizer.zero_grad()
        outputs = model(features)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        with torch.no_grad():
            features += alpha * features.grad.sign()
            perturbation = torch.clamp(features - original_features, min=-epsilon, max=epsilon)
            features = torch.clamp(original_features + perturbation, min=0, max=1)
        features.requires_grad = True
    return qnp.array([f.detach().numpy() for f in features])

def generate_pgd_adversarial_example_quantum(params:qnp.ndarray, features:qnp.ndarray, labels:qnp.ndarray, epsilon:float=2, alpha:float=0.01, num_iter:int=50) -> np.ndarray:
    """
    Generate adversarial examples using Projected Gradient Descent for the quantum model
    The adversarial examples are generated by perturbing the input features using the following formula:
    x' = x + alpha * sign(grad(loss, x))

    Arguments:
    - params (qnp.ndarray): Parameters of the quantum circuit
    - features (qnp.ndarray): Feature matrix to generate adversarial examples
    - labels (qnp.ndarray): Labels corresponding to the image data
    - epsilon (float): Maximum perturbation allowed (default: 0.2)
    - alpha (float): Step size for the perturbation (default: 0.01)
    - num_iter (int): Number of iterations for the PGD algorithm (default: 30)
    """
    features = torch.tensor(features.numpy(), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    original_features = features.clone()
    features.requires_grad = True
    
    for i in range(num_iter):
        optimizer = torch.optim.Adam([features], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        optimizer.zero_grad()
        outputs = nn.Sigmoid()(cost_circuit(features, params, testing=True))
        outputs = torch.cat((outputs, 1 - outputs), dim=0)
        outputs = outputs.view(-1, 2)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        with torch.no_grad():
            features += alpha * features.grad.sign()
            perturbation = torch.clamp(features - original_features, min=-epsilon, max=epsilon)
            features = torch.clamp(original_features + perturbation, min=0, max=1)
        features.requires_grad = True
    return qnp.array([f.detach().numpy() for f in features])

## Main execution
num_layers = 2
epochs = 2
params = qnp.random.random((num_layers, num_qubits, 3))
print("-----------------------------------")
print("Quantum Model Training...")
params = train_quantum_model(X_train, y_train, params, epochs=epochs)
print("Quantum Model Trained, Testing...")
accuracy_quantum = test_quantum_model(X_test, y_test, params)
print(f"Accuracy of the Parametrized Quantum Circuit: {accuracy_quantum*100:.2f}%")
print(f"Number of parameters in the quantum model: {len(params.flatten())}")
classical_model = nn.Sequential(
    nn.Linear(X_train.shape[1], 16),
    nn.ReLU(),
    nn.Linear(16, 2),
    nn.Sigmoid()
)
print("-----------------------------------")
print("Classical Model Training...")
trained_model = train_classical_model(classical_model, X_train, y_train, epochs=epochs*5)
print("Classical Model Trained, Testing...")
accuracy_classic = test_classical_model(trained_model, X_test, y_test)
print(f"Accuracy of the Classical Model with 5 times the epochs: {accuracy_classic*100:.2f}%")
print(f"Number of parameters in the classical model: {sum(p.numel() for p in trained_model.parameters())}")
adversarial_example_classical = generate_pgd_adversarial_example_classical(trained_model, X_test, y_test)
adversarial_example_quantum = generate_pgd_adversarial_example_quantum(params, X_test, y_test)
accuracy_classic_adv = test_classical_model(trained_model, adversarial_example_classical, y_test)
accuracy_quantum_adv = test_quantum_model(adversarial_example_quantum, y_test, params)
print("-----------------------------------")
print(f"Accuracy of the Classical Model on the Adversarial Example: {accuracy_classic_adv*100:.2f}%")
print(f"Accuracy of the Quantum Model on the Adversarial Example: {accuracy_quantum_adv*100:.2f}%")
print(f"Drop in accuracy for the Classical Model: {(accuracy_classic - accuracy_classic_adv)*100:.2f}%")
print(f"Drop in accuracy for the Quantum Model: {(accuracy_quantum - accuracy_quantum_adv)*100:.2f}%")
print(f"Difference in accuracy between the Classical and Quantum Model: {abs(accuracy_classic_adv - accuracy_quantum_adv)*100:.2f}%\nIs the quantum model more robust? {accuracy_quantum_adv >= accuracy_classic_adv}")
accuracy_quantum_random = test_quantum_model(adversarial_example_quantum, y_test, params, override=True)
print("-----------------------------------")
print(f"Accuracy of the Quantum Model on the Adversarial Example: {accuracy_quantum_random*100:.2f}% (Random Qubit Sampling)")
print(f"Difference in accuracy between random and deterministic sampling: {abs(accuracy_quantum_adv - accuracy_quantum_random)*100:.2f}%\nIs random sampling better? {accuracy_quantum_random >= accuracy_quantum_adv}\nIs random sampling better than the classical model? {accuracy_quantum_random >= accuracy_classic_adv}")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
fig, ax = qml.draw_mpl(draw_circuit, style = "black_white", expansion_strategy="device", show_all_wires=True, decimals = 2)(X_train[0], params)
fig.savefig("CircuitIris.png", dpi = 600)
plt.show()
