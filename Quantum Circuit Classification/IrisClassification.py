from sklearn.datasets import load_iris
import numpy as np
from pennylane import numpy as qnp
from pennylane.measurements import ProbabilityMP
import matplotlib.pyplot as plt
from torch import nn
import pennylane as qml
import torch
from sklearn.model_selection import train_test_split
from typing import Union
torch.manual_seed(3407)
np.random.seed(3407)
qnp.random.seed(3407)

data = load_iris()
X = data.data
y = data.target
X = X[y != 2]
y = y[y != 2]
print(f"Number of data points: {len(X)}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3407)
X_train = qnp.array(X_train)
X_test = qnp.array(X_test)
y_train = qnp.array(y_train)
y_test = qnp.array(y_test)
num_qubits = 2 * np.log2(X_train.shape[1]).astype(int)
dev = qml.device("default.qubit", wires=num_qubits)

def quantum_circuit(features:qnp.ndarray, params:qnp.ndarray, qubit_to_sample: int, qubits_to_encode:Union[range, list]) -> ProbabilityMP:
    """
    Returns the probability values of the Pauli-Z operator on the qubit specified by qubit_to_sample
    - features (qnp.ndarray): Input features to the quantum circuit
    - params (qnp.ndarray): Parameters of the quantum circuit
    - qubit_to_sample (int): Qubit to sample from the quantum circuit
    - qubits_to_encode (Union[range, list]): Qubits to encode the input features
    """
    if qubit_to_sample >= num_qubits:
        raise ValueError("The qubit to sample must be less than the number of qubits")
    qml.AmplitudeEmbedding(features, wires=qubits_to_encode, normalize=True)
    for qubit in range(num_qubits):
        if qubit not in qubits_to_encode:
            qml.Hadamard(wires=qubit)
    qml.StronglyEntanglingLayers(params, wires=range(num_qubits))
    return qml.probs(wires=qubit_to_sample)

@qml.qnode(dev)
def cost_circuit(features:qnp.ndarray, params:qnp.ndarray, testing:bool = False) -> float:
    """
    Executes the quantum circuit and returns the expectation value of the Pauli-Z operator on the first qubit or a random qubit, depending on the testing parameter
    
    Arguments:
    - features (qnp.ndarray): Input features to the quantum circuit
    - params (qnp.ndarray): Parameters of the quantum circuit
    - testing (bool): Whether the model is being tested or not (ie if only one qubit should be used)
    """
    if testing: qubit_to_sample = 0
    else: qubit_to_sample = qnp.random.randint(num_qubits)
    if testing: qubits_to_encode = range(int(num_qubits / 2))
    else: qubits_to_encode = qnp.random.choice(num_qubits, size = int(num_qubits/2), replace=False).tolist()
    return quantum_circuit(features, params, qubit_to_sample, qubits_to_encode)

@qml.qnode(dev)
def draw_circuit(features:qnp.ndarray, params:qnp.ndarray) -> float:
    """
    Executes the quantum circuit and returns the expectation value of the Pauli-Z operator on the first qubit, for visualization purposes
    
    Arguments:
    - features (qnp.ndarray): Input features to the quantum circuit
    - params (qnp.ndarray): Parameters of the quantum circuit
    """
    qubit_to_sample = 0
    qubits_to_encode = range(int(num_qubits / 2))
    return quantum_circuit(features, params, qubit_to_sample, qubits_to_encode)


def cost_function(params: qnp.ndarray, features: qnp.ndarray, labels: qnp.ndarray, not_random: bool) -> float:
    """
    Binary cross-entropy cost function for classification using probabilities directly from the quantum circuit.

    Arguments:
    - params (qnp.ndarray): Parameters of the quantum circuit
    - features (qnp.ndarray): Data to train
    - labels (qnp.ndarray): Labels corresponding to the data
    - not_random (bool): Whether to sample the same qubit each time or not
    """
    loss = 0.0
    for f, label in zip(features, labels):
        probabilities = cost_circuit(f, params, not_random)
        probability = probabilities[1]  # Probability of the '0' state, which we map to label 0
        loss += -label * qnp.log(probability + 1e-9) - (1 - label) * qnp.log(1 - probability + 1e-9)
    loss = loss / len(features)
    return loss
optimizer = qml.AdamOptimizer(stepsize=0.0001, beta1=0.9, beta2=0.999, eps=1e-8) ## Setting the optimizer to have the same behaviour between torch and pennylane

def train_quantum_model(data:qnp.ndarray, labels:qnp.ndarray, params:qnp.ndarray, 
                        epochs=10, deterministic:bool = False) -> qnp.ndarray:
    """
    Trains the quantum model using the cost function and optimizer

    Arguments:
    - data (qnp.ndarray): Data to train
    - labels (qnp.ndarray): Labels corresponding to the data
    - params (qnp.ndarray): Parameters of the quantum circuit
    - epochs (int): Number of epochs to train the model
    - deterministic (bool): Whether to sample the same qubit each time or not
    """
    features = data
    for epoch in range(epochs):
        params, cost = optimizer.step_and_cost(lambda p: cost_function(p, features, labels, deterministic), params)
        print(f"Epoch {epoch+1}: Cost = {cost}")
    return params

def test_quantum_model(data: qnp.ndarray, labels: qnp.ndarray, params: qnp.ndarray, override:bool = False, threshold=0.5, num_rep: int = 3) -> float:
    """
    Tests the quantum model and returns the accuracy using probabilities.

    Arguments:
    - data (qnp.ndarray): data to test
    - labels (qnp.ndarray): Labels corresponding to the data
    - params (qnp.ndarray): Parameters of the quantum circuit
    - override (bool): Whether to override certain settings in the cost circuit
    - threshold (float): Threshold for the binary classification
    - num_rep (int): Number of times each prediction is repeated, if override is True
    """
    features = data
    predictions = [cost_circuit(f, params, not override)[1] for f in features] 
    if override:
        predictions = []
        for i in range(num_rep):
            new_pred = [cost_circuit(f, params, not override)[1] for f in features]
            predictions.append(new_pred)
        predictions = qnp.array(predictions).mean(axis=0)
        predictions = (predictions > threshold).astype(int)
    else:
        predictions = (qnp.array(predictions) > threshold).astype(int) 
    accuracy = qnp.mean(predictions != labels)
    return accuracy.numpy()

def train_classical_model(model:nn.Module, features:np.ndarray, labels:np.ndarray, epochs:int=10) -> nn.Module:
    """
    Trains the classical model using the mean squared error loss

    Arguments:
    - model (torch.nn.Module): Model to train
    - features (qnp.ndarray): Feature matrix to train
    - labels (qnp.ndarray): Labels corresponding to the data
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
    - labels (qnp.ndarray): Labels corresponding to the data
    """
    features = torch.tensor(features.numpy(), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        predictions = torch.argmax(outputs, dim=1)
    accuracy = torch.mean((predictions == torch.tensor(labels)).float())
    return accuracy.numpy()

def generate_pgd_adversarial_example_classical(model:nn.Module, features:qnp.ndarray, labels:qnp.ndarray, epsilon:float=2, alpha:float=0.01, num_iter:int=50) -> qnp.ndarray:
    """
    Generate adversarial examples using Projected Gradient Descent for the classical model
    The adversarial examples are generated by perturbing the input features using the following formula:
    x' = x + alpha * sign(grad(loss, x))

    Arguments:
    - model (torch.nn.Module): Trained model
    - features (qnp.ndarray): Feature matrix to generate adversarial examples
    - labels (qnp.ndarray): Labels corresponding to the data
    - epsilon (float): Maximum perturbation allowed (default: 2)
    - alpha (float): Step size for the perturbation (default: 0.01)
    - num_iter (int): Number of iterations for the PGD algorithm (default: 50)
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
    - labels (qnp.ndarray): Labels corresponding to the data
    - epsilon (float): Maximum perturbation allowed (default: 2)
    - alpha (float): Step size for the perturbation (default: 0.01)
    - num_iter (int): Number of iterations for the PGD algorithm (default: 50)
    """
    features = torch.tensor(features.numpy(), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    original_features = features.clone()
    features.requires_grad = True
    
    for i in range(num_iter):
        optimizer = torch.optim.Adam([features], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        optimizer.zero_grad()
        outputs = cost_circuit(features, params, testing=True)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        with torch.no_grad():
            features += alpha * features.grad.sign()
            perturbation = torch.clamp(features - original_features, min=-epsilon, max=epsilon)
            features = torch.clamp(original_features + perturbation, min=0, max=1)
        features.requires_grad = True
    return qnp.array([f.detach().numpy() for f in features])

num_layers = 4
epochs = 3
params = qnp.random.random((num_layers, num_qubits, 3))
print("-----------------------------------")
print("Quantum Model Training...")
params = train_quantum_model(X_train, y_train, params, epochs=epochs, deterministic = True)
print("Quantum Model Trained, Testing...")
accuracy_quantum = test_quantum_model(X_test, y_test, params)
print(f"Accuracy of the Parametrized Quantum Circuit: {accuracy_quantum*100:.2f}%")
print(f"Number of parameters in the quantum model: {len(params.flatten())}")
classical_model = nn.Sequential(
    nn.Linear(X_train.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 2),
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
print(f"Difference in accuracy between the Classical and Quantum Model: {abs(float(accuracy_classic_adv) - float(accuracy_quantum_adv))*100:.2f}%\nIs the quantum model more robust? {accuracy_quantum_adv >= accuracy_classic_adv}")
accuracy_quantum_random = test_quantum_model(adversarial_example_quantum, y_test, params, override=True)
print("-----------------------------------")
print(f"Accuracy of the Quantum Model on the Adversarial Example: {accuracy_quantum_random*100:.2f}% (Random Qubit Sampling)")
print(f"Difference in accuracy between random and deterministic sampling: {abs(accuracy_quantum_adv - accuracy_quantum_random)*100:.2f}%\nIs random sampling better? {accuracy_quantum_random >= accuracy_quantum_adv}\nIs random sampling better than the classical model? {accuracy_quantum_random >= accuracy_classic_adv}")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
fig, ax = qml.draw_mpl(draw_circuit, style = "black_white", expansion_strategy="device", show_all_wires=True, decimals = 2)(X_train[0], params)
fig.savefig("CircuitIris.png", dpi = 600)
plt.show()
