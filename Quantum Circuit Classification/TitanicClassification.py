"""
Note: this file was never run due to computational constraints, may not be actually working
"""
import pennylane as qml
from pennylane.measurements import ProbabilityMP
from pennylane import numpy as qnp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import kaggle 
from typing import Union
import pandas as pd
torch.manual_seed(3407)
np.random.seed(3407)
qnp.random.seed(3407)
kaggle.api.authenticate()

kaggle.api.dataset_download_files('yasserh/titanic-dataset', path='data', unzip=True)
dataset = pd.read_csv('data/Titanic-Dataset.csv')
dataset = dataset.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
onehot = OneHotEncoder()
embarked = onehot.fit_transform(dataset['Embarked'].values.reshape(-1,1)).toarray()
embarked = pd.DataFrame(embarked, columns=['Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_nan'])
dataset = pd.concat([dataset, embarked], axis=1)
dataset = dataset.drop(columns=['Embarked'])
sex_onehot = onehot.fit_transform(dataset['Sex'].values.reshape(-1,1)).toarray()
sex_onehot = pd.DataFrame(sex_onehot, columns = ['Male', 'Female'])
dataset = pd.concat([dataset, sex_onehot], axis=1)
dataset = dataset.drop(columns=['Sex', 'Female'])
dataset = dataset.dropna()
dataset = dataset.drop(columns=['Embarked_nan'])
X = dataset.drop(columns=['Survived'])
y = dataset['Survived']
X = X.to_numpy()
y = y.to_numpy()


num_qubits = 8
dev = qml.device("default.qubit", wires=num_qubits)

def quantum_circuit(features:qnp.ndarray, params:qnp.ndarray, qubit_to_sample: int, qubits_to_encode:Union[range, list]) -> ProbabilityMP:
    """
    Returns the expectation value of the Pauli-Z operator on the qubit specified by qubit_to_sample
    - features (qnp.ndarray): Input features to the quantum circuit
    - params (qnp.ndarray): Parameters of the quantum circuit
    - qubit_to_sample (int): Qubit to sample from the quantum circuit
    - qubits_to_encode (Union[range, list]): Qubits to encode the input features
    """
    if qubit_to_sample >= num_qubits:
        raise ValueError("The qubit to sample must be less than the number of qubits")
    qml.AmplitudeEmbedding(features, wires=qubits_to_encode, normalize=True, pad_with=0.0)
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
    if testing: qubits_to_encode = range(4)
    else: qubits_to_encode = qnp.random.choice(num_qubits, size = 4, replace=False).tolist()
    ##Uncomment the line below to see the qubit being sampled
    ##print(f"Sampling for qubit: {qubit_to_sample + 1}")
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
    qubits_to_encode = range(4)
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
optimizer = qml.AdamOptimizer(stepsize=0.001, beta1=0.9, beta2=0.999, eps=1e-8) ## Setting the optimizer to have the same behaviour between torch and pennylane

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
    accuracy = qnp.mean(predictions == labels)
    return accuracy.numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3407)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
X_train_qnp = qnp.array(X_train)
y_train_qnp = qnp.array(y_train)
X_test_qnp = qnp.array(X_test)
y_test_qnp = qnp.array(y_test)
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
MLP = nn.Sequential(
    nn.Linear(9, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
    nn.Softmax(dim=1)
)
optim = optim.Adam(MLP.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()
epochs = 10
for epoch in range(epochs):
    loss_c = 0
    for X_batch, y_batch in train_loader:
        optim.zero_grad()
        outputs = MLP(X_batch)
        loss_val = loss(outputs, y_batch)
        loss_val.backward()
        optim.step()
        loss_c += loss_val.item()
    if (epoch +1) % 10 == 0 or (epoch + 1) == epochs:
        print(f'Epoch {epoch+1}: Loss = {loss_c / len(train_loader)}')
MLP.eval()
with torch.no_grad():
    outputs = MLP(X_test_tensor)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = torch.mean((predictions == y_test_tensor).float())
    print(f"Accuracy of the Neural Network: {accuracy * 100}%")
print(f"Number of parameters in the neural network: {sum(p.numel() for p in MLP.parameters())}")



num_layers = 2
qparams = qnp.random.uniform(0, np.pi, (num_layers,num_qubits, 3))
print(f"Number of parameters in the quantum model: {len(qparams.flatten())}")
qparams = train_quantum_model(X_train_qnp, y_train_qnp, qparams, epochs=2, deterministic = True)
print("Quantum Model Trained, Testing...")
accuracy = test_quantum_model(X_test_qnp, y_test_qnp, qparams)
print(f"Accuracy of the Parametrized Quantum Circuit: {accuracy * 100}%")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
fig, ax = qml.draw_mpl(draw_circuit, style = "black_white", expansion_strategy="device", show_all_wires=True, decimals = 2)(X_train_qnp[0], qparams)
fig.savefig("CircuitTitanic.png", dpi = 600)
plt.show()
