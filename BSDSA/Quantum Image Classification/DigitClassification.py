from sklearn.datasets import load_digits
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
 
class NeuralPreprocessor(nn.Module):
    def __init__(self):
        super(NeuralPreprocessor, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(4, 1, 3, padding=1) 
        self.cout = None
        self.lin1 = nn.Linear(16, 2) 
        self.sigmoid = nn.Sigmoid()
        self.lin_params = sum(p.numel() for p in self.lin1.parameters())

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)            
        x = F.relu(self.conv2(x))  
        x = x.view(-1, 16)         
        self.cout = x

        x = self.lin1(x)
        return self.sigmoid(x)


digits = load_digits()
X = digits.data
y = digits.target
valid = (y == 0) | (y == 1)
X = X[valid]
y = y[valid]

X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 8, 8)
y_tensor = torch.tensor(y, dtype=torch.long)

model = NeuralPreprocessor()
epochs_train = 50
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
new_X = None
for epoch in range(epochs_train):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    new_X = model.cout.detach().numpy()


print("Convolutional Features Retrieved, Training QML Model...")
num_qubits = int(np.log2(new_X.shape[1]))
dev = qml.device("default.qubit", wires=num_qubits)
def quantum_circuit(features:np.ndarray, params:np.ndarray) -> ExpectationMP:
    """
    Returns the expectation value of the Pauli-Z operator on the first qubit
    """
    qml.AmplitudeEmbedding(features, wires=range(num_qubits), normalize = True)
    qml.StronglyEntanglingLayers(params, wires=range(num_qubits))
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def cost_circuit(features:np.ndarray, params:np.ndarray) -> float:
    """
    Executes the quantum circuit and returns the expectation value of the Pauli-Z operator on the first qubit
    """
    return quantum_circuit(features, params)

## 4. Define the cost function and optimizer
def sigmoid(x):
    """
    Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    Maps any real value into the range (0, 1), suitable for binary classification.
    """
    return 1 / (1 + qnp.exp(-x))

def cost_function(params:np.ndarray, features:np.ndarray, labels:np.ndarray) -> float:
    """
    Binary cross-entropy cost function for classification
    """
    loss = 0.0
    for f, label in zip(features, labels):
        logits = cost_circuit(f, params)  # ensure cost_circuit outputs logits
        prediction = sigmoid(logits)       # apply sigmoid to convert logits to probabilities
        loss += -label * qnp.log(prediction + 1e-9) - (1 - label) * qnp.log(1 - prediction + 1e-9)  # 1e-9 for numerical stability
    loss = loss / len(features)
    return loss

optimizer = qml.GradientDescentOptimizer(stepsize=0.2) ## Gradient descent optimizer

## 5. Define the training and testing functions
def train_quantum_model(data:np.ndarray, labels:np.ndarray, params:np.ndarray, epochs=10) -> np.ndarray:
    """
    Trains the quantum model using the cost function and optimizer
    """
    features = [img.flatten() for img in data] ## Simple flattening of the image data
    features = qnp.array(features)
    for epoch in range(epochs):
        params, cost = optimizer.step_and_cost(lambda p: cost_function(p, features, labels), params)
        print(f"Epoch {epoch+1}: Cost = {cost}")
    return params

def test_quantum_model(data:np.ndarray, labels:np.ndarray, params:np.ndarray) -> float:
    """
    Tests the quantum model and returns the accuracy
    """
    features = [img.flatten() for img in data]
    predictions = [cost_circuit(f, params) for f in features] ##Raw logits predicted by the quantum circuit
    accuracy = np.mean([(np.sign(pred) != lab) for pred, lab in zip(predictions, labels)])
    return accuracy

qdata = qnp.array(new_X)
qlabels = qnp.array(y)
qdata_train, qdata_test, qlabels_train, qlabels_test = train_test_split(qdata, qlabels, test_size=0.2, random_state=62101)
num_layers = 2
qparams = qnp.random.uniform(0, np.pi, (num_layers,num_qubits, 3))
qparams = train_quantum_model(qdata_train, qlabels_train, qparams, epochs=10)
print("Quantum Model Trained, Testing...")
accuracy = test_quantum_model(qdata_test, qlabels_test, qparams)
print(f"Accuracy of the Parametrized Quantum Circuit: {accuracy}")
print(f"Number of parameters in the quantum model: {len(qparams.flatten())}")

new_X_tensor = torch.tensor(new_X, dtype=torch.float32)
X_classic_train, X_classic_test, y_classic_train, y_classic_test = train_test_split(new_X_tensor, y_tensor, test_size=0.2, random_state=62101)
classical_model = nn.Sequential(
    nn.Linear(16, 2),
    nn.Sigmoid()
)
print("Training Classical Model...")
epochs = 10
optimizer = torch.optim.Adam(classical_model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = classical_model(X_classic_train)
    loss = criterion(outputs, y_classic_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item() / len(X_classic_train)}")
model.eval()
print("Testing Classical Model...")
with torch.no_grad():
    outputs = model(X_tensor)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = torch.mean((predictions == y_tensor).float())
    print(f"Accuracy of the Neural Network: {accuracy}")
print(f"Number of parameters in the classical model: {sum(p.numel() for p in classical_model.parameters())}")
