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
def quantum_circuit(features:qnp.ndarray, params:qnp.ndarray, qubit_to_sample: int) -> ExpectationMP:
    """
    Returns the expectation value of the Pauli-Z operator on the qubit specified by qubit_to_sample
    - features (qnp.ndarray): Input features to the quantum circuit
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
    Executes the quantum circuit and returns the expectation value of the Pauli-Z operator on either a random qubit or the first qubit, depending on the testing parameter
    
    Arguments:
    - features (qnp.ndarray): Input features to the quantum circuit
    - params (qnp.ndarray): Parameters of the quantum circuit
    - testing (bool): Whether the model is being tested or not (ie if only one qubit should be used)
    - verbose (bool): Whether to print the qubit being sampled (for debugging purposes
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
    - features (qnp.ndarray): Input features to the quantum circuit
    - params (qnp.ndarray): Parameters of the quantum circuit
    """
    qubit_to_sample = 0
    return quantum_circuit(features, params, qubit_to_sample)

def sigmoid(x:qnp.ndarray) -> np.ndarray:
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

optimizer = qml.AdamOptimizer(stepsize=0.01, beta1=0.9, beta2=0.999, eps=1e-8) ## Setting the optimizer to have the same behaviour between torch and pennylane

def train_quantum_model(data:qnp.ndarray, labels:qnp.ndarray, params:qnp.ndarray, epochs=10, deterministic:bool = False) -> np.ndarray:
    """
    Trains the quantum model using the cost function and optimizer

    Arguments:
    - data (qnp.ndarray): Image data to train
    - labels (qnp.ndarray): Labels corresponding to the image data
    - params (qnp.ndarray): Parameters of the quantum circuit
    - epochs (int): Number of epochs to train the model
    - deterministic (bool): Whether to sample the same qubit each time or not
    """
    features = [img.flatten() for img in data] ## Simple flattening of the image data
    for epoch in range(epochs):
        params, cost = optimizer.step_and_cost(lambda p: cost_function(p, features, labels, deterministic), params)
        print(f"Epoch {epoch+1}: Cost = {cost}")
    return params

def test_quantum_model(data:qnp.ndarray, labels:qnp.ndarray, params:qnp.ndarray, override: bool = False) -> float:
    """
    Tests the quantum model and returns the accuracy

    Arguments:
    - data (qnp.ndarray): Image data to test
    - labels (qnp.ndarray): Labels corresponding to the image data
    - params (qnp.ndarray): Parameters of the quantum circuit
    - override (bool): Whether to override the testing setting and sample random qubits
    """
    features = [img.flatten() for img in data]
    raw_logits = [cost_circuit(f, params, testing = (not override)) for f in features]
    accuracy = np.mean([(np.sign(pred) != lab) for pred, lab in zip(raw_logits, labels)])
    return accuracy

qdata = qnp.array(new_X)
qlabels = qnp.array(y)
qdata_train, qdata_test, qlabels_train, qlabels_test = train_test_split(qdata, qlabels, test_size=0.2, random_state=62101)
num_layers = 2
epochs = 2
qparams = qnp.random.uniform(0, np.pi, (num_layers,num_qubits, 3))
qparams = train_quantum_model(qdata_train, qlabels_train, qparams, epochs=epochs)
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

##Visualizing the quantum circuit
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
fig, ax = qml.draw_mpl(draw_circuit, style = "black_white", expansion_strategy="device", show_all_wires=True, decimals = 2)(qdata[0], qparams)
fig.savefig("CircuitDigits.png", dpi = 600)
plt.show()
