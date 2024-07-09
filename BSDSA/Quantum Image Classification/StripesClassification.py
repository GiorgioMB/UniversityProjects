import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns
from pennylane.measurements import ExpectationMP
import torch.nn as nn
import torch
torch.manual_seed(62101)
np.random.seed(62101)
 
## 1. Generate synthetic binary image data
def generate_data(num_samples: int = 100, size: int = 4, noise: bool = False,  noise_level: float = 0.1, noise_type = "uniform") -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic binary image data with horizontal and vertical stripes
    Arguments:
    - num_samples (int): Number of samples to generate (default: 100)
    - size (int): Size of the square image (default: 4)
    - noise (bool): Whether to add noise to the data (default: False)
    - noise_level (float): Level of noise to add to the data (default: 0.1)
    - noise_type (str): Type of noise to add to the data (default: "uniform") - options: "uniform", "normal"

    Returns:
    - data (np.ndarray): Generated image data
    - labels (np.ndarray): Generated labels

    Note: Number of stripes is size // 2
    """
    if noise_level < 0 or noise_level > 1:
        raise ValueError("Noise level must be between 0 and 1")
    if (not noise) and (noise_level != 0.1):
        print("Warning, noise level was set but noise was not enabled. Data will be generated without noise.")
    labels = np.random.randint(2, size=num_samples)
    data = np.zeros((num_samples, size, size))

    horizontal_indices = np.where(labels == 0)[0]
    data[horizontal_indices, ::2, :] = 1

    vertical_indices = np.where(labels == 1)[0]
    data[vertical_indices, :, ::2] = 1
    
    if noise: ## Add noise to the data
        if noise_type == "uniform":
           noise = np.random.uniform(low = -noise_level, high = noise_level, size = (num_samples, size, size))
        elif noise_type == "normal":
           noise = np.random.randn(num_samples, size, size) * noise_level
        else:
            raise ValueError("Noise type must be either 'uniform' or 'normal'")
        data += noise
    elif noise_level != 0.1:
        print("Warning, noise level was set but noise was not enabled. Data will be generated without noise.")
    return data, labels

## 2. Visualize the generated data
def visualize_data(data: np.ndarray = None, labels: np.ndarray = None, max_samples: int = 10) -> plt.figure:
    """
    Data visualization function
    Arguments:
    - data (np.ndarray): Image data to visualize (default: None)
    - labels (np.ndarray): Labels corresponding to the image data (default: None)
    - max_samples (int): Maximum number of samples to visualize (default: 10)

    Returns:
    - plt.figure: Visualization of the image data

    Note: if data and labels are not provided, they will be generated with size = 4 and num_samples = max_samples
    """
    if isinstance(data, type(None)) or isinstance(labels, type(None)):
        data, labels = generate_data(num_samples = max_samples)
    sns.set_style("white")
    fig, axs = plt.subplots(1, max_samples, figsize = (15, 3))
    for i in range(max_samples):
        axs[i].imshow(data[i], cmap = 'grey')
        axs[i].axis('off')
        axs[i].set_title(f"Label: {labels[i]}")
    plt.show()
    return fig

## 3. Define the quantum circuit
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)
def quantum_circuit(features:np.ndarray, params:np.ndarray, qubit_to_sample: int) -> ExpectationMP:
    """
    Returns the expectation value of the Pauli-Z operator on the first qubit
    - features (np.ndarray): Input features to the quantum circuit
    - params (np.ndarray): Parameters of the quantum circuit
    - qubit_to_sample (int): Qubit to sample from the quantum circuit
    """
    if qubit_to_sample >= num_qubits:
        raise ValueError("The qubit to sample must be less than the number of qubits")
    qml.AmplitudeEmbedding(features, wires=range(num_qubits), normalize=True)
    qml.StronglyEntanglingLayers(params, wires=range(num_qubits))
    return qml.expval(qml.PauliZ(qubit_to_sample))

@qml.qnode(dev)
def cost_circuit(features:np.ndarray, params:np.ndarray) -> float:
    """
    Executes the quantum circuit and returns the expectation value of the Pauli-Z operator on the first qubit
    """
    qubit_to_sample = np.random.randint(num_qubits)
    ##Uncomment the line below to see the qubit being sampled
    ##print(f"Sampling for qubit: {qubit_to_sample + 1}")
    return quantum_circuit(features, params, qubit_to_sample)

## 4. Define the cost function and optimizer
def sigmoid(x):
    """
    Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    Maps any real value into the range (0, 1), suitable for binary classification.
    """
    return 1 / (1 + np.exp(-x))

def cost_function(params:np.ndarray, features:np.ndarray, labels:np.ndarray) -> float:
    """
    Binary cross-entropy cost function for classification
    """
    loss = 0.0
    for f, label in zip(features, labels):
        logits = cost_circuit(f, params)  # ensure cost_circuit outputs logits
        prediction = sigmoid(logits)       # apply sigmoid to convert logits to probabilities
        loss += -label * np.log(prediction + 1e-9) - (1 - label) * np.log(1 - prediction + 1e-9)  # 1e-9 for numerical stability
    loss = loss / len(features)
    return loss

optimizer = qml.GradientDescentOptimizer(stepsize=0.2) ## Gradient descent optimizer

## 5. Define the training and testing functions
def train_quantum_model(data:np.ndarray, labels:np.ndarray, params:np.ndarray, epochs=10) -> np.ndarray:
    """
    Trains the quantum model using the cost function and optimizer
    """
    features = [img.flatten() for img in data] ## Simple flattening of the image data
    for epoch in range(epochs):
        params, cost = optimizer.step_and_cost(lambda p: cost_function(p, features, labels), params)
        print(f"Epoch {epoch+1}: Cost = {cost}")
    return params

def test_quantum_model(data:np.ndarray, labels:np.ndarray, params:np.ndarray) -> float:
    """
    Tests the quantum model and returns the accuracy
    """
    features = [img.flatten() for img in data]
    predictions = [cost_circuit(f, params) for f in features]
    accuracy = np.mean([(np.sign(pred) != lab) for pred, lab in zip(predictions, labels)])
    return accuracy


## 6. Define training and testing functions for the classical model
def train_classical_model(model, data, labels, epochs=10):
    """
    Trains the classical model using the mean squared error loss
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    features = [img.flatten().numpy() for img in data]
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item() / len(features)}")
    return model

def test_classical_model(model, data, labels):
    """
    Tests the classical model and returns the accuracy
    """
    features = torch.tensor([img.flatten().numpy() for img in data], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        predictions = torch.sign(outputs).detach().numpy().flatten()
    accuracy = np.mean([pred == lab for pred, lab in zip(predictions, labels)])
    return accuracy


if __name__ == "__main__":
    ##Note if size of the image is changed, the number of qubits must be changed accordingly above in the script to be 2log2(size)
    train_data, train_labels = generate_data(num_samples=100, size=4, noise=True, noise_level=0.2, noise_type="normal")
    test_data, test_labels = generate_data(num_samples=20, size=4, noise=True, noise_level=0.1, noise_type="normal")
    visualize_data(train_data, train_labels)
    visualize_data(test_data, test_labels)
    save_image = True
    num_layers = 2
    params = np.random.random((num_layers, num_qubits, 3))
    epochs = 10
    classical_model = nn.Sequential(
        nn.Linear(16, 2),
        nn.Sigmoid()
    )
    trained_classical_model = train_classical_model(classical_model, train_data, train_labels, epochs=epochs)
    accuracy_classical = test_classical_model(trained_classical_model, test_data, test_labels)
    print(f"Number of parameters in the classical model: {sum(p.numel() for p in trained_classical_model.parameters())}")
    print(f"Accuracy of the classical model: {accuracy_classical*100:.2f}%")
    ## Training the model
    trained_params = train_quantum_model(train_data, train_labels, params, epochs=epochs)
    ## Testing the model
    accuracy = test_quantum_model(test_data, test_labels, trained_params)
    print(f"Number of parameters: {len(trained_params.flatten())}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    ## Visualizing the quantum circuit and saving the image
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, ax = qml.draw_mpl(cost_circuit, style = "black_white", expansion_strategy="device", show_all_wires=True, decimals = 2)(train_data[0].flatten(), trained_params)
    if save_image:
        fig.savefig("Circuit Stripes.png", dpi = 600)
    plt.show()
