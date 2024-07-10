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
    Returns the expectation value of the Pauli-Z operator on the qubit specified by qubit_to_sample
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
def cost_circuit(features:np.ndarray, params:np.ndarray, testing:bool = False) -> float:
    """
    Executes the quantum circuit and returns the expectation value of the Pauli-Z operator on the first qubit or a random qubit, depending on the testing parameter
    
    Arguments:
    - features (np.ndarray): Input features to the quantum circuit
    - params (np.ndarray): Parameters of the quantum circuit
    - testing (bool): Whether the model is being tested or not (ie if only one qubit should be used)
    """
    if testing: qubit_to_sample = 0
    else: qubit_to_sample = np.random.randint(num_qubits)
    ##Uncomment the line below to see the qubit being sampled
    ##print(f"Sampling for qubit: {qubit_to_sample + 1}")
    return quantum_circuit(features, params, qubit_to_sample)

@qml.qnode(dev)
def draw_circuit(features:np.ndarray, params:np.ndarray) -> float:
    """
    Executes the quantum circuit and returns the expectation value of the Pauli-Z operator on the first qubit, for visualization purposes
    
    Arguments:
    - features (np.ndarray): Input features to the quantum circuit
    - params (np.ndarray): Parameters of the quantum circuit
    """
    qubit_to_sample = 0
    return quantum_circuit(features, params, qubit_to_sample)

## 4. Define the cost function and optimizer
def sigmoid(x:np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    Maps any real value into the range (0, 1), suitable for binary classification.
    """
    return 1 / (1 + np.exp(-x))

def cost_function(params:np.ndarray, features:np.ndarray, labels:np.ndarray, not_random:bool) -> float:
    """
    Binary cross-entropy cost function for classification

    Arguments:
    - params (np.ndarray): Parameters of the quantum circuit
    - features (np.ndarray): Image data to train
    - labels (np.ndarray): Labels corresponding to the image data
    - not_random (bool): Whether to sample the same qubit each time or not
    """
    loss = 0.0
    for f, label in zip(features, labels):
        logits = cost_circuit(f, params, not_random)  # ensure cost_circuit outputs logits
        prediction = sigmoid(logits)       # apply sigmoid to convert logits to probabilities
        loss += -label * np.log(prediction + 1e-9) - (1 - label) * np.log(1 - prediction + 1e-9)  # 1e-9 for numerical stability
    loss = loss / len(features)
    return loss

optimizer = qml.AdamOptimizer(stepsize=0.01, beta1=0.9, beta2=0.999, eps=1e-8) ## Setting the optimizer to have the same behaviour between torch and pennylane

## 5. Define the training and testing functions
def train_quantum_model(data:np.ndarray, labels:np.ndarray, params:np.ndarray, epochs=10, deterministic:bool = False) -> np.ndarray:
    """
    Trains the quantum model using the cost function and optimizer

    Arguments:
    - data (np.ndarray): Image data to train
    - labels (np.ndarray): Labels corresponding to the image data
    - params (np.ndarray): Parameters of the quantum circuit
    - epochs (int): Number of epochs to train the model
    - deterministic (bool): Whether to sample the same qubit each time or not
    """
    features = [img.flatten() for img in data] ## Simple flattening of the image data
    for epoch in range(epochs):
        params, cost = optimizer.step_and_cost(lambda p: cost_function(p, features, labels, deterministic), params)
        print(f"Epoch {epoch+1}: Cost = {cost}")
    return params

def test_quantum_model(data:np.ndarray, labels:np.ndarray, params:np.ndarray, override: bool = False) -> float:
    """
    Tests the quantum model and returns the accuracy

    Arguments:
    - data (np.ndarray): Image data to test
    - labels (np.ndarray): Labels corresponding to the image data
    - params (np.ndarray): Parameters of the quantum circuit
    - override (bool): Whether to override the testing setting and sample random qubits
    """
    features = [img.flatten() for img in data]
    raw_logits = [cost_circuit(f, params, testing = (not override)) for f in features]
    accuracy = np.mean([(np.sign(pred) != lab) for pred, lab in zip(raw_logits, labels)])
    return accuracy


## 6. Define training and testing functions for the classical model
def train_classical_model(model:nn.Module, data:np.ndarray, labels:np.ndarray, epochs:int=10) -> nn.Module:
    """
    Trains the classical model using the mean squared error loss

    Arguments:
    - model (torch.nn.Module): Model to train
    - data (np.ndarray): Image data to train
    - labels (np.ndarray): Labels corresponding to the image data
    - epochs (int): Number of epochs to train the model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas = (0.9, 0.999), eps = 1e-8)
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

def test_classical_model(model: nn.Module, data:np.ndarray, labels:np.ndarray) -> float:
    """
    Tests the classical model and returns the accuracy

    Arguments:
    - model (torch.nn.Module): Trained model
    - data (np.ndarray): Image data to test
    - labels (np.ndarray): Labels corresponding to the image data
    """
    features = torch.tensor([img.flatten().numpy() for img in data], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        predictions = torch.argmax(outputs, dim=1)
    accuracy = torch.mean((predictions == torch.tensor(labels)).float())
    return accuracy

## 7. Generate adversarial examples with Projected Gradient Descent
def generate_pgd_adversarial_example_classical(model:nn.Module, data:np.ndarray, labels:np.ndarray, epsilon:float=0.2, alpha:float=0.01, num_iter:int=30) -> np.ndarray:
    """
    Generate adversarial examples using Projected Gradient Descent for the classical model
    The adversarial examples are generated by perturbing the input features using the following formula:
    x' = x + alpha * sign(grad(loss, x))

    Arguments:
    - model (torch.nn.Module): Trained model
    - data (np.ndarray): Image data to generate adversarial examples
    - labels (np.ndarray): Labels corresponding to the image data
    - epsilon (float): Maximum perturbation allowed (default: 0.2)
    - alpha (float): Step size for the perturbation (default: 0.01)
    - num_iter (int): Number of iterations for the PGD algorithm (default: 30)
    """
    features = torch.tensor([img.flatten().numpy() for img in data], dtype=torch.float32)
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
    return np.array([f.detach().numpy() for f in features])

def generate_pgd_adversarial_example_quantum(params:np.ndarray, data:np.ndarray, labels:np.ndarray, epsilon:float=0.2, alpha:float=0.01, num_iter:int=30) -> np.ndarray:
    """
    Generate adversarial examples using Projected Gradient Descent for the quantum model
    The adversarial examples are generated by perturbing the input features using the following formula:
    x' = x + alpha * sign(grad(loss, x))

    Arguments:
    - params (np.ndarray): Parameters of the quantum circuit
    - data (np.ndarray): Image data to generate adversarial examples
    - labels (np.ndarray): Labels corresponding to the image data
    - epsilon (float): Maximum perturbation allowed (default: 0.2)
    - alpha (float): Step size for the perturbation (default: 0.01)
    - num_iter (int): Number of iterations for the PGD algorithm (default: 30)
    """
    features = torch.tensor([img.flatten().numpy() for img in data], dtype=torch.float32)
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
    return np.array([f.detach().numpy() for f in features])


if __name__ == "__main__":
    ##Note if size of the image is changed, the number of qubits must be changed accordingly above in the script to be 2log2(size)
    train_data, train_labels = generate_data(num_samples=100, size=4, noise=False)
    test_data, test_labels = generate_data(num_samples=100, size=4, noise=True, noise_level=0.3, noise_type="normal")
    visualize_data(train_data, train_labels)
    visualize_data(test_data, test_labels)
    save_image = True
    num_layers = 2
    params = np.random.random((num_layers, num_qubits, 3))
    epochs = 2
    classical_model = nn.Sequential(
        nn.Linear(16, 2),
        nn.Sigmoid()
    )
    deterministic_qubit_sampling = False ## Set to True to check whether the model performs better if only one qubit is sampled

    ## Train without noise and test with noise
    trained_classical_model = train_classical_model(classical_model, train_data, train_labels, epochs=epochs)
    accuracy_classical = test_classical_model(trained_classical_model, test_data, test_labels)
    print(f"Number of parameters in the classical model: {sum(p.numel() for p in trained_classical_model.parameters())}")
    print(f"Accuracy of the classical model: {accuracy_classical*100:.2f}%")
    ## Training the model
    trained_params = train_quantum_model(train_data, train_labels, params, epochs=epochs, deterministic = deterministic_qubit_sampling)
    ## Testing the model
    accuracy = test_quantum_model(test_data, test_labels, trained_params)
    print(f"Number of parameters: {len(trained_params.flatten())}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    ## Visualizing the quantum circuit and saving the image
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, ax = qml.draw_mpl(draw_circuit, style = "black_white", expansion_strategy="device", show_all_wires=True, decimals = 2)(train_data[0].flatten(), trained_params)
    if save_image:
        fig.savefig("CircuitStripes.png", dpi = 600)
    plt.show()

    ## Generating adversarial examples
    perturbed_classical_features = generate_pgd_adversarial_example_classical(trained_classical_model, test_data, test_labels)
    perturbed_quantum_features = generate_pgd_adversarial_example_quantum(trained_params, test_data, test_labels)
    print("Adversarial examples generated successfully")
    ## Displaying the adversarial examples
    visualize_data(perturbed_classical_features.reshape((-1,4,4)), test_labels)
    visualize_data(perturbed_quantum_features.reshape((-1,4,4)), test_labels)

    ## Testing the models with the adversarial examples
    accuracy_classical_perturbed = test_classical_model(trained_classical_model, perturbed_classical_features, test_labels)
    accuracy_quantum_perturbed = test_quantum_model(perturbed_quantum_features, test_labels, trained_params)
    print(f"Accuracy of the classical model with PGD adversarial examples: {accuracy_classical_perturbed*100:.2f}%")
    print(f"Accuracy of the quantum model with PGD adversarial examples: {accuracy_quantum_perturbed*100:.2f}%")
    print(f"Accuracy loss of the classical model: {accuracy_classical*100 - accuracy_classical_perturbed*100:.2f}%\nAccuracy loss of the quantum model: {accuracy*100 - accuracy_quantum_perturbed*100:.2f}%")
    accuracy_quantum_perturbed_random = test_quantum_model(perturbed_quantum_features, test_labels, trained_params, override = True)
    print(f"Accuracy of the quantum model with PGD adversarial examples: {accuracy_quantum_perturbed_random*100:.2f}% (random qubit sampling)")
    print(f"Difference between random qubit sampling and deterministic: {accuracy_quantum_perturbed_random*100 - accuracy_quantum_perturbed*100:.2f}%")
