import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from regex import D
import seaborn as sns
from pennylane.measurements import ExpectationMP
import torch.nn as nn
import torch
from einops.layers.torch import Rearrange

torch.manual_seed(62101)
np.random.seed(62101)
## 0. Define Vision Transformer model for image classification
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, emb_size=64, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size*patch_size*in_channels, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class ViT(nn.Module):
    def __init__(self, image_size=4, patch_size=4, num_classes=2, channels=1, emb_size=64, depth=3, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(channels, patch_size, emb_size, dropout)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.randn((image_size // patch_size) ** 2 + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])

        return self.mlp_head(x)
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
def visualize_data(data: np.ndarray = None, labels: np.ndarray = None, max_samples: int = 10, title:str = None) -> plt.figure:
    """
    Data visualization function
    Arguments:
    - data (np.ndarray): Image data to visualize (default: None)
    - labels (np.ndarray): Labels corresponding to the image data (default: None)
    - max_samples (int): Maximum number of samples to visualize (default: 10)
    - title (str): Title of the visualization (default: None)
    Returns:
    - plt.figure: Visualization of the image data

    Note: if data and labels are not provided, they will be generated with size = 4 and num_samples = max_samples
    """
    if isinstance(data, type(None)) or isinstance(labels, type(None)):
        data, labels = generate_data(num_samples = max_samples)
    sns.set_style("white")
    fig, axs = plt.subplots(1, max_samples, figsize = (15, 3))
    if title is not None:
        fig.suptitle(title)
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
def train_classical_model(model:nn.Module, data:np.ndarray, labels:np.ndarray, epochs:int=10, flatten:bool = True, batching: bool = False, batch_size:int = 64) -> nn.Module:
    """
    Trains the classical model using the mean squared error loss

    Arguments:
    - model (torch.nn.Module): Model to train
    - data (np.ndarray): Image data to train
    - labels (np.ndarray): Labels corresponding to the image data
    - epochs (int): Number of epochs to train the model
    - flatten (bool): Whether to flatten the image data or not
    - batching (bool): Whether to use batching or not
    - batch_size (int): Size of the batch to use
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas = (0.9, 0.999), eps = 1e-8)
    if flatten:
        features = [img.flatten().numpy() for img in data]
    else:
        features = [img.numpy() for img in data]
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    model.train()
    if batching:
        dataset = torch.utils.data.TensorDataset(features, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss = {loss.item() / len(batch_features)}")
            elif (epoch + 1) == epochs:
                print(f"Epoch {epoch+1}: Loss = {loss.item() / len(batch_features)}")
    else:
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss = {loss.item() / len(features)}")
            elif (epoch + 1) == epochs:
                print(f"Epoch {epoch+1}: Loss = {loss.item() / len(features)}")
    return model

def test_classical_model(model: nn.Module, data:np.ndarray, labels:np.ndarray, flatten:bool = True, batching: bool = False, batch_size:int = 64) -> float:
    """
    Tests the classical model and returns the accuracy

    Arguments:
    - model (torch.nn.Module): Trained model
    - data (np.ndarray): Image data to test
    - labels (np.ndarray): Labels corresponding to the image data
    - flatten (bool): Whether to flatten the image data or not
    - batching (bool): Whether to use batching or not
    - batch_size (int): Size of the batch to use
    """
    if flatten:
        features = torch.tensor([img.flatten().numpy() for img in data], dtype=torch.float32)
    else:
        features = torch.tensor([img.numpy() for img in data], dtype=torch.float32)
    model.eval()
    if batching:
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(features, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        predictions = []
        for batch_features, batch_labels in dataloader:
            with torch.no_grad():
                outputs = model(batch_features)
                predictions.extend(torch.argmax(outputs, dim=1))
        predictions = torch.tensor(predictions)

    else:
        with torch.no_grad():
            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)
    accuracy = torch.mean((predictions == torch.tensor(labels)).float())
    return accuracy

## 7. Generate adversarial examples with Projected Gradient Descent
def generate_pgd_adversarial_example_classical(model:nn.Module, data:np.ndarray, labels:np.ndarray, epsilon:float=0.4, alpha:float=0.01, num_iter:int=100, flatten:bool = True) -> np.ndarray:
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
    if flatten:
        features = torch.tensor([img.flatten().numpy() for img in data], dtype=torch.float32)
    else:
        features = torch.tensor([img.numpy() for img in data], dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    original_features = features.clone()
    features.requires_grad = True
    optimizer = torch.optim.Adam([features], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    for i in range(num_iter):
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

def generate_pgd_adversarial_example_quantum(params:np.ndarray, data:np.ndarray, labels:np.ndarray, epsilon:float=0.4, alpha:float=0.01, num_iter:int=100) -> np.ndarray:
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
    optimizer = torch.optim.Adam([features], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    for i in range(num_iter):
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
    train_data, train_labels = generate_data(num_samples=10000, size=4, noise=False)
    test_data, test_labels = generate_data(num_samples=100, size=4, noise=True, noise_level=0.5, noise_type="normal")
    visualize_data(train_data, train_labels)
    visualize_data(test_data, test_labels)
    save_image = True
    num_layers = 2
    params = np.random.random((num_layers, num_qubits, 3))
    epochs = 1
    MLPclassifier = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.Sigmoid()
    )
    CNNClassifier = nn.Sequential(
        nn.Conv2d(1, 6, 2),
        nn.ReLU(),
        nn.Conv2d(6, 16, 2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.Sigmoid()
    )
    AttentionClassifier = ViT()
    deterministic_qubit_sampling = False ## Set to True to check whether the model performs better if only one qubit is sampled

    ## Train without noise and test with noise
    ## Classical models
    trained_MLP = train_classical_model(MLPclassifier, train_data, train_labels, epochs=epochs * 5)
    accuracy_MLP = test_classical_model(trained_MLP, test_data, test_labels)
    print(f"Number of parameters in the MLP model: {sum(p.numel() for p in trained_MLP.parameters())}")
    print(f"Accuracy of the MLP model (x5 epochs): {accuracy_MLP*100:.2f}%")
    print("----------------------------")
    trained_CNN = train_classical_model(CNNClassifier, train_data.reshape(-1, 1, 4, 4), train_labels, epochs=epochs * 5, flatten=False)
    accuracy_CNN = test_classical_model(trained_CNN, test_data.reshape(-1, 1, 4, 4), test_labels, flatten=False)
    print(f"Number of parameters in the CNN model: {sum(p.numel() for p in trained_CNN.parameters())}")
    print(f"Accuracy of the CNN model (x5 epochs): {accuracy_CNN*100:.2f}%")
    print("----------------------------")
    trained_Attention = train_classical_model(AttentionClassifier, train_data.reshape(-1, 1, 4, 4), train_labels, epochs=epochs * 5, flatten=False, batching = True)
    accuracy_Attention = test_classical_model(trained_Attention, test_data.reshape(-1, 1, 4, 4), test_labels, flatten=False)
    print(f"Number of parameters in the Attention model: {sum(p.numel() for p in trained_Attention.parameters())}")
    print(f"Accuracy of the Attention model: {accuracy_Attention*100:.2f}%")

    ## Quantum model
    print("----------------------------")
    trained_params = train_quantum_model(train_data, train_labels, params, epochs=epochs, deterministic = deterministic_qubit_sampling)
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
    low_noise_data, low_noise_labels = generate_data(num_samples=100, size=4, noise=True, noise_level=0.1, noise_type="normal")
    perturbed_MLP_features = generate_pgd_adversarial_example_classical(trained_MLP, low_noise_data, low_noise_labels)
    perturbed_CNN_features = generate_pgd_adversarial_example_classical(trained_CNN, low_noise_data.reshape(-1, 1, 4, 4), low_noise_labels, flatten=False)
    perturbed_Attention_features = generate_pgd_adversarial_example_classical(trained_Attention, low_noise_data.reshape(-1, 1, 4, 4), low_noise_labels, flatten=False)
    perturbed_quantum_features = generate_pgd_adversarial_example_quantum(trained_params, low_noise_data, low_noise_labels)
    print("Adversarial examples generated successfully")
    ## Displaying the adversarial examples
    visualize_data(perturbed_MLP_features.reshape(-1,4,4), low_noise_labels, title="Adversarial Examples for the MLP Model")
    visualize_data(perturbed_CNN_features.reshape(-1,4,4), low_noise_labels, title="Adversarial Examples for the CNN Model")
    visualize_data(perturbed_quantum_features.reshape(-1,4,4), low_noise_labels, title="Adversarial Examples for the Quantum Model")
    visualize_data(perturbed_Attention_features.reshape(-1,4,4), low_noise_labels, title="Adversarial Examples for the Attention Model")
    ## Testing the models with the adversarial examples
    accuracy_MLP_perturbed = test_classical_model(trained_MLP, perturbed_MLP_features, low_noise_labels)
    accuracy_CNN_perturbed = test_classical_model(trained_CNN, perturbed_CNN_features, low_noise_labels, flatten=False)
    accuracy_Attention_perturbed = test_classical_model(trained_Attention, perturbed_Attention_features, low_noise_labels, flatten=False)
    accuracy_quantum_perturbed = test_quantum_model(perturbed_quantum_features, low_noise_labels, trained_params)
    print(f"Accuracy of the MLP model with PGD adversarial examples: {accuracy_MLP_perturbed*100:.2f}%")
    print(f"Accuracy of the CNN model with PGD adversarial examples: {accuracy_CNN_perturbed*100:.2f}%")
    print(f"Accuracy of the Attention model with PGD adversarial examples: {accuracy_Attention_perturbed*100:.2f}%")
    print(f"Accuracy of the quantum model with PGD adversarial examples: {accuracy_quantum_perturbed*100:.2f}%")
    print("----------------------------")
    print(f"Accuracy loss of the MLP model: {abs(accuracy_MLP*100 - accuracy_MLP_perturbed*100):.2f}%")
    print(f"Accuracy loss of the CNN model: {abs(accuracy_CNN*100 - accuracy_CNN_perturbed*100):.2f}%")
    print(f"Accuracy loss of the Attention model: {abs(accuracy_Attention*100 - accuracy_Attention_perturbed*100):.2f}%")
    print(f"Accuracy loss of the quantum model: {abs(accuracy*100 - accuracy_quantum_perturbed*100):.2f}%")
    print(f"Is the quantum model best? {accuracy_quantum_perturbed > accuracy_MLP_perturbed and accuracy_quantum_perturbed > accuracy_CNN_perturbed and accuracy_quantum_perturbed > accuracy_Attention_perturbed}")
    accuracy_quantum_perturbed_random = test_quantum_model(perturbed_quantum_features, low_noise_labels, trained_params, override = True)
    print("----------------------------")
    print(f"Accuracy of the quantum model with PGD adversarial examples: {accuracy_quantum_perturbed_random*100:.2f}% (random qubit sampling)")
    print(f"Difference between random qubit sampling and deterministic: {abs(accuracy_quantum_perturbed_random*100 - accuracy_quantum_perturbed*100):.2f}%\nIs random qubit sampling better? {accuracy_quantum_perturbed_random > accuracy_quantum_perturbed}")
    print(f"Is random qubit sampling better than the classical models? {accuracy_quantum_perturbed_random > accuracy_MLP_perturbed and accuracy_quantum_perturbed_random > accuracy_CNN_perturbed and accuracy_quantum_perturbed_random > accuracy_Attention_perturbed}")
