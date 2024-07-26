import pennylane as qml
from pennylane import numpy as qnp
import numpy as np
import os
from pennylane.measurements import ExpectationMP
import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
from torch.nn import init
import kaggle
os.makedirs('data', exist_ok=True)
kaggle.api.authenticate()
kaggle.api.dataset_download_files('andrewmvd/animal-faces', path='data/', unzip=True)
torch.manual_seed(3407) ## This seed was seen as best for torch models in "Torch.manual_seed(3407) is all you need", Picard, 2021
np.random.seed(3407)
qnp.random.seed(3407)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dir = 'data/afhq/train'
classes = os.listdir(train_dir)
X = []
y = []
X_ViT = []
print("Beginning Data Loading...")
for i, c in enumerate(classes):
    for img in os.listdir(f'{train_dir}/{c}'):
        X.append(np.array(Image.open(f'{train_dir}/{c}/{img}').convert('L').resize((128, 128))))
        X_ViT.append(np.array(Image.open(f'{train_dir}/{c}/{img}').convert('L').resize((224, 224))))
        y.append(i)
X_train = np.array(X)
X_train_ViT = np.array(X_ViT)
y_train = np.array(y)
X_train = np.array(X)
X_train_ViT = np.array(X_ViT)
y_train = np.array(y)
##Shuffle the data
shuffle = np.random.permutation(len(X_train))
X_train = X_train[shuffle]
y_train = y_train[shuffle]
X_train_ViT = X_train_ViT[shuffle]
print("Data Loaded, Shape of X and y:")
print(X_train.shape, y_train.shape)
##display one image per class
for i in range(3):
    plt.imshow(X_train[y_train == i][0], cmap='gray')
    plt.title(classes[i])
    plt.show()
test_dir = 'data/afhq/val'
X_test = []
X_test_ViT = []
y_test = []
print("Beginning Test Data Loading...")
for i, c in enumerate(classes):
    for img in os.listdir(f'{test_dir}/{c}'):
        X_test.append(np.array(Image.open(f'{test_dir}/{c}/{img}').convert('L').resize((128, 128))))
        X_test_ViT.append(np.array(Image.open(f'{test_dir}/{c}/{img}').convert('L').resize((224, 224))))
        y_test.append(i)

X_test = np.array(X_test)
y_test = np.array(y_test)
X_test_ViT = np.array(X_test_ViT)
shuffle = np.random.permutation(len(X_test))
X_test = X_test[shuffle]
y_test = y_test[shuffle]
X_test_ViT = X_test_ViT[shuffle]
print("Test Data Loaded, Shape of X_test and y_test:")
print(X_test.shape, y_test.shape)
for i in range(3):
    plt.imshow(X_test[y_test == i][0], cmap='gray')
    plt.title(classes[i])
    plt.show()

num_qubits = 2 * int(np.log2(128))
dev = qml.device("default.qubit", wires=num_qubits)
@qml.qnode(dev, interface='torch')
def quantum_circuit(features, params):
    qml.AmplitudeEmbedding(features, wires=range(num_qubits), normalize = True)
    qml.StronglyEntanglingLayers(params, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

class QMLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_qlayers, num_lin_layers):
        super(QMLP, self).__init__()
        if num_lin_layers < 1:
            raise ValueError("Number of Linear Layers must be at least 1")
        if num_qlayers < 1:
            raise ValueError("Number of Quantum Layers must be at least 1")
        elif num_lin_layers == 1:
            print("Warning, as there are no hidden layers, hidden dimension is ignored")

        self.quantum_params = nn.Parameter(torch.Tensor(num_qlayers, num_qubits, 3))
        init.xavier_normal_(self.quantum_params) ##This was found to be best by "Alleviating Barren Plateaus in Parameterized Quantum Machine Learning Circuits" Kashif et al (2023)
        self.lin_layers = nn.ModuleList()
        if num_lin_layers > 1:
            self.lin_layers.append(nn.Linear(num_qubits, hidden_dim))
            for i in range(num_lin_layers - 2):
                self.lin_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.head = nn.Linear(hidden_dim, output_dim)
        else:
            self.head = nn.Linear(num_qubits, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        qc_output = quantum_circuit(x, self.quantum_params)
        qc_output = torch.stack(qc_output).float().permute(1, 0)
        x = qc_output
        for layer in self.lin_layers:
            x = self.relu(layer(x))
        x = self.head(x)
        return self.softmax(x)

def train_torch_model(model, X, y, epochs, optimizer, loss, batch_size=32):
    if type(X) != torch.Tensor:
        X = torch.tensor(X, dtype=torch.float32).to(device)
    if type(y) != torch.Tensor:
        y = torch.tensor(y, dtype=torch.long).to(device)
    model.train()
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in tqdm(dataloader, desc="Processing Batches", total=len(dataloader)):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss_val = loss(outputs, y_batch)
            running_loss += loss_val.item()
            loss_val.backward()
            optimizer.step()
        running_loss /= len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss}")
    return model

def test_torch_model(model, X, y, batch_size=32):
    if type(X) != torch.Tensor:
        X = torch.tensor(X, dtype=torch.float32).to(device)
    if type(y) != torch.Tensor:
        y = torch.tensor(y, dtype=torch.long).to(device)
    model.eval()
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for X_batch, y_batch in tqdm(dataloader, desc="Processing Batches", total=len(dataloader)):
            outputs = model(X_batch)
            predicted = torch.argmax(outputs, dim=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy}%")
    return accuracy

def generate_pgd_adversarial_data(model, X, y, epsilon=0.1, alpha=0.01, iters=10):
    if type(X) != torch.Tensor:
        X = torch.tensor(X, dtype=torch.float32).to(device)
    if type(y) != torch.Tensor:
        y = torch.tensor(y, dtype=torch.long).to(device)
    X_adv = X.clone().detach().requires_grad_(True)
    for i in range(iters):
        outputs = model(X_adv)
        loss = criterion(outputs, y)
        loss.backward()
        grad = X_adv.grad.data
        X_adv = X_adv + alpha * grad.sign()
        X_adv = torch.max(torch.min(X_adv, X + epsilon), X - epsilon).clamp(0, 1)
        X_adv = X_adv.detach().requires_grad_(True)
    return X_adv


MLP = nn.Sequential(
    nn.Linear(128**2, 8192),
    nn.ReLU(),
    nn.Linear(8192, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 3)
).to(device)
CNN = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(32*32*32, 128),
    nn.ReLU(),
    nn.Linear(128, 3)
).to(device)
CVT = timm.create_model('convformer_b36', pretrained=True)
first_conv_layer = CVT.stem.conv
new_first_conv = nn.Conv2d(1, first_conv_layer.out_channels, 
                           kernel_size=first_conv_layer.kernel_size, 
                           stride=first_conv_layer.stride, 
                           padding=first_conv_layer.padding, 
                           bias=(first_conv_layer.bias is not None))
new_first_conv.weight.data = first_conv_layer.weight.data.mean(dim=1, keepdim=True)
CVT.stem.conv = new_first_conv
CVT.head.fc.fc2 = nn.Linear(in_features=3072, out_features=3, bias=True)
CVT.to(device)
ViT = timm.create_model('vit_base_patch16_224', pretrained=True)
first_conv_layer = ViT.patch_embed.proj
new_first_conv = nn.Conv2d(1, first_conv_layer.out_channels, 
                           kernel_size=first_conv_layer.kernel_size, 
                           stride=first_conv_layer.stride, 
                           padding=first_conv_layer.padding, 
                           bias=(first_conv_layer.bias is not None))
new_first_conv.weight.data = first_conv_layer.weight.data.mean(dim=1, keepdim=True)
ViT.patch_embed.proj = new_first_conv
ViT.head = nn.Linear(ViT.head.in_features, 3)
ViT.to(device)
Hybrid_PQC = QMLP(128, 3, 6, 4).to(device)
##clone the starting parameters
starting_qparams = Hybrid_PQC.quantum_params.clone()
##Print the number of parameters in each model
print(f"Number of Parameters in MLP: {sum(p.numel() for p in MLP.parameters())}")
print(f"Number of Parameters in CNN: {sum(p.numel() for p in CNN.parameters())}")
print(f"Number of Parameters in CVT: {sum(p.numel() for p in CVT.parameters())}")
print(f"Number of Parameters in ViT: {sum(p.numel() for p in ViT.parameters())}")
print(f"Number of Parameters in QMLP PQC: {sum(p.numel() for p in Hybrid_PQC.parameters())}")
MLP_optimizer = torch.optim.Adam(MLP.parameters(), lr=0.01)
CNN_optimizer = torch.optim.Adam(CNN.parameters(), lr=0.001)
CVT_optimizer = torch.optim.Adam(CVT.parameters(), lr=1e-5)
ViT_optimizer = torch.optim.Adam(ViT.parameters(), lr=1e-5)
Hybrid_optimizer = torch.optim.NAdam(Hybrid_PQC.parameters(), lr=0.0008)
X_train_vision = X_train.reshape(-1, 1, 128, 128)
X_test_vision = X_test.reshape(-1, 1, 128, 128)
X_train_flat = X_train.reshape(-1, 128**2)
X_test_flat = X_test.reshape(-1, 128**2)
X_train_ViT = X_train_ViT.reshape(-1, 1, 224, 224)
X_test_ViT = X_test_ViT.reshape(-1, 1, 224, 224)
criterion = nn.CrossEntropyLoss()
print("Training Hybrid Model...")
Hybrid_PQC = train_torch_model(Hybrid_PQC, X_train_flat, y_train, 20, Hybrid_optimizer, criterion)
print("Testing Hybrid Model...")
test_torch_model(Hybrid_PQC, X_test_flat, y_test)
assert not torch.allclose(Hybrid_PQC.quantum_params, starting_qparams)
print("---------------------------------")
print("Training CNN Model...")
CNN = train_torch_model(CNN, X_train_vision, y_train, 10, CNN_optimizer, criterion)
print("Testing CNN Model...")
test_torch_model(CNN, X_test_vision, y_test)
print("---------------------------------")
print("Training ViT Model...")
ViT = train_torch_model(ViT, X_train_ViT, y_train, 10, ViT_optimizer, criterion)
print("Testing ViT Model...")
test_torch_model(ViT, X_test_ViT, y_test)
print("---------------------------------")
print("Training CVT Model...")
CVT = train_torch_model(CVT, X_train_vision, y_train, 10, CVT_optimizer, criterion)
print("Testing CVT Model...")
test_torch_model(CVT, X_test_vision, y_test)
print("---------------------------------")
print("Training MLP Model...")
MLP = train_torch_model(MLP, X_train_flat, y_train, 10, MLP_optimizer, criterion)
print("Testing MLP Model...")
test_torch_model(MLP, X_test_flat, y_test)
print("---------------------------------")
print("Generating Adversarial Data...")
X_adv_PQC = generate_pgd_adversarial_data(Hybrid_PQC, X_test_flat, y_test)
X_adv_CNN = generate_pgd_adversarial_data(CNN, X_test_vision, y_test)
X_adv_ViT = generate_pgd_adversarial_data(ViT, X_test_ViT, y_test)
X_adv_CVT = generate_pgd_adversarial_data(CVT, X_test_vision, y_test)
X_adv_MLP = generate_pgd_adversarial_data(MLP, X_test_flat, y_test)
print("Displaying Adversarial Data...")
plt.figure(figsize=(20, 20))
for i in range(5):
    plt.subplot(5, 2, 2*i+1)
    plt.imshow(X_test[y_test == i][0], cmap='gray')
    plt.title(classes[i])
    plt.subplot(5, 2, 2*i+2)
    plt.imshow(X_adv_PQC[y_test == i][0].cpu().numpy().reshape(128, 128), cmap='gray')
    plt.title("Adversarial")
plt.show()
print("Testing models on Adversarial Data...")
print("Hybrid Model:")
test_torch_model(Hybrid_PQC, X_adv_PQC, y_test)
print("---------------------------------")
print("CNN Model:")
test_torch_model(CNN, X_adv_CNN, y_test)
print("---------------------------------")
print("ViT Model:")
test_torch_model(ViT, X_adv_ViT, y_test)
print("---------------------------------")
print("CVT Model:")
test_torch_model(CVT, X_adv_CVT, y_test)
print("---------------------------------")
print("MLP Model:")
test_torch_model(MLP, X_adv_MLP, y_test)
print("---------------------------------")
