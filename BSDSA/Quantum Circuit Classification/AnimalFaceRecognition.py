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
import kaggle
import timm
from tqdm import tqdm
torch.manual_seed(3407) ## This seed was seen as best for torch models in "Torch.manual_seed(3407) is all you need", Picard, 2021
np.random.seed(3407)
qnp.random.seed(3407)
kaggle.api.authenticate()
os.makedirs('data', exist_ok=True)
kaggle.api.dataset_download_files('andrewmvd/animal-faces', path='data/', unzip=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dir = 'data/afhq/train'
classes = os.listdir(train_dir)
X = []
y = []
print("Beginning Data Loading...")
for i, c in enumerate(classes):
    for img in os.listdir(f'{train_dir}/{c}'):
        X.append(np.array(Image.open(f'{train_dir}/{c}/{img}').convert('L').resize((128, 128))))
        y.append(i)
X_train = np.array(X)
y_train = np.array(y)
##Shuffle the data
shuffle = np.random.permutation(len(X_train))
X_train = X_train[shuffle]
y_train = y_train[shuffle]
print("Data Loaded, Shape of X and y:")
print(X.shape, y.shape)
##display one image per class
for i in range(3):
    plt.imshow(X_train[y_train == i][0], cmap='gray')
    plt.title(classes[i])
    plt.show()

test_dir = 'data/afhq/val'
X_test = []
y_test = []
print("Beginning Test Data Loading...")
for i, c in enumerate(classes):
    for img in os.listdir(f'{test_dir}/{c}'):
        X_test.append(np.array(Image.open(f'{test_dir}/{c}/{img}').convert('L').resize((128, 128))))
        y_test.append(i)
X_test = np.array(X_test)
y_test = np.array(y_test)
shuffle = np.random.permutation(len(X_test))
X_test = X_test[shuffle]
y_test = y_test[shuffle]
print("Test Data Loaded, Shape of X_test and y_test:")
print(X_test.shape, y_test.shape)
for i in range(3):
    plt.imshow(X_test[y_test == i][0], cmap='gray')
    plt.title(classes[i])
    plt.show()

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
new_first_conv
CVT.stem.conv = new_first_conv
CVT.head.fc.fc2 = nn.Linear(in_features=3072, out_features=3, bias=True)
CVT.to(device)
ViT = timm.create_model('vit_base_patch16_224', pretrained=True)
ViT.head = nn.Linear(ViT.head.in_features, 3)
ViT.to(device)
MLP_optimizer = torch.optim.Adam(MLP.parameters(), lr=0.01)
CNN_optimizer = torch.optim.Adam(CNN.parameters(), lr=0.001)
CVT_optimizer = torch.optim.Adam(CVT.parameters(), lr=1e-5)
ViT_optimizer = torch.optim.Adam(ViT.parameters(), lr=1e-5)
X_train_vision = X_train.reshape(-1, 1, 128, 128)
X_test_vision = X_test.reshape(-1, 1, 128, 128)
X_train_flat = X_train.reshape(-1, 128**2)
criterion = nn.CrossEntropyLoss()

print("Training CNN Model...")
CNN = train_torch_model(CNN, X_train_vision, y_train, 10, CNN_optimizer, criterion)
print("Testing CNN Model...")
test_torch_model(CNN, X_test_vision, y_test)
print("---------------------------------")
print("Training CVT Model...")
CVT = train_torch_model(CVT, X_train_vision, y_train, 10, CVT_optimizer, criterion)
print("Testing CVT Model...")
test_torch_model(CVT, X_test_vision, y_test)
print("---------------------------------")
print("Training ViT Model...")
ViT = train_torch_model(ViT, X_train_vision, y_train, 10, ViT_optimizer, criterion)
print("Testing ViT Model...")
test_torch_model(ViT, X_test_vision, y_test)
print("---------------------------------")
print("Training MLP Model...")
MLP = train_torch_model(MLP, X_train_flat, y_train, 10, MLP_optimizer, criterion)
print("Testing MLP Model...")
test_torch_model(MLP, X_test.reshape(-1, 128**2), y_test)#%%
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
import kaggle
import timm
from tqdm import tqdm
torch.manual_seed(3407) ## This seed was seen as best for torch models in "Torch.manual_seed(3407) is all you need", Picard, 2021
np.random.seed(3407)
qnp.random.seed(3407)
kaggle.api.authenticate()
os.makedirs('data', exist_ok=True)
kaggle.api.dataset_download_files('andrewmvd/animal-faces', path='data/', unzip=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dir = 'data/afhq/train'
classes = os.listdir(train_dir)
X = []
y = []
print("Beginning Data Loading...")
for i, c in enumerate(classes):
    for img in os.listdir(f'{train_dir}/{c}'):
        X.append(np.array(Image.open(f'{train_dir}/{c}/{img}').convert('L').resize((128, 128))))
        y.append(i)
X_train = np.array(X)
y_train = np.array(y)
##Shuffle the data
shuffle = np.random.permutation(len(X_train))
X_train = X_train[shuffle]
y_train = y_train[shuffle]
print("Data Loaded, Shape of X and y:")
print(X.shape, y.shape)
##display one image per class
for i in range(3):
    plt.imshow(X_train[y_train == i][0], cmap='gray')
    plt.title(classes[i])
    plt.show()

test_dir = 'data/afhq/val'
X_test = []
y_test = []
print("Beginning Test Data Loading...")
for i, c in enumerate(classes):
    for img in os.listdir(f'{test_dir}/{c}'):
        X_test.append(np.array(Image.open(f'{test_dir}/{c}/{img}').convert('L').resize((128, 128))))
        y_test.append(i)
X_test = np.array(X_test)
y_test = np.array(y_test)
shuffle = np.random.permutation(len(X_test))
X_test = X_test[shuffle]
y_test = y_test[shuffle]
print("Test Data Loaded, Shape of X_test and y_test:")
print(X_test.shape, y_test.shape)
for i in range(3):
    plt.imshow(X_test[y_test == i][0], cmap='gray')
    plt.title(classes[i])
    plt.show()

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
new_first_conv
CVT.stem.conv = new_first_conv
CVT.head.fc.fc2 = nn.Linear(in_features=3072, out_features=3, bias=True)
CVT.to(device)
ViT = timm.create_model('vit_base_patch16_224', pretrained=True)
ViT.head = nn.Linear(ViT.head.in_features, 3)
ViT.to(device)
MLP_optimizer = torch.optim.Adam(MLP.parameters(), lr=0.01)
CNN_optimizer = torch.optim.Adam(CNN.parameters(), lr=0.001)
CVT_optimizer = torch.optim.Adam(CVT.parameters(), lr=1e-5)
ViT_optimizer = torch.optim.Adam(ViT.parameters(), lr=1e-5)
X_train_vision = X_train.reshape(-1, 1, 128, 128)
X_test_vision = X_test.reshape(-1, 1, 128, 128)
X_train_flat = X_train.reshape(-1, 128**2)
criterion = nn.CrossEntropyLoss()

print("Training CNN Model...")
CNN = train_torch_model(CNN, X_train_vision, y_train, 10, CNN_optimizer, criterion)
print("Testing CNN Model...")
test_torch_model(CNN, X_test_vision, y_test)
print("---------------------------------")
print("Training CVT Model...")
CVT = train_torch_model(CVT, X_train_vision, y_train, 10, CVT_optimizer, criterion)
print("Testing CVT Model...")
test_torch_model(CVT, X_test_vision, y_test)
print("---------------------------------")
print("Training ViT Model...")
ViT = train_torch_model(ViT, X_train_vision, y_train, 10, ViT_optimizer, criterion)
print("Testing ViT Model...")
test_torch_model(ViT, X_test_vision, y_test)
print("---------------------------------")
print("Training MLP Model...")
MLP = train_torch_model(MLP, X_train_flat, y_train, 10, MLP_optimizer, criterion)
print("Testing MLP Model...")
test_torch_model(MLP, X_test.reshape(-1, 128**2), y_test)
