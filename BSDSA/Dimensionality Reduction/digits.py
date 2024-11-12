from encoder import *

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn.functional as F
from torch import dropout, lt, nn

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import sklearn.datasets
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, classification_report
from sklearn.manifold import TSNE, LocallyLinearEmbedding

from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, dense_to_sparse

from torch_geometric.nn import VGAE, GATConv
np.random.seed(42)
torch.manual_seed(42)

digits = sklearn.datasets.load_digits()
X = digits.data
y = digits.target
k = 16
##Start with a naive "prior" of a k-nearest neighbor graph
adj_matrix = kneighbors_graph(X, k, mode='connectivity', include_self=False)
adj_matrix = adj_matrix + adj_matrix.T ##ensure symmetry
adj_matrix[adj_matrix > 1] = 1 ##binarize
edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix) ##convert to PyG format
data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, edge_weight=edge_weight, y=torch.tensor(y))

##hyperparameters
hidden = 32
latent = 16
heads = 4
dropout = 0.1
lr = 0.001
threshold_adj = 0.5 ##binary threshold for the adjacency matrix
##

##model definition
encoder = GATEncoder(data.num_features, hidden, latent, heads, dropout)
model = VGAE(encoder)

##optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

## Split data 80/20
train_mask = torch.rand(data.num_nodes) < 0.8
test_mask = ~train_mask 
print(f"{train_mask.sum().numpy()/data.num_nodes*100:.2f}% of nodes are in the training set")

data.train_mask = train_mask
data.test_mask = test_mask

##training main call
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index, data.edge_weight)
    train_edges_mask = data.train_mask[data.edge_index[0]] & data.train_mask[data.edge_index[1]]
    train_edges = data.edge_index[:, train_edges_mask]
    loss = model.recon_loss(z, train_edges) ##reconstruction loss
    loss = loss + (1 / data.num_nodes) * model.kl_loss() ##add KL divergence
    loss.backward()
    optimizer.step()
    return loss.item()

##training loop
epochs = 150
for epoch in range(1, epochs + 1):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')


##forward pass to both evaluate the model and get the embeddings
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.edge_index, data.edge_weight)
    adj_pred = model.decoder.forward_all(z)  
adj_pred_binary = (adj_pred > threshold_adj).float()
edge_index_pred, _ = dense_to_sparse(adj_pred_binary)
embeddings = z.cpu().numpy()

##Check how good the reconstruction is by comparing how well clustering works against the true labels
n_clusters = len(np.unique(y))
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)
nmi = normalized_mutual_info_score(y, cluster_labels)
ari = adjusted_rand_score(y, cluster_labels)
print(f'VGAE Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')

##compare to LTSA embeddings
ltsa = LocallyLinearEmbedding(n_components=latent, random_state=42, n_neighbors=k, method='ltsa')
embeddings_ltsa = ltsa.fit_transform(data.x)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_ltsa = kmeans.fit_predict(embeddings_ltsa)
nmi = normalized_mutual_info_score(y, cluster_labels_ltsa)
ari = adjusted_rand_score(y, cluster_labels_ltsa)
print(f'LTSA Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')

## Visualize the clusters in 2d using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

## Note, although they are mostly well separated, there are some "crosses" between clusters, but this could
## be due to the way the t-SNE algorithm works as well as the inherent nature of the data (e.g. a 3 can be sometimes mistaken for an 8)
plt.figure(figsize=(8, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y, cmap='tab10', s=10)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title('t-SNE Visualization of VGAE Embeddings')
plt.show()


## Check how well the embeddings work for downstream tasks
classifier1 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))
classifier2 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))

optimizer1 = torch.optim.Adam(classifier1.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(classifier2.parameters(), lr=0.01)

train_ltsa_embeddings = torch.tensor(embeddings_ltsa[train_mask], dtype=torch.float)
train_labels = torch.tensor(y[train_mask], dtype=torch.long)
test_ltsa_embeddings = torch.tensor(embeddings_ltsa[test_mask], dtype=torch.float)
test_labels = torch.tensor(y[test_mask], dtype=torch.long)

train_vgae_embeddings = torch.tensor(embeddings[train_mask], dtype=torch.float)
test_vgae_embeddings = torch.tensor(embeddings[test_mask], dtype=torch.float)

epochs = 100

for epoch in range(1, epochs + 1):
    classifier1.train()
    classifier2.train()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    output1 = classifier1(train_ltsa_embeddings)
    output2 = classifier2(train_vgae_embeddings)
    loss1 = F.nll_loss(output1, train_labels)
    loss2 = F.nll_loss(output2, train_labels)
    loss1.backward()
    loss2.backward()
    optimizer1.step()
    optimizer2.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d}, LTSA Loss: {loss1:.4f}, VGAE Loss: {loss2:.4f}')

with torch.no_grad():
    classifier1.eval()
    classifier2.eval()
    pred1 = classifier1(test_ltsa_embeddings).argmax(dim=1)
    pred2 = classifier2(test_vgae_embeddings).argmax(dim=1)

print('LTSA Classification Report:')
print(classification_report(test_labels, pred1.numpy()))
print('VGAE Classification Report:')
print(classification_report(test_labels, pred2.numpy()))
