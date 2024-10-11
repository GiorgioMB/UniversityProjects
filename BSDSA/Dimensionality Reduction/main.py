#%%
from sklearn import datasets
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.neighbors import NearestNeighbors
from torch_geometric import nn
from torch_sparse import add_
from xgboost import train 

def get_k_nearest_neighbors(data, k):
    """
    Given a dataset `data` (each row is a node), find the k-nearest neighbors for each node.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    return indices

def neighborhood_preservation_rate(original_data, embedded_data, k):
    """
    Compute the neighborhood preservation rate between original and embedded data.
    """
    original_neighbors = get_k_nearest_neighbors(original_data, k)
    embedded_neighbors = get_k_nearest_neighbors(embedded_data, k)
    npr_list = []
    for i in range(len(original_neighbors)):
        n_orig = set(original_neighbors[i])
        n_embed = set(embedded_neighbors[i])
        overlap = len(n_orig & n_embed)  # Intersection of original and embedded neighbors
        npr = overlap / k
        npr_list.append(npr)
    
    global_npr = np.mean(npr_list)
    return global_npr

def trustworthiness(high_dim_data, low_dim_data, k):
    """
    Compute the trustworthiness of the low-dimensional embedding.
    high_dim_data: original high-dimensional data
    low_dim_data: low-dimensional embedding of the data
    k: number of neighbors to consider
    """
    n = len(high_dim_data)
    knn_high = get_k_nearest_neighbors(high_dim_data, k)
    knn_low = get_k_nearest_neighbors(low_dim_data, k)
    trustworthiness_sum = 0.0
    
    for i in range(n):
        false_neighbors = set(knn_low[i]) - set(knn_high[i])
        
        for j in false_neighbors:
            if j in knn_high[i]:
                rank = np.where(knn_high[i] == j)[0][0] + 1
                trustworthiness_sum += rank - k
            else:
                trustworthiness_sum += (k + 1)
    
    normalization_factor = (2 / (n * k * (2 * n - 3 * k - 1)))
    trustworthiness_score = 1 - normalization_factor * trustworthiness_sum
    
    return trustworthiness_score

def continuity(high_dim_data, low_dim_data, k):
    """
    Compute the continuity of the low-dimensional embedding.
    high_dim_data: original high-dimensional data
    low_dim_data: low-dimensional embedding of the data
    k: number of neighbors to consider
    """
    n = len(high_dim_data)
    
    knn_high = get_k_nearest_neighbors(high_dim_data, k)
    knn_low = get_k_nearest_neighbors(low_dim_data, k)
    
    continuity_sum = 0.0
    
    for i in range(n):
        missing_neighbors = set(knn_high[i]) - set(knn_low[i])
        
        for j in missing_neighbors:
            if j in knn_low[i]:
                rank = np.where(knn_low[i] == j)[0][0] + 1
                continuity_sum += rank - k
            else:
                continuity_sum += (k + 1)  
    normalization_factor = (2 / (n * k * (2 * n - 3 * k - 1)))
    continuity_score = 1 - normalization_factor * continuity_sum
    
    return continuity_score


swiss_roll = datasets.make_swiss_roll(5000, noise=0.1, random_state=42)
data = swiss_roll[0]
min_dist_thres = 4
G = nx.Graph()
print("Beginning to construct the graph")
for i in range(len(data)):
    G.add_node(i, features=data[i])
print("Nodes added")
min_dist = np.inf
max_dist = -np.inf
distances = []
for i in range(len(data)):
    for j in range(len(data)):
        if i == j:
            continue
        dist = distance.euclidean(data[i], data[j])
        if dist < min_dist:
            min_dist = dist
        if dist > max_dist:
            max_dist = dist
        if dist < min_dist_thres:
            G.add_edge(i, j, weight=1/dist)
        distances.append(dist)
print("Edges added")
print("Minimum distance: ", min_dist)
print("Maximum distance: ", max_dist)
plt.hist(distances, bins=100)
plt.show()
data = from_networkx(G)
print(data)

#%%
##Graph Autoencoder that maps the data to a lower dimensional space and then back to the original space
class GraphAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GraphAutoencoder, self).__init__()
        self.conv1 = nn.GATConv(in_channels, hidden_channels, heads=heads, concat = False, add_self_loops=True)
        self.conv2 = nn.GATConv(hidden_channels, out_channels, heads=heads, concat = False, add_self_loops=True)
        self.heads = heads
    def encode(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        return x
    
    def decode(self, x, edge_index):
        x = self.conv2(x, edge_index)
        return x
    
    def forward(self, x, edge_index, edge_attr):
        x = self.encode(x, edge_index, edge_attr)
        x = self.decode(x, edge_index)
        return x

model = GraphAutoencoder(in_channels=3, hidden_channels=2, out_channels=3, heads=8)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):.2f}")
edge_index = data.edge_index
X = torch.tensor(data.features, dtype=torch.float)
edge_attr = data.weight
train_mask = np.random.choice([True, False], size=len(data.features), p=[0.8, 0.2])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = torch.nn.MSELoss()

for epoch in range(180):
    model.train()
    optimizer.zero_grad()
    out = model(X, edge_index, edge_attr)
    loss = criterion(out[train_mask], X[train_mask])
    
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    out = model(X, edge_index, edge_attr)
    test_loss = criterion(out[~train_mask], X[~train_mask])
    print(f"Test loss: {test_loss.item()}")


with torch.no_grad():
    low_dim_out = model.encode(X, edge_index, edge_attr)
npr_low_dim = neighborhood_preservation_rate(data.features, low_dim_out, k = 25)
npr_reconstr = neighborhood_preservation_rate(data.features, out, k = 25)
print(f"Neighborhood preservation rate: {npr_low_dim * 100:.2f}%")
print(f"Neighborhood preservation rate for reconstructed data: {npr_reconstr * 100:.2f}%")
continuity_score = continuity(data.features, low_dim_out, k=25)
trustworthiness_score = trustworthiness(data.features, low_dim_out, k=25)
print(f"Continuity score: {continuity_score*100:.2f}%")
print(f"Trustworthiness score: {trustworthiness_score*100:.2f}%")



# %%
##Compare to PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_fitted = pca.fit(data.features[train_mask])
pca_out = pca_fitted.transform(data.features)
npr_pca = neighborhood_preservation_rate(data.features, pca_out, k=25)
print(f"Neighborhood preservation rate for PCA: {npr_pca * 100:.2f}%")
continuity_pca = continuity(data.features, pca_out, k=25)
trustworthiness_pca = trustworthiness(data.features, pca_out, k=25)
print(f"Continuity score for PCA: {continuity_pca*100:.2f}%")
print(f"Trustworthiness score for PCA: {trustworthiness_pca*100:.2f}%")





# %%
##And LLE
from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method='ltsa')
lle_out = lle.fit(data.features[train_mask])
lle_out = lle_out.transform(data.features)
npr_lle = neighborhood_preservation_rate(data.features, lle_out, k=25)
print(f"Neighborhood preservation rate for LLE: {npr_lle * 100:.2f}%")
continuity_lle = continuity(data.features, lle_out, k=25)
trustworthiness_lle = trustworthiness(data.features, lle_out, k=25)
print(f"Continuity score for LLE: {continuity_lle*100:.2f}%")
print(f"Trustworthiness score for LLE: {trustworthiness_lle*100:.2f}%")
# %%
##And LPP
from lpproj import LocalityPreservingProjection 
lle2 = LocalityPreservingProjection(n_neighbors=10, n_components=2)
lpp_out = lle2.fit(data.features[train_mask]).transform(data.features)
npr_lpp = neighborhood_preservation_rate(data.features, lpp_out, k=25)
print(f"Neighborhood preservation rate for LPP: {npr_lpp * 100:.2f}%")
continuity_lpp = continuity(data.features, lpp_out, k=25)
trustworthiness_lpp = trustworthiness(data.features, lpp_out, k=25)
print(f"Continuity score for LPP: {continuity_lpp*100:.2f}%")
print(f"Trustworthiness score for LPP: {trustworthiness_lpp*100:.2f}%")

# %%
##And a normal auto encoder
class TraditionalAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TraditionalAutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        self.elu = torch.nn.ELU()

    def encode(self, x):
        for layer in self.encoder:
            x = self.elu(layer(x))
        return x
    
    def decode(self, x):
        for layer in self.decoder:
            x = self.elu(layer(x))
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

model_trad = TraditionalAutoEncoder(in_channels=3, hidden_channels=2, out_channels=3)
print(f"Number of parameters: {sum(p.numel() for p in model_trad.parameters())}")
optimizer_trad = torch.optim.Adam(model_trad.parameters(), lr=1e-2)
criterion_trad = torch.nn.MSELoss()

for epoch in range(800):
    model_trad.train()
    optimizer_trad.zero_grad()
    out_trad = model_trad(X)
    loss_trad = criterion_trad(out_trad[train_mask], X[train_mask])
    
    loss_trad.backward()
    optimizer_trad.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss_trad.item()}")

model_trad.eval()
with torch.no_grad():
    out_trad = model_trad(X)
    test_loss_trad = criterion_trad(out_trad[~train_mask], X[~train_mask])
    print(f"Test loss: {test_loss_trad.item()}")

low_dim_out_trad = model_trad.encode(X).detach()
npr_low_dim_trad = neighborhood_preservation_rate(data.features, low_dim_out_trad, k = 25)
npr_reconstr_trad = neighborhood_preservation_rate(data.features, out_trad, k = 25)
print(f"Neighborhood preservation rate for traditional AE: {npr_low_dim_trad * 100:.2f}%")
print(f"Neighborhood preservation rate for reconstructed data: {npr_reconstr_trad * 100:.2f}%")
continuity_score_trad = continuity(data.features, low_dim_out_trad, k=25)
trustworthiness_score_trad = trustworthiness(data.features, low_dim_out_trad, k=25)
print(f"Continuity score for traditional AE: {continuity_score_trad*100:.2f}%")
print(f"Trustworthiness score for traditional AE: {trustworthiness_score_trad*100:.2f}%")
