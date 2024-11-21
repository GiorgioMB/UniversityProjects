from graph_modules import *
from classical_modules import *
from metrics import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as patches
import seaborn as sns
from time import time

import torch
import torch.nn.functional as F
from torch import dropout, nn

from sklearn.datasets import load_digits
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, classification_report, silhouette_score
from sklearn.manifold import TSNE, LocallyLinearEmbedding,  trustworthiness, SpectralEmbedding, MDS, Isomap
from sklearn.decomposition import PCA

from pydiffmap import diffusion_map as dm
from pydiffmap.kernel import Kernel

from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, dense_to_sparse
from torch_geometric.nn import VGAE, Node2Vec

np.random.seed(42)
torch.manual_seed(42)
sns.set(context='paper', style='whitegrid', font_scale=1.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

colors = {
    'PCA': '#D55E00',               # Orange
    'MDS': '#E69F00',               # Yellow
    'Isomap': '#56B4E9',            # Sky Blue
    'LLE': '#FF0000',               # Red
    'LTSA': '#0072B2',              # Blue
    'Laplacian Eigenmaps': '#8A2BE2',  # Violet
    'Diffusion Map': '#009E73',         # Green
    'AE': '#F0E442',               # Bright Yellow
    'VAE': '#F58231',              # Light Orange
    'Node2Vec': '#000000',         # Black
    'VGAE': '#911EB4'              # Dark Purple
}


def generate_d_dimensional_flipped_spiral(d, n, noise=0.2, radial_scaling=1.5):
    """
    Generate a d-dimensional dataset with d flipped spiral arms.
    Each arm alternates between mirrored and standard directions.
    """
    # Initialize the data matrix (X) and labels (y)
    X = np.zeros((d * n, d))
    y = np.zeros(d * n, dtype=int)

    # Generate the arms
    for i in range(d):
        # Generate the angle theta
        theta = np.linspace(0, 4 * np.pi, n)  # Full spiral
        r = radial_scaling * theta  # Radial component grows linearly with theta

        # Map (r, theta) to Cartesian coordinates
        x = r * np.cos(theta) + noise * np.random.randn(n)
        y_coord = r * np.sin(theta) + noise * np.random.randn(n)

        # Apply flipping to alternate arms
        if i % 2 == 1:  # Flip for odd-indexed arms
            x = -x
            y_coord = -y_coord

        # Assign to the appropriate dimensions
        X[i * n:(i + 1) * n, 0] = x
        X[i * n:(i + 1) * n, 1] = y_coord

        # Additional dimensions (systematic offsets or mirrored patterns)
        for dim in range(2, d):
            if i % 2 == 1:
                X[i * n:(i + 1) * n, dim] = -0.5 * (dim + 1) + noise * np.random.randn(n)
            else:
                X[i * n:(i + 1) * n, dim] = 0.5 * (dim + 1) + noise * np.random.randn(n)

        # Assign class labels (alternating by arm index)
        y[i * n:(i + 1) * n] = i % 2

    return X, y

def visualize_spiral():
    X, y = generate_d_dimensional_flipped_spiral(2, 1000)
    ##plot with sns
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis', legend='full')
    plt.title('2D Spiral Dataset')
    plt.legend(title='Arm Index', loc='upper right')
    plt.tight_layout()
    plt.show()



##hyperparameters
d = 48
n = 2000
k = 16 ##number of neighbors for the k-nearest neighbor graph
hidden = 32
latent = 16
heads = 4
dropout = 0.1
walk_length = 20
context_size = 10
walks_per_node = 10
num_negative_samples = 5
p = 1
q = 1
lr = 0.001
epochs = 500
epochs_downstream = 1000
##
X, y = generate_d_dimensional_flipped_spiral(d, n)
visualize_spiral()


##Start with a naive "prior" of a k-nearest neighbor graph
adj_matrix = kneighbors_graph(X, k, mode='connectivity', include_self=False)
adj_matrix = adj_matrix + adj_matrix.T ##ensure symmetry
adj_matrix[adj_matrix > 1] = 1 ##binarize
edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix) ##convert to PyG format
data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, edge_weight=edge_weight, y=torch.tensor(y))



encoder = VGATEncoder(data.num_features, hidden, latent, heads, dropout)
model = VGAE(encoder).to(device)

node2vec = Node2Vec(edge_index=data.edge_index, embedding_dim=latent, walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node, num_negative_samples=num_negative_samples, p=p, q=q).to(device)

vae_encoder = Encoder(data.num_features, hidden, latent, 4)
vae_decoder = Decoder(latent, hidden, data.num_features, 4)
vae_mean_module = MeanModule(latent, latent)
vae_logvar_module = LogVarModule(latent, latent)
trad_vae = VAE(vae_encoder, vae_decoder, vae_mean_module, vae_logvar_module).to(device)

encoder_ae = Encoder(data.num_features, hidden, latent, 4)
decoder_ae = Decoder(latent, hidden, data.num_features, 4)
ae = AutoEncoder(encoder_ae, decoder_ae).to(device)

print("Number of parameters in VGAE:", sum(p.numel() for p in model.parameters()))
print("Number of parameters in Node2Vec:", sum(p.numel() for p in node2vec.parameters()))
print("Number of parameters in VAE:", sum(p.numel() for p in trad_vae.parameters()))
print("Number of parameters in AE:", sum(p.numel() for p in ae.parameters()))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer_node2vec = torch.optim.Adam(node2vec.parameters(), lr=lr)
optimizer_trad = torch.optim.Adam(trad_vae.parameters(), lr=lr)
optimizer_ae = torch.optim.Adam(ae.parameters(), lr=lr)

train_mask = torch.rand(data.num_nodes) < 0.8
test_mask = ~train_mask 
print(f"{train_mask.sum().numpy()/data.num_nodes*100:.2f}% of nodes are in the training set")

data.train_mask = train_mask.to(device)
data.test_mask = test_mask.to(device)

def train_vgae():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index, data.edge_weight)
    train_edges_mask = data.train_mask[data.edge_index[0]] & data.train_mask[data.edge_index[1]]
    train_edges = data.edge_index[:, train_edges_mask]
    recon_loss = model.recon_loss(z, train_edges)  
    kl_loss = (1 / data.num_nodes) * model.kl_loss()  
    loss = recon_loss + kl_loss  
    loss.backward()
    optimizer.step()
    return kl_loss.item(), recon_loss.item()


def train_n2v():
    loader = node2vec.loader(batch_size=128, shuffle=True)
    pos_rw, neg_rw = next(iter(loader))
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer_node2vec.zero_grad()
        loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer_node2vec.step()
        total_loss += loss.item()
    return total_loss / len(loader)


##training loops
print("Starting Node2Vec training")
t0 = time()
n2v_losses = []
for epoch in range(1, epochs + 1):
    loss = train_n2v()
    n2v_losses.append(loss)
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}', end='\r')
t1 = time()
print(f"\nNode2Vec training took {t1-t0:.2f} seconds")

kl_losses = []
recon_losses = []
print("\nStarting VGAE training")
t0 = time()
for epoch in range(1, epochs + 1):
    kl_loss, recon_loss = train_vgae()
    kl_losses.append(kl_loss)
    recon_losses.append(recon_loss)
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:03d}, KL Loss: {kl_loss:.4f}, Recon Loss: {recon_loss:.4f}', end='\r')
t1 = time()
print(f"\nVGAE training took {t1-t0:.2f} seconds")
print("\nStarting traditional VAE training")
trad_loss_kl = []
trad_loss_recon = []
t0 = time()
for epoch in range(1, epochs + 1):
    trad_vae.train()
    optimizer_trad.zero_grad()
    recon_x, mean, logvar = trad_vae(data.x)
    recon_loss = trad_vae.recon_loss(recon_x[train_mask], data.x[train_mask])
    kl_loss = trad_vae.kl_loss(mean, logvar)
    loss = recon_loss + kl_loss
    loss.backward()
    optimizer_trad.step()
    trad_loss_kl.append(kl_loss.item())
    trad_loss_recon.append(recon_loss.item())
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:03d}, KL Loss: {kl_loss:.4f}, Recon Loss: {recon_loss:.4f}', end='\r')
t1 = time()
print(f"\nVAE training took {t1-t0:.2f} seconds")
print("\nStarting traditional AE training")
trad_ae_losses = []
t0 = time()
for epoch in range(1, epochs + 1):
    ae.train()
    optimizer_ae.zero_grad()
    recon_x = ae(data.x)
    loss = F.mse_loss(recon_x[train_mask], data.x[train_mask])
    loss.backward()
    optimizer_ae.step()
    trad_ae_losses.append(loss.item())
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:03d}, Recon Loss: {loss:.4f}', end='\r')
t1 = time()
print(f"\nAE training took {t1-t0:.2f} seconds")



plt.figure(figsize=(8, 4))
sns.lineplot(x=range(1, epochs + 1), y=kl_losses, label='VGAE KL Loss', color=colors['VGAE'])
sns.lineplot(x=range(1, epochs + 1), y=trad_loss_kl, label='VAE KL Loss', color=colors['VAE'])
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training KL Loss')
plt.legend()
ax_main = plt.gca()
ax_inset = inset_axes(ax_main, width="30%", height="30%", loc="lower left", 
                      bbox_to_anchor=(1.05, 0.2, 1, 1), 
                      bbox_transform=ax_main.transAxes)
for losses, label, color in zip(
    [trad_loss_kl, kl_losses],
    ['VAE', 'VGAE'],
    [colors['VAE'], colors['VGAE']]
):
    ax_inset.plot(range(490, 500), losses[490:500], label=f'{label} Downstream Loss', color=color)
ax_inset.set_xlim(490, 500)
ax_inset.set_ylim(0, 1)
ax_inset.set_xticks([490, 495, 500])
ax_inset.set_yticks([0, 0.5, 1])
ax_inset.tick_params(axis='both', which='major', labelsize=8)
rect = patches.Rectangle((490, 0), 10, 1, linewidth=1, edgecolor='black', facecolor='none')
ax_main.add_patch(rect)
mark_inset(ax_main, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
plt.show()



plt.figure(figsize=(8, 4))
sns.lineplot(x=range(1, epochs + 1), y=recon_losses, label='VGAE Reconstruction Loss', color=colors['VGAE'])
sns.lineplot(x=range(1, epochs + 1), y=trad_loss_recon, label='VAE Reconstruction Loss', color=colors['VAE'])
sns.lineplot(x=range(1, epochs + 1), y=trad_ae_losses, label='AE Reconstruction Loss', color=colors['AE'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Reconstruction Loss')
plt.legend()
ax_main = plt.gca()
ax_inset = inset_axes(ax_main, width="30%", height="30%", loc="lower left",
                        bbox_to_anchor=(1.05, 0.2, 1, 1),
                        bbox_transform=ax_main.transAxes)

for losses, label, color in zip(
    [trad_loss_recon, recon_losses, trad_ae_losses],
    ['VAE', 'VGAE', 'AE'],
    [colors['VAE'], colors['VGAE'], colors['AE']]
):
    ax_inset.plot(range(490, 500), losses[490:500], label=f'{label} Downstream Loss', color=color)

ax_inset.set_xlim(490, 500)
ax_inset.set_ylim(10, 15)
ax_inset.set_xticks([490, 495, 500])
ax_inset.set_yticks([10, 12.5, 15])
ax_inset.tick_params(axis='both', which='major', labelsize=8)

rect = patches.Rectangle((490, 10), 10, 5, linewidth=1, edgecolor='black', facecolor='none')
ax_main.add_patch(rect)
mark_inset(ax_main, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
plt.show()

plt.figure(figsize=(8, 4))
sns.lineplot(x=range(1, epochs + 1), y=n2v_losses, label='Node2Vec Loss', color=colors['Node2Vec'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Node2Vec Training Loss')
plt.legend()
plt.show()

model.eval()
trad_vae.eval()
node2vec.eval()
ae.eval()
with torch.no_grad():
    z = model.encode(data.x, data.edge_index, data.edge_weight)
    z_trad = trad_vae.project(data.x)
    z_node2vec = node2vec()
    z_ae = ae.encode(data.x)



##Check how good the reconstruction is by comparing different metrics
n_clusters = len(np.unique(y))
pca = PCA(n_components=latent, random_state=42)
t0 = time()
embeddings_pca = pca.fit_transform(data.x)
t1 = time()
print(f"PCA took {t1-t0:.2f} seconds")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_pca = kmeans.fit_predict(embeddings_pca)
nmi = normalized_mutual_info_score(y, cluster_labels_pca)
ari = adjusted_rand_score(y, cluster_labels_pca)
print(f'PCA Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')

mds = MDS(n_components=latent, random_state=42)
t0 = time()
embeddings_mds = mds.fit_transform(data.x)
t1 = time()
print(f"MDS took {t1-t0:.2f} seconds")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_mds = kmeans.fit_predict(embeddings_mds)
nmi = normalized_mutual_info_score(y, cluster_labels_mds)
ari = adjusted_rand_score(y, cluster_labels_mds)
print(f'MDS Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')

isomap = Isomap(n_components=latent, n_neighbors=k)
t0 = time()
embeddings_isomap = isomap.fit_transform(data.x)
t1 = time()
print(f"Isomap took {t1-t0:.2f} seconds")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_isomap = kmeans.fit_predict(embeddings_isomap)
nmi = normalized_mutual_info_score(y, cluster_labels_isomap)
ari = adjusted_rand_score(y, cluster_labels_isomap)
print(f'Isomap Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')


lle = LocallyLinearEmbedding(n_components=latent, random_state=42, n_neighbors=k)
t0 = time()
embeddings_lle = lle.fit_transform(data.x)
t1 = time()
print(f"LLE took {t1-t0:.2f} seconds")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_lle = kmeans.fit_predict(embeddings_lle)
nmi = normalized_mutual_info_score(y, cluster_labels_lle)
ari = adjusted_rand_score(y, cluster_labels_lle)
print(f'LLE Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')

ltsa = LocallyLinearEmbedding(n_components=latent, random_state=42, n_neighbors=k, method='ltsa')
t0 = time()
embeddings_ltsa = ltsa.fit_transform(data.x)
t1 = time()
print(f"LTSA took {t1-t0:.2f} seconds")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_ltsa = kmeans.fit_predict(embeddings_ltsa)
nmi = normalized_mutual_info_score(y, cluster_labels_ltsa)
ari = adjusted_rand_score(y, cluster_labels_ltsa)
print(f'LTSA Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')

laplacian = SpectralEmbedding(n_components = latent, random_state=42, affinity='nearest_neighbors')
t0 = time()
embeddings_laplacian = laplacian.fit_transform(data.x)
t1 = time()
print(f"Laplacian Eigenmaps took {t1-t0:.2f} seconds")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_laplacian = kmeans.fit_predict(embeddings_laplacian)
nmi = normalized_mutual_info_score(y, cluster_labels_laplacian)
ari = adjusted_rand_score(y, cluster_labels_laplacian)
print(f'Laplacian Eigenmaps Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')

kernel = Kernel()
# Initialize and apply the Diffusion Map
diffmap = dm.DiffusionMap(kernel_object=kernel, alpha=0.2, n_evecs=latent, oos='nystroem')
t0 = time()
embeddings_diffmap = diffmap.fit_transform(data.x)
t1 = time()
print(f"Diffusion Map took {t1-t0:.2f} seconds")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_diffmap = kmeans.fit_predict(embeddings_diffmap)
nmi = normalized_mutual_info_score(y, cluster_labels_diffmap)
ari = adjusted_rand_score(y, cluster_labels_diffmap)
print(f'Diffusion Map Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')

embeddings_ae = z_ae.cpu().numpy()
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_ae = kmeans.fit_predict(embeddings_ae)
nmi = normalized_mutual_info_score(y, cluster_labels_ae)
ari = adjusted_rand_score(y, cluster_labels_ae)
print(f'AE Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')

embeddings_trad = z_trad.cpu().numpy()
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_trad = kmeans.fit_predict(embeddings_trad)
nmi = normalized_mutual_info_score(y, cluster_labels_trad)
ari = adjusted_rand_score(y, cluster_labels_trad)
print(f'VAE Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')

embeddings_n2v = z_node2vec.detach().cpu().numpy()
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_n2v = kmeans.fit_predict(embeddings_n2v)
nmi = normalized_mutual_info_score(y, cluster_labels_n2v)
ari = adjusted_rand_score(y, cluster_labels_n2v)
print(f'Node2Vec Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')

embeddings = z.cpu().numpy()
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)
nmi = normalized_mutual_info_score(y, cluster_labels)
ari = adjusted_rand_score(y, cluster_labels)
print(f'VGAE Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}')

silhouette_pca = silhouette_score(embeddings_pca, y)
silhouette_mds = silhouette_score(embeddings_mds, y)
silhouette_isomap = silhouette_score(embeddings_isomap, y)
silhouette_lle = silhouette_score(embeddings_lle, y)
silhouette_ltsa = silhouette_score(embeddings_ltsa, y)
silhouette_laplacian = silhouette_score(embeddings_laplacian, y)
silhouette_diffmap = silhouette_score(embeddings_diffmap, y)
silhouette_trad = silhouette_score(embeddings_trad, y)
silhouette_vgae = silhouette_score(embeddings, y)
silhouette_ae = silhouette_score(embeddings_ae, y)
silhouette_n2v = silhouette_score(embeddings_n2v, y)

print('Silhouette Score PCA:', silhouette_pca)
print('Silhouette Score MDS:', silhouette_mds)
print('Silhouette Score Isomap:', silhouette_isomap)
print('Silhouette Score LLE:', silhouette_lle)
print('Silhouette Score LTSA:', silhouette_ltsa)
print('Silhouette Score Laplacian Eigenmaps:', silhouette_laplacian)
print('Silhouette Score Diffusion Map:', silhouette_diffmap)
print('Silhouette Score AE:', silhouette_ae)
print('Silhouette Score VAE:', silhouette_trad)
print('Silhouette Score Node2Vec:', silhouette_n2v)
print('Silhouette Score VGAE:', silhouette_vgae)

corr_pca = calculate_pairwise_distance_correlation(data.x.cpu().numpy(), embeddings_pca)
corr_mds = calculate_pairwise_distance_correlation(data.x.cpu().numpy(), embeddings_mds)
corr_isomap = calculate_pairwise_distance_correlation(data.x.cpu().numpy(), embeddings_isomap)
corr_lle = calculate_pairwise_distance_correlation(data.x.cpu().numpy(), embeddings_lle)
corr_ltsa = calculate_pairwise_distance_correlation(data.x.cpu().numpy(), embeddings_ltsa)
corr_laplacian = calculate_pairwise_distance_correlation(data.x.cpu().numpy(), embeddings_laplacian)
corr_diffmap = calculate_pairwise_distance_correlation(data.x.cpu().numpy(), embeddings_diffmap)
corr_trad = calculate_pairwise_distance_correlation(data.x.cpu().numpy(), embeddings_trad)
corr_vgae = calculate_pairwise_distance_correlation(data.x.cpu().numpy(), embeddings)
corr_ae = calculate_pairwise_distance_correlation(data.x.cpu().numpy(), embeddings_ae)
corr_n2v = calculate_pairwise_distance_correlation(data.x.cpu().numpy(), embeddings_n2v)

print('Pairwise Distance Correlation PCA:', corr_pca)
print('Pairwise Distance Correlation MDS:', corr_mds)
print('Pairwise Distance Correlation Isomap:', corr_isomap)
print('Pairwise Distance Correlation LLE:', corr_lle)
print('Pairwise Distance Correlation LTSA:', corr_ltsa)
print('Pairwise Distance Correlation Laplacian Eigenmaps:', corr_laplacian)
print('Pairwise Distance Correlation Diffusion Map:', corr_diffmap)
print('Pairwise Distance Correlation AE:', corr_ae)
print('Pairwise Distance Correlation VAE:', corr_trad)
print('Pairwise Distance Correlation Node2Vec:', corr_n2v)
print('Pairwise Distance Correlation VGAE:', corr_vgae)

trust_mds = trustworthiness(data.x.cpu().numpy(), embeddings_mds, n_neighbors=10)
trust_isomap = trustworthiness(data.x.cpu().numpy(), embeddings_isomap, n_neighbors=10)
trust_vgae = trustworthiness(data.x.cpu().numpy(), embeddings, n_neighbors=10)
trust_lle = trustworthiness(data.x.cpu().numpy(), embeddings_lle, n_neighbors=10)
trust_vae = trustworthiness(data.x.cpu().numpy(), embeddings_trad, n_neighbors=10)
trust_pca = trustworthiness(data.x.cpu().numpy(), embeddings_pca, n_neighbors=10)
trust_ltsa = trustworthiness(data.x.cpu().numpy(), embeddings_ltsa, n_neighbors=10)
trust_laplace = trustworthiness(data.x.cpu().numpy(), embeddings_laplacian, n_neighbors=10)
trust_diffmap = trustworthiness(data.x.cpu().numpy(), embeddings_diffmap, n_neighbors=10)
trust_ae = trustworthiness(data.x.cpu().numpy(), embeddings_ae, n_neighbors=10)
trust_n2v = trustworthiness(data.x.cpu().numpy(), embeddings_n2v, n_neighbors=10)

print('Trustworthiness PCA:', trust_pca)
print('Trustworthiness MDS:', trust_mds)
print('Trustworthiness Isomap:', trust_isomap)
print('Trustworthiness LLE:', trust_lle)
print('Trustworthiness LTSA:', trust_ltsa)
print('Trustworthiness Laplacian Eigenmaps:', trust_laplace)
print('Trustworthiness Diffusion Map:', trust_diffmap)
print('Trustworthiness AE:', trust_ae)
print('Trustworthiness VAE:', trust_vae)
print('Trustworthiness Node2Vec:', trust_n2v)
print('Trustworthiness VGAE:', trust_vgae)

continuity_vgae = calculate_continuity(data.x.cpu().numpy(), embeddings, neighbours=10)
continuity_lle = calculate_continuity(data.x.cpu().numpy(), embeddings_lle, neighbours=10)
continuity_vae = calculate_continuity(data.x.cpu().numpy(), embeddings_trad, neighbours=10)
continuity_pca = calculate_continuity(data.x.cpu().numpy(), embeddings_pca, neighbours=10)
continuity_ltsa = calculate_continuity(data.x.cpu().numpy(), embeddings_ltsa, neighbours=10)
continuity_laplacian = calculate_continuity(data.x.cpu().numpy(), embeddings_laplacian, neighbours=10)
continuity_diffmap = calculate_continuity(data.x.cpu().numpy(), embeddings_diffmap, neighbours=10)
continuity_ae = calculate_continuity(data.x.cpu().numpy(), embeddings_ae, neighbours=10)
continuity_n2v = calculate_continuity(data.x.cpu().numpy(), embeddings_n2v, neighbours=10)
continuity_mds = calculate_continuity(data.x.cpu().numpy(), embeddings_mds, neighbours=10)
continuity_isomap = calculate_continuity(data.x.cpu().numpy(), embeddings_isomap, neighbours=10)

print('Continuity PCA:', continuity_pca)
print('Continuity MDS:', continuity_mds)
print('Continuity Isomap:', continuity_isomap)
print('Continuity LLE:', continuity_lle)
print('Continuity LTSA:', continuity_ltsa)
print('Continuity Laplacian Eigenmaps:', continuity_laplacian)
print('Continuity Diffusion Map:', continuity_diffmap)
print('Continuity AE:', continuity_ae)
print('Continuity VAE:', continuity_vae)
print('Continuity Node2Vec:', continuity_n2v)
print('Continuity VGAE:', continuity_vgae)

npr_vgae = calculate_npr(data.x.cpu().numpy(), embeddings, neighbours=10)
npr_vgae = calculate_npr(data.x.cpu().numpy(), embeddings, neighbours=10)
npr_vae = calculate_npr(data.x.cpu().numpy(), embeddings_trad, neighbours=10)
npr_pca = calculate_npr(data.x.cpu().numpy(), embeddings_pca, neighbours=10)
npr_lle = calculate_npr(data.x.cpu().numpy(), embeddings_lle, neighbours=10)
npr_ltsa = calculate_npr(data.x.cpu().numpy(), embeddings_ltsa, neighbours=10)
npr_laplacian = calculate_npr(data.x.cpu().numpy(), embeddings_laplacian, neighbours=10)
npr_diffmap = calculate_npr(data.x.cpu().numpy(), embeddings_diffmap, neighbours=10)
npr_ae = calculate_npr(data.x.cpu().numpy(), embeddings_ae, neighbours=10)
npr_n2v = calculate_npr(data.x.cpu().numpy(), embeddings_n2v, neighbours=10)
npr_mds = calculate_npr(data.x.cpu().numpy(), embeddings_mds, neighbours=10)
npr_isomap = calculate_npr(data.x.cpu().numpy(), embeddings_isomap, neighbours=10)

print('NPR PCA:', npr_pca)
print('NPR MDS:', npr_mds)
print('NPR Isomap:', npr_isomap)
print('NPR LLE:', npr_lle)
print('NPR LTSA:', npr_ltsa)
print('NPR Laplacian Eigenmaps:', npr_laplacian)
print('NPR Diffusion Map:', npr_diffmap)
print('NPR AE:', npr_ae)
print('NPR VAE:', npr_vae)
print('NPR Node2Vec:', npr_n2v)
print('NPR VGAE:', npr_vgae)

stress_pca = calculate_stress(data.x.cpu().numpy(), embeddings_pca)
stress_mds = calculate_stress(data.x.cpu().numpy(), embeddings_mds)
stress_isomap = calculate_stress(data.x.cpu().numpy(), embeddings_isomap)
stress_lle = calculate_stress(data.x.cpu().numpy(), embeddings_lle)
stress_ltsa = calculate_stress(data.x.cpu().numpy(), embeddings_ltsa)
stress_laplacian = calculate_stress(data.x.cpu().numpy(), embeddings_laplacian)
stress_diffmap = calculate_stress(data.x.cpu().numpy(), embeddings_diffmap)
stress_trad = calculate_stress(data.x.cpu().numpy(), embeddings_trad)
stress_vgae = calculate_stress(data.x.cpu().numpy(), embeddings)
stress_ae = calculate_stress(data.x.cpu().numpy(), embeddings_ae)
stress_n2v = calculate_stress(data.x.cpu().numpy(), embeddings_n2v)

print('Stress PCA:', stress_pca)
print('Stress MDS:', stress_mds)
print('Stress Isomap:', stress_isomap)
print('Stress LLE:', stress_lle)
print('Stress LTSA:', stress_ltsa)
print('Stress Laplacian Eigenmaps:', stress_laplacian)
print('Stress Diffusion Map:', stress_diffmap)
print('Stress AE:', stress_ae)
print('Stress VAE:', stress_trad)
print('Stress Node2Vec:', stress_n2v)
print('Stress VGAE:', stress_vgae)


## Visualize the clusters in 2d using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

## Note, although they are mostly well separated, there are some "crosses" between clusters, but this could
## be due to the way the t-SNE algorithm works as well as the inherent nature of the data (e.g. a 3 can be sometimes mistaken for an 8)
plt.figure(figsize=(8, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y, cmap='tab10', s=10)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.xticks([])
plt.yticks([])
plt.title('t-SNE Visualization of VGAE Embeddings')
plt.show()

embeddings_2d_n2v = tsne.fit_transform(embeddings_n2v)
plt.figure(figsize=(8, 8))
scatter = plt.scatter(embeddings_2d_n2v[:, 0], embeddings_2d_n2v[:, 1], c=y, cmap='tab10', s=10)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.xticks([])
plt.yticks([])
plt.title('t-SNE Visualization of Node2Vec Embeddings')
plt.show()

embeddings_2d_vae = tsne.fit_transform(embeddings_trad)
plt.figure(figsize=(8, 8))
scatter = plt.scatter(embeddings_2d_vae[:, 0], embeddings_2d_vae[:, 1], c=y, cmap='tab10', s=10)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.xticks([])
plt.yticks([])
plt.title('t-SNE Visualization of VAE Embeddings')
plt.show()

embeddings_2d_pca = tsne.fit_transform(embeddings_pca)
plt.figure(figsize=(8, 8))
scatter = plt.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], c=y, cmap='tab10', s=10)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.xticks([])
plt.yticks([])
plt.title('t-SNE Visualization of PCA Embeddings')
plt.show()

## Check how well the embeddings work for downstream tasks
classifier1 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))
classifier2 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))
classifier3 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))
classifier4 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))
classifier5 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))
classifier6 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))
classifier7 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))
classifier8 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))
classifier9 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))
classifier10 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))
classifier11 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 10), nn.LogSoftmax(dim=1))

optimizer1 = torch.optim.Adam(classifier1.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(classifier2.parameters(), lr=0.01)
optimizer3 = torch.optim.Adam(classifier3.parameters(), lr=0.01)
optimizer4 = torch.optim.Adam(classifier4.parameters(), lr=0.01)
optimizer5 = torch.optim.Adam(classifier5.parameters(), lr=0.01)
optimizer6 = torch.optim.Adam(classifier6.parameters(), lr=0.01)
optimizer7 = torch.optim.Adam(classifier7.parameters(), lr=0.01)
optimizer8 = torch.optim.Adam(classifier8.parameters(), lr=0.01)
optimizer9 = torch.optim.Adam(classifier9.parameters(), lr=0.01)
optimizer10 = torch.optim.Adam(classifier10.parameters(), lr=0.01)
optimizer11 = torch.optim.Adam(classifier11.parameters(), lr=0.01)

train_ltsa_embeddings = torch.tensor(embeddings_ltsa[train_mask], dtype=torch.float)
train_labels = torch.tensor(y[train_mask], dtype=torch.long)
test_ltsa_embeddings = torch.tensor(embeddings_ltsa[test_mask], dtype=torch.float)
test_labels = torch.tensor(y[test_mask], dtype=torch.long)

train_vgae_embeddings = torch.tensor(embeddings[train_mask], dtype=torch.float)
test_vgae_embeddings = torch.tensor(embeddings[test_mask], dtype=torch.float)

train_vae_embeddings = torch.tensor(embeddings_trad[train_mask], dtype=torch.float)
test_vae_embeddings = torch.tensor(embeddings_trad[test_mask], dtype=torch.float)

train_pca_embeddings = torch.tensor(embeddings_pca[train_mask], dtype=torch.float)
test_pca_embeddings = torch.tensor(embeddings_pca[test_mask], dtype=torch.float)

train_diffmap_embeddings = torch.tensor(embeddings_diffmap[train_mask], dtype=torch.float)
test_diffmap_embeddings = torch.tensor(embeddings_diffmap[test_mask], dtype=torch.float)

train_ae_embeddings = torch.tensor(embeddings_ae[train_mask], dtype=torch.float)
test_ae_embeddings = torch.tensor(embeddings_ae[test_mask], dtype=torch.float)

train_n2v_embeddings = torch.tensor(embeddings_n2v[train_mask], dtype=torch.float)
test_n2v_embeddings = torch.tensor(embeddings_n2v[test_mask], dtype=torch.float)

train_lle_embeddings = torch.tensor(embeddings_lle[train_mask], dtype=torch.float)
test_lle_embeddings = torch.tensor(embeddings_lle[test_mask], dtype=torch.float)

train_laplacian_embeddings = torch.tensor(embeddings_laplacian[train_mask], dtype=torch.float)
test_laplacian_embeddings = torch.tensor(embeddings_laplacian[test_mask], dtype=torch.float)

train_mds_embeddings = torch.tensor(embeddings_mds[train_mask], dtype=torch.float)
test_mds_embeddings = torch.tensor(embeddings_mds[test_mask], dtype=torch.float)

train_isomap_embeddings = torch.tensor(embeddings_isomap[train_mask], dtype=torch.float)
test_isomap_embeddings = torch.tensor(embeddings_isomap[test_mask], dtype=torch.float)

classifier1.train()
classifier2.train()
classifier3.train()
classifier4.train()
classifier5.train()
classifier6.train()
classifier7.train()
classifier8.train()
classifier9.train()
classifier10.train()
classifier11.train()
losses_vgae_downstream = []
losses_lle_downstream = []
losses_ltsa_downstream = []
losses_vae_downstream = []
losses_pca_downstream = []
losses_diffmap_downstream = []
losses_ae_downstream = []
losses_n2v_downstream = []
losses_laplacian_downstream = []
losses_mds_downstream = []
losses_isomap_downstream = []
for epoch in range(1, epochs_downstream + 1):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    optimizer4.zero_grad()
    optimizer5.zero_grad()
    optimizer6.zero_grad()
    optimizer7.zero_grad()
    optimizer8.zero_grad()
    optimizer9.zero_grad()
    optimizer10.zero_grad()
    optimizer11.zero_grad()
    output1 = classifier1(train_ltsa_embeddings)
    output2 = classifier2(train_vgae_embeddings)
    output3 = classifier3(train_vae_embeddings)
    output4 = classifier4(train_pca_embeddings)
    output5 = classifier5(train_diffmap_embeddings)
    output6 = classifier6(train_ae_embeddings)
    output7 = classifier7(train_n2v_embeddings)
    output8 = classifier8(train_lle_embeddings)
    output9 = classifier9(train_laplacian_embeddings)
    output10 = classifier10(train_mds_embeddings)
    output11 = classifier11(train_isomap_embeddings)
    loss1 = F.nll_loss(output1, train_labels)
    loss2 = F.nll_loss(output2, train_labels)
    loss3 = F.nll_loss(output3, train_labels)
    loss4 = F.nll_loss(output4, train_labels)
    loss5 = F.nll_loss(output5, train_labels)
    loss6 = F.nll_loss(output6, train_labels)
    loss7 = F.nll_loss(output7, train_labels)
    loss8 = F.nll_loss(output8, train_labels)
    loss9 = F.nll_loss(output9, train_labels)
    loss10 = F.nll_loss(output10, train_labels)
    loss11 = F.nll_loss(output11, train_labels)
    losses_ltsa_downstream.append(loss1.item())
    losses_vgae_downstream.append(loss2.item())
    losses_vae_downstream.append(loss3.item())
    losses_pca_downstream.append(loss4.item())
    losses_diffmap_downstream.append(loss5.item())
    losses_ae_downstream.append(loss6.item())
    losses_n2v_downstream.append(loss7.item())
    losses_lle_downstream.append(loss8.item())
    losses_laplacian_downstream.append(loss9.item())
    losses_mds_downstream.append(loss10.item())
    losses_isomap_downstream.append(loss11.item())
    loss1.backward()
    loss2.backward()
    loss3.backward()
    loss4.backward()
    loss5.backward()
    loss6.backward()
    loss7.backward()
    loss8.backward()
    loss9.backward()
    loss10.backward()
    loss11.backward()
    optimizer1.step()
    optimizer2.step()
    optimizer3.step()
    optimizer4.step()
    optimizer5.step()
    optimizer6.step()
    optimizer7.step()
    optimizer8.step()
    optimizer9.step()
    optimizer10.step()
    optimizer11.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch:04d}', end='\r')



plt.figure(figsize=(15, 10))
sns.lineplot(x=range(1, epochs + 1), y=losses_pca_downstream, label='PCA Downstream Loss', color=colors['PCA'])
sns.lineplot(x=range(1, epochs + 1), y=losses_mds_downstream, label='MDS Downstream Loss', color=colors['MDS'])
sns.lineplot(x=range(1, epochs + 1), y=losses_isomap_downstream, label='Isomap Downstream Loss', color=colors['Isomap'])
sns.lineplot(x=range(1, epochs + 1), y=losses_lle_downstream, label='LLE Downstream Loss', color=colors['LLE'])
sns.lineplot(x=range(1, epochs + 1), y=losses_ltsa_downstream, label='LTSA Downstream Loss', color=colors['LTSA'])
sns.lineplot(x=range(1, epochs + 1), y=losses_laplacian_downstream, label='Laplacian Eigenmaps Downstream Loss', color=colors['Laplacian Eigenmaps'])
sns.lineplot(x=range(1, epochs + 1), y=losses_diffmap_downstream, label='Diffusion Map Downstream Loss', color=colors['Diffusion Map'])
sns.lineplot(x=range(1, epochs + 1), y=losses_ae_downstream, label='AE Downstream Loss', color=colors['AE'])
sns.lineplot(x=range(1, epochs + 1), y=losses_vae_downstream, label='VAE Downstream Loss', color=colors['VAE'])
sns.lineplot(x=range(1, epochs + 1), y=losses_n2v_downstream, label='Node2Vec Downstream Loss', color=colors['Node2Vec'])
sns.lineplot(x=range(1, epochs + 1), y=losses_vgae_downstream, label='VGAE Downstream Loss', color=colors['VGAE'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Downstream Task Loss')
plt.legend(loc="upper left")
ax_main = plt.gca()
ax_inset = inset_axes(ax_main, width="30%", height="30%", loc="lower left", 
                      bbox_to_anchor=(1.05, 0.2, 1, 1), 
                      bbox_transform=ax_main.transAxes)

for losses, label, color in zip(
    [losses_laplacian_downstream, losses_lle_downstream, losses_ltsa_downstream, losses_vgae_downstream, losses_vae_downstream, losses_pca_downstream, losses_diffmap_downstream, losses_ae_downstream, losses_n2v_downstream],
    ['Laplacian Eigenmaps','LLE','LTSA', 'VGAE', 'VAE', 'PCA', 'Diffusion Map', 'AE', 'Node2Vec', 'MDS', 'Isomap'],
    [colors['Laplacian Eigenmaps'], colors['LLE'], colors['LTSA'], colors['VGAE'], colors['VAE'], colors['PCA'], colors['Diffusion Map'], colors['AE'], colors['Node2Vec'], colors['MDS'], colors['Isomap']]
):
    ax_inset.plot(range(990, 1000), losses[990:1000], label=f'{label} Downstream Loss', color=color)

ax_inset.set_xlim(990, 1000)
ax_inset.set_ylim(0, 0.05)
ax_inset.set_xticks([990, 995, 1000])
ax_inset.set_yticks([0, 0.025, 0.05])
ax_inset.tick_params(axis='both', which='major', labelsize=8)

rect = patches.Rectangle((990, 0), 10, 0.02, linewidth=1, edgecolor='black', facecolor='none')
ax_main.add_patch(rect)
mark_inset(ax_main, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

plt.show()


with torch.no_grad():
    classifier1.eval()
    classifier2.eval()
    classifier3.eval()
    classifier4.eval()
    classifier5.eval()
    classifier6.eval()
    classifier7.eval()
    classifier8.eval()
    classifier9.eval()
    classifier10.eval()
    classifier11.eval()
    pred1 = classifier1(test_ltsa_embeddings).argmax(dim=1)
    pred2 = classifier2(test_vgae_embeddings).argmax(dim=1)
    pred3 = classifier3(test_vae_embeddings).argmax(dim=1)
    pred4 = classifier4(test_pca_embeddings).argmax(dim=1)
    pred5 = classifier5(test_diffmap_embeddings).argmax(dim=1)
    pred6 = classifier6(test_ae_embeddings).argmax(dim=1)
    pred7 = classifier7(test_n2v_embeddings).argmax(dim=1)
    pred8 = classifier8(test_lle_embeddings).argmax(dim=1)
    pred9 = classifier9(test_laplacian_embeddings).argmax(dim=1)
    pred10 = classifier10(test_mds_embeddings).argmax(dim=1)
    pred11 = classifier11(test_isomap_embeddings).argmax(dim=1)

print('PCA Classification Report:')
print(classification_report(test_labels, pred4.numpy()))
print('MDS Classification Report:')
print(classification_report(test_labels, pred10.numpy()))
print('Isomap Classification Report:')
print(classification_report(test_labels, pred11.numpy()))
print('LLE Classification Report:')
print(classification_report(test_labels, pred8.numpy()))
print('LTSA Classification Report:')
print(classification_report(test_labels, pred1.numpy()))
print('Laplacian Eigenmaps Classification Report:')
print(classification_report(test_labels, pred9.numpy()))
print('Diffusion Map Classification Report:')
print(classification_report(test_labels, pred5.numpy()))
print('AE Classification Report:')
print(classification_report(test_labels, pred6.numpy()))
print('VAE Classification Report:')
print(classification_report(test_labels, pred3.numpy()))
print('Node2Vec Classification Report:')
print(classification_report(test_labels, pred7.numpy()))
print('VGAE Classification Report:')
print(classification_report(test_labels, pred2.numpy()))
