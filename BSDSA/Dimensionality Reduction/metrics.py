from sklearn.metrics import pairwise_distances
from scipy.spatial import distance_matrix
import numpy as np

def calculate_npr(X_true, X_embedd, neighbours=5):
    dist_true = pairwise_distances(X_true)
    dist_embedd = pairwise_distances(X_embedd)
    
    true_nearest_neighbors = np.argsort(dist_true, axis=1)[:, 1:neighbours + 1]  # Skip self-distance at index 0
    embedd_nearest_neighbors = np.argsort(dist_embedd, axis=1)[:, 1:neighbours + 1]
    
    preserved_neighbors = 0
    for i in range(X_true.shape[0]):
        preserved_neighbors += len(set(true_nearest_neighbors[i]).intersection(set(embedd_nearest_neighbors[i])))
    
    npr = preserved_neighbors / (X_true.shape[0] * neighbours)
    return npr

def calculate_continuity(X_true, X_embedd, neighbours=5):
    dist_true = pairwise_distances(X_true)
    dist_embedd = pairwise_distances(X_embedd)
    
    true_nearest_neighbors = np.argsort(dist_true, axis=1)[:, 1:neighbours + 1]  # Skip self-distance at index 0
    continuity_sum = 0
    n_samples = X_true.shape[0]
    for i in range(n_samples):
        embedd_ranks = np.argsort(np.argsort(dist_embedd[i]))
        
        for j in true_nearest_neighbors[i]:
            rank_in_embedd = embedd_ranks[j]
            if rank_in_embedd >= neighbours:
                continuity_sum += 1  
    max_possible_displacement = neighbours * n_samples
    continuity_score = 1 - (continuity_sum / max_possible_displacement)
    continuity_score = max(0, min(continuity_score, 1))
    
    return continuity_score

def calculate_pairwise_distance_correlation(X_true, X_embedd):
    dist_true = pairwise_distances(X_true)
    dist_embedd = pairwise_distances(X_embedd)
    corr = np.corrcoef(dist_true.ravel(), dist_embedd.ravel())[0, 1]
    return corr

def calculate_stress(original_data, embedded_data):
    d_high = distance_matrix(original_data, original_data)
    d_low = distance_matrix(embedded_data, embedded_data)
    stress = np.sqrt(np.sum((d_high - d_low) ** 2) / np.sum(d_high ** 2))
    return stress
