import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import torch


def load_data():
    # Load data
    print(f"Number of rows in feature data: {X.shape[0]}")
    print(f"Number of rows in label data: {Y.shape[0]}")
    
    # Ensure alignment
    assert X.shape[0] == Y.shape[0], "Feature and label data must have the same number of rows."
    
    # Minimal preprocessing: lower case and remove excessive punctuation
    X = X.str.lower().replace('[^a-zA-Z0-9\s]', ' ', regex=True)
    print("Successfully converted text to lower case and removed punctuation.")
    
    
    # Import SentenceTransformer
    model = SentenceTransformer('gtr-t5-large')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Vectorize abstracts
    X_vectorized = model.encode(X.tolist(), show_progress_bar=True, device=device)
    print(f"Successfully vectorized abstracts.")
    
    # Normalize vectors
    X_vectorized = normalize(X_vectorized, norm='l2')
    print(f"Successfully normalized vectors.")
    
    return X_vectorized, Y
