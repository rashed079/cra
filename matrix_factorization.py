# src/matrix_factorization.py
import numpy as np
from sklearn.decomposition import NMF

def matrix_factorization(X_train, n_components=2):
    nmf = NMF(n_components=n_components)
    W = nmf.fit_transform(X_train)
    H = nmf.components_
    return W, H

if __name__ == "__main__":
    from data_processing import preprocess_data, load_data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    W, H = matrix_factorization(X_train)
    print("Latent Features (W):", W)
    print("Feature Contributions (H):", H)