import os
import numpy as np
import pandas as pd
import cupy as cp
from cuml.preprocessing import StandardScaler as cuStandardScaler
from sklearn.model_selection import train_test_split 

def load_data_lda(window_size=30, stride=4, apply_lda=False, test_size=0.2,
                  npz_path='/workspace/Data_100_x.npz', save_processed=True, mode='unsupervised'):
    """
    mode: str
        - 'supervised': Use all data for training (faulty + fault-free).
        - 'unsupervised': Use only fault-free data for training.
    """

    def create_windows(X, y, window, stride):
        num_windows = (len(X) - window) // stride + 1
        X_indices = np.arange(window)[None, :] + np.arange(num_windows)[:, None] * stride
        y_indices = np.arange(window - 1, len(X), stride)
        X_win = X[X_indices]
        y_win = y[y_indices]
        return X_win.astype(np.float32), y_win.astype(np.int32)

    # Try loading preprocessed data if available
    if os.path.exists(npz_path):
        print(f"Loading preprocessed data from {npz_path}...")
        data = np.load(npz_path)
        X_train_scaled = data['X_train']
        X_test_scaled = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        # Apply filtering based on mode
        if mode == 'unsupervised':
            print("Unsupervised mode: Filtering fault-free data for training...")
            fault_free_indices = y_train == 0
            X_train_scaled = X_train_scaled[fault_free_indices]
            y_train = y_train[fault_free_indices]

        print("Preprocessed data loaded and filtered.")

    else:
        raise FileNotFoundError(f"File not found: {npz_path}")

    print("Creating windowed datasets...")
    X_train_windows, y_train_windows = create_windows(X_train_scaled, y_train, window_size, stride)
    X_test_windows, y_test_windows = create_windows(X_test_scaled, y_test, window_size, stride)

    if apply_lda:
        print("Applying Custom GPU LDA...")

        num_samples, win_size, num_features = X_train_windows.shape
        X_train_flat = X_train_windows.reshape(-1, num_features)
        X_test_flat = X_test_windows.reshape(-1, num_features)

        X_train_gpu = cp.asarray(X_train_flat)
        y_train_gpu = cp.asarray(y_train_windows.repeat(win_size))

        class_labels = cp.unique(y_train_gpu)
        mean_vectors = {label: X_train_gpu[y_train_gpu == label].mean(axis=0) for label in class_labels}
        overall_mean = X_train_gpu.mean(axis=0)

        SW = cp.zeros((num_features, num_features), dtype=cp.float32)
        for label, mean_vec in mean_vectors.items():
            class_scatter = X_train_gpu[y_train_gpu == label] - mean_vec
            SW += class_scatter.T @ class_scatter

        SB = cp.zeros((num_features, num_features), dtype=cp.float32)
        for label, mean_vec in mean_vectors.items():
            n = cp.sum(y_train_gpu == label)
            mean_diff = (mean_vec - overall_mean).reshape(-1, 1)
            SB += n * (mean_diff @ mean_diff.T)

        eig_vals, eig_vecs = cp.linalg.eigh(cp.linalg.inv(SW) @ SB)
        eig_vecs = eig_vecs[:, ::-1]
        eig_vecs = eig_vecs[:, :20]

        X_train_lda = (X_train_gpu @ eig_vecs).get()
        X_test_lda = (cp.asarray(X_test_flat) @ eig_vecs).get()

        num_train_samples = X_train_lda.shape[0] // win_size
        num_test_samples = X_test_lda.shape[0] // win_size

        X_train_windows = X_train_lda.reshape(num_train_samples, win_size, 20).astype(np.float32)
        X_test_windows = X_test_lda.reshape(num_test_samples, win_size, 20).astype(np.float32)

    print("Data loading complete.")
    return X_train_windows, X_test_windows, y_train_windows, y_test_windows
