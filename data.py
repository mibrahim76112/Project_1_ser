import os
import pyreadr
import numpy as np
import pandas as pd
import cupy as cp
from cuml.preprocessing import StandardScaler as cuStandardScaler
from sklearn.model_selection import train_test_split 

def load_data_lda(window_size=30, stride=4, apply_lda=False, test_size=0.2,
                  npz_path='data.npz', save_processed=True):
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
        print("Preprocessed data loaded.")
    else:
        print("Loading raw datasets...")
        df_ff = pyreadr.read_r(r'TEP_FaultFree_Training.RData')['fault_free_training']
        df_faulty = pyreadr.read_r(r'TEP_Faulty_Training.RData')['faulty_training']

        df_all = pd.concat([df_ff, df_faulty])
        fault_free = df_all[df_all['faultNumber'] == 0].iloc[:, 3:]
        #X = fault_free.values.astype(np.float32)
        #y = df_all[df_all['faultNumber'] == 0]['faultNumber'].values.astype(np.int32)

        # Split train/test
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=41)
        X_all = df_all.iloc[:, 3:].values.astype(np.float32)
        y_all = df_all['faultNumber'].values  # Labels

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=41)


        # Scale on GPU
        print("Scaling datasets on GPU...")
        scaler = cuStandardScaler()
        X_train_scaled = scaler.fit_transform(cp.asarray(X_train)).get()
        X_test_scaled = scaler.transform(cp.asarray(X_test)).get()

        # Save processed data for future use
        if save_processed:
            np.savez(npz_path,
                     X_train=X_train_scaled,
                     X_test=X_test_scaled,
                     y_train=y_train,
                     y_test=y_test)
            print(f"Processed data saved to {npz_path}")

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
