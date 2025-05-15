# ===================== data.py =====================
import pyreadr
import numpy as np
import pandas as pd
import cupy as cp
from cuml.preprocessing import StandardScaler as cuStandardScaler
from sklearn.model_selection import train_test_split

def load_data(window_size=20, stride=5, num_samples=50000, test_size=0.2, random_state=42):
    # Load training data (fault-free and faulty)
    df_ff = pyreadr.read_r(r'TEP_FaultFree_Training.RData')['fault_free_training']
    df_faulty = pyreadr.read_r(r'TEP_Faulty_Training.RData')['faulty_training']

    # Extract features and labels
    X_ff = df_ff.iloc[:, 3:].values
    y_ff = df_ff['faultNumber'].values
    
    X_faulty = df_faulty.iloc[:, 3:].values
    y_faulty = df_faulty['faultNumber'].values
    
    # Combine fault-free and faulty data
    X_all = np.concatenate((X_ff, X_faulty), axis=0)
    y_all = np.concatenate((y_ff, y_faulty), axis=0)
    print("Before splitting")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state
    )
    print("After")


    # ============= GPU Scaling =============
    scaler = cuStandardScaler()
    X_train_scaled = scaler.fit_transform(cp.asarray(X_train_raw))
    X_test_scaled = scaler.transform(cp.asarray(X_test_raw))

    # ============= GPU Windowing =============
    def create_windows_gpu(X, y, window, stride):
        num_windows = (len(X) - window) // stride + 1
        X_indices = cp.arange(window)[None, :] + cp.arange(num_windows)[:, None] * stride
        y_indices = cp.arange(window - 1, len(X), stride)
        X_win = X[X_indices]
        y_win = y[y_indices]
        return X_win.get(), y_win.get()

    X_train, y_train = create_windows_gpu(X_train_scaled, cp.asarray(y_train_raw), window_size, stride)
    X_test, y_test = create_windows_gpu(X_test_scaled, cp.asarray(y_test_raw), window_size, stride)

    # ============= Optional Downsampling of Test Set =============
    if num_samples < len(X_test):
        random_indices = np.random.choice(len(X_test), num_samples, replace=False)
        X_test = X_test[random_indices]
        y_test = y_test[random_indices]

    print("Sample X_train window (first sample):", X_train[0])
    print("Corresponding y_train label:", y_train[0])
    print("Sample X_test window (first sample):", X_test[0])
    print("Corresponding y_test label:", y_test[0])
    print(f"Returning {len(np.unique(y_train))} train classes and {len(np.unique(y_test))} test classes")

    return X_train, X_test, y_train, y_test, len(np.unique(y_test))
