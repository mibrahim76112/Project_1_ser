# ===================== data.py =====================
import pyreadr
import numpy as np
import pandas as pd
import cupy as cp
from cuml.preprocessing import StandardScaler as cuStandardScaler

def load_data(window_size=20, stride=5, num_samples=50000):
    # Load training data
    df_ff = pyreadr.read_r(r'TEP_FaultFree_Training.RData')['fault_free_training']
    df_faulty = pyreadr.read_r(r'TEP_Faulty_Training.RData')['faulty_training']

    # Load testing data
    df_ff_test = pyreadr.read_r(r'TEP_FaultFree_Testing.RData')['fault_free_testing']
    df_faulty_test = pyreadr.read_r(r'TEP_Faulty_Testing.RData')['faulty_testing']

    # Extract features and labels
    X_ff = df_ff.iloc[:, 3:].values
    y_ff = df_ff['faultNumber'].values
    
    X_faulty = df_faulty.iloc[:, 3:].values
    y_faulty = df_faulty['faultNumber'].values
    
    # **Concatenate the labels correctly**
    X_train_all = np.concatenate((X_ff, X_faulty), axis=0)
    y_train_all = np.concatenate((y_ff, y_faulty), axis=0)

    # ============= GPU Scaling =============
    scaler = cuStandardScaler()
    X_train_all_scaled = scaler.fit_transform(cp.asarray(X_train_all))
    
    # ============= GPU Windowing =============
    def create_windows_gpu(X, y, window, stride):
        num_windows = (len(X) - window) // stride + 1
        
        # Generate indices directly on GPU
        X_indices = cp.arange(window)[None, :] + cp.arange(num_windows)[:, None] * stride
        y_indices = cp.arange(window - 1, len(X), stride)
        
        # Perform windowed slicing
        X_win = X[X_indices]
        y_win = y[y_indices]

        # Move back to CPU
        return X_win.get(), y_win.get()

    # Train on both fault-free and faulty (now the labels are properly aligned)
    X_train, y_train = create_windows_gpu(X_train_all_scaled, cp.asarray(y_train_all), window_size, stride)

    # ** Test data is now loaded directly from testing files ** 
    X_ff_test = df_ff_test.iloc[:, 3:].values
    y_ff_test = df_ff_test['faultNumber'].values
    
    X_faulty_test = df_faulty_test.iloc[:, 3:].values
    y_faulty_test = df_faulty_test['faultNumber'].values

    # Combine both fault-free and faulty for the test set
    X_all_test = np.concatenate((X_ff_test, X_faulty_test), axis=0)
    y_all_test = np.concatenate((y_ff_test, y_faulty_test), axis=0)

    # ============= Random Sampling BEFORE Scaling =============
    if num_samples < len(X_all_test):
        random_indices = np.random.choice(len(X_all_test), num_samples, replace=False)
        X_all_test = X_all_test[random_indices]
        y_all_test = y_all_test[random_indices]

    # ============= GPU Scaling for Test Set =============
    X_all_test_scaled = scaler.transform(cp.asarray(X_all_test))

    # Create windowed test data
    X_test, y_test = create_windows_gpu(X_all_test_scaled, cp.asarray(y_all_test), window_size, stride)

    return X_train, X_test, y_train, y_test, len(np.unique(y_faulty))
