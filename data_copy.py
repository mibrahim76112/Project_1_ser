import pyreadr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cupy as cp
from cuml.preprocessing import StandardScaler as cuStandardScaler


def load_data(window_size=10, stride=3, test_size=0.2):
    # Load data from RData files
    df_FaultFree = pyreadr.read_r(r'TEP_FaultFree_Training.RData')['fault_free_training']
    df_Faulty = pyreadr.read_r(r'TEP_Faulty_Training.RData')['faulty_training']

    # Combine datasets
    df = pd.concat([df_FaultFree, df_Faulty])

    # Separate features and labels
    X = df.iloc[:, 3:].values  # Features
    y = df['faultNumber'].values  # Labels

    # Standardize using GPU (only on fault-free data)
    fault_free_data = df[df['faultNumber'] == 0].iloc[:, 3:].values
    fault_free_data_gpu = cp.asarray(fault_free_data)

    scaler = cuStandardScaler()
    scaler.fit(fault_free_data_gpu)

    # Apply scaling to the entire dataset (including faulty data for inference)
    X_gpu = cp.asarray(X)
    X_scaled_gpu = scaler.transform(X_gpu)
    X_scaled = cp.asnumpy(X_scaled_gpu)  # Move back to CPU for windowing

    # Create sliding windows on the fault-free data for training
    def create_windows(X, y, window_size, stride):
        X_windows = []
        y_windows = []

        for i in range(0, len(X) - window_size + 1, stride):
            X_windows.append(X[i:i + window_size])
            y_windows.append(y[i + window_size - 1])

        return np.array(X_windows), np.array(y_windows)

    # Select only the fault-free data for training
    df_fault_free = df[df['faultNumber'] == 0]  # Fault-free data
    X_train, y_train = create_windows(df_fault_free.iloc[:, 3:].values, df_fault_free['faultNumber'].values, window_size, stride)

    # Select both fault-free and faulty data for testing
    X_test, y_test = create_windows(X_scaled, y, window_size, stride)

    # Train-test split for validation (optional)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    return X_train, X_test, y_train, y_test, len(np.unique(y))
