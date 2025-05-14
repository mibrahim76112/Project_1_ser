import pyreadr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cupy as cp
# If you are using the CPU, use this import
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# If you are using the GPU, use this import
from cuml.preprocessing import StandardScaler as cuStandardScaler

def load_data(window_size=20, stride=5):
    df_ff = pyreadr.read_r(r'TEP_FaultFree_Training.RData')['fault_free_training']
    df_faulty = pyreadr.read_r(r'TEP_Faulty_Training.RData')['faulty_training']

    X_ff = df_ff.iloc[:, 3:].values
    y_ff = df_ff['faultNumber'].values
    X_faulty = df_faulty.iloc[:, 3:].values
    y_faulty = df_faulty['faultNumber'].values

    scaler = StandardScaler()
    X_ff_scaled = scaler.fit_transform(X_ff)
    X_faulty_scaled = scaler.transform(X_faulty)

    def create_windows(X, y, window, stride):
        X_win, y_win = [], []
        for i in range(0, len(X) - window, stride):
            X_win.append(X[i:i+window])
            y_win.append(y[i + window - 1])
        return np.array(X_win), np.array(y_win)

    # Train on fault-free only
    X_train, y_train = create_windows(X_ff_scaled, y_ff, window_size, stride)

    # Test on both faulty and fault-free
    df_all_test = pd.concat([df_ff, df_faulty])
    X_all = df_all_test.iloc[:, 3:].values
    y_all = df_all_test['faultNumber'].values
    X_all_scaled = scaler.transform(X_all)

    X_test, y_test = create_windows(X_all_scaled, y_all, window_size, stride)

    return X_train, X_test, y_train, y_test



def load_data_lda(window_size=30, stride=4, apply_lda=False):
    df_ff = pyreadr.read_r(r'TEP_FaultFree_Training.RData')['fault_free_training']
    df_faulty = pyreadr.read_r(r'TEP_Faulty_Training.RData')['faulty_training']

    X_ff = df_ff.iloc[:, 3:].values.astype(np.float32)
    y_ff = df_ff['faultNumber'].values
    X_faulty = df_faulty.iloc[:, 3:].values.astype(np.float32)
    y_faulty = df_faulty['faultNumber'].values

    # Scale on GPU
    scaler = cuStandardScaler()
    X_ff_scaled = scaler.fit_transform(cp.asarray(X_ff)).get()
    X_faulty_scaled = scaler.transform(cp.asarray(X_faulty)).get()

    X_all_train = np.concatenate([X_ff_scaled, X_faulty_scaled])
    y_all_train = np.concatenate([y_ff, y_faulty])

    def create_windows(X, y, window, stride):
        X_win, y_win = [], []
        for i in range(0, len(X) - window, stride):
            X_win.append(X[i:i+window])
            y_win.append(y[i + window - 1])
        return np.array(X_win, dtype=np.float32), np.array(y_win)

    X_train, y_train = create_windows(X_all_train, y_all_train, window_size, stride)

    df_all_test = pd.concat([df_ff, df_faulty])
    X_all = df_all_test.iloc[:, 3:].values.astype(np.float32)
    y_all = df_all_test['faultNumber'].values
    X_all_scaled = scaler.transform(cp.asarray(X_all)).get()
    X_test, y_test = create_windows(X_all_scaled, y_all, window_size, stride)

    if apply_lda:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        lda = LinearDiscriminantAnalysis(n_components=min(20, np.unique(y_train).size - 1))
        X_train_lda = lda.fit_transform(X_train_flat, y_train)
        X_test_lda = lda.transform(X_test_flat)

        time_steps = window_size
        feature_dim = X_train_lda.shape[1] // time_steps
        X_train = X_train_lda.reshape(-1, time_steps, feature_dim).astype(np.float32)
        X_test = X_test_lda.reshape(-1, time_steps, feature_dim).astype(np.float32)

    return X_train, X_test, y_train, y_test