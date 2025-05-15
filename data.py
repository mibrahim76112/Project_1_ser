# ================ data.py ================
import pyreadr
import numpy as np
import pandas as pd
import cupy as cp
from cuml.preprocessing import StandardScaler as cuStandardScaler

def read_training_data():
    """Reads and merges fault-free and faulty training data from .RData files."""
    b1 = pyreadr.read_r("TEP_FaultFree_Training.RData")['fault_free_training']
    b2 = pyreadr.read_r("TEP_Faulty_Training.RData")['faulty_training']
    train_ts = pd.concat([b1, b2])
    return train_ts.sort_values(by=['faultNumber', 'simulationRun'])

def sample_train_and_test(train_ts):
    """Samples train and test data from training set using predefined logic."""
    sampled_train, sampled_test = pd.DataFrame(), pd.DataFrame()
    
    # Sampled Train
    frames_train = []
    for i in sorted(train_ts['faultNumber'].unique()):
        if i == 0:
            frames_train.append(train_ts[train_ts['faultNumber'] == i].iloc[0:20000])
        else:
            fr = []
            b = train_ts[train_ts['faultNumber'] == i]
            for x in range(1, 25):
                b_x = b[b['simulationRun'] == x].iloc[20:500]
                fr.append(b_x)
            frames_train.append(pd.concat(fr))
    sampled_train = pd.concat(frames_train)

    # Sampled Test
    frames_test = []
    for i in sorted(train_ts['faultNumber'].unique()):
        if i == 0:
            frames_test.append(train_ts[train_ts['faultNumber'] == i].iloc[30000:32000])
        else:
            fr = []
            b = train_ts[train_ts['faultNumber'] == i]
            for x in range(36, 46):
                b_x = b[b['simulationRun'] == x].iloc[160:660]
                fr.append(b_x)
            frames_test.append(pd.concat(fr))
    sampled_test = pd.concat(frames_test)

    return sampled_train, sampled_test

def scale_and_window(X_df, y_col='faultNumber', window_size=20, stride=5):
    """Scales and applies sliding window on GPU using CuML."""
    y = X_df[y_col].values
    X = X_df.iloc[:, 3:].values  # Skip metadata columns

    scaler = cuStandardScaler()
    X_scaled = scaler.fit_transform(cp.asarray(X))

    # Create windows
    num_windows = (len(X_scaled) - window_size) // stride + 1
    X_indices = cp.arange(window_size)[None, :] + cp.arange(num_windows)[:, None] * stride
    y_indices = cp.arange(window_size - 1, len(X_scaled), stride)

    X_win = X_scaled[X_indices]
    y_win = cp.asarray(y)[y_indices]

    return X_win.get(), y_win.get()

def load_sampled_data(window_size=20, stride=5):
    """Main function to load sampled and windowed train/test data (no CV)."""
    train_ts = read_training_data()
    sampled_train, sampled_test = sample_train_and_test(train_ts)

    print("[INFO] Scaling and windowing training data...")
    X_train, y_train = scale_and_window(sampled_train, window_size=window_size, stride=stride)

    print("[INFO] Scaling and windowing test data...")
    X_test, y_test = scale_and_window(sampled_test, window_size=window_size, stride=stride)

    return X_train, X_test, y_train, y_test
