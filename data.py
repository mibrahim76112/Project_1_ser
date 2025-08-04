import pyreadr
import numpy as np
import pandas as pd

# Optional GPU support
try:
    import cupy as cp
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
except ImportError:
    cp = np  # Fallback to NumPy
    GPU_AVAILABLE = False

from sklearn.preprocessing import StandardScaler as skStandardScaler


def read_training_data():
    """
    Reads the Tennessee Eastman Process (TEP) training data from RData files.

    Returns:
        pd.DataFrame: Concatenated and sorted DataFrame of fault-free and faulty training data.
    """
    b1 = pyreadr.read_r("/content/TEP_FaultFree_Training.RData")['fault_free_training']
    b2 = pyreadr.read_r("/content/TEP_Faulty_Training.RData")['faulty_training']
    train_ts = pd.concat([b1, b2])
    return train_ts.sort_values(by=['faultNumber', 'simulationRun'])


def sample_train_and_test(train_ts, type_model):
    """
    Samples training and test data based on model type (supervised or unsupervised).

    Args:
        train_ts (pd.DataFrame): The full training dataset.
        type_model (str): Either "supervised" or "unsupervised".

    Returns:
        tuple: (sampled_train, sampled_test) as pandas DataFrames.
    """
    sampled_train, sampled_test = pd.DataFrame(), pd.DataFrame()
    fault_0_data = train_ts[train_ts['faultNumber'] == 0]

    frames_train = []
    if type_model == "supervised":
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
    elif type_model == "unsupervised":
        frames_train.append(fault_0_data)
    sampled_train = pd.concat(frames_train)

    frames_test = []
    for i in sorted(train_ts['faultNumber'].unique()):
        if i == 0:
            frames_test.append(fault_0_data.iloc[20000:22000])
        else:
            fr = []
            b = train_ts[train_ts['faultNumber'] == i]
            for x in range(26, 35):
                b_x = b[b['simulationRun'] == x].iloc[160:660]
                fr.append(b_x)
            frames_test.append(pd.concat(fr))
    sampled_test = pd.concat(frames_test)

    return sampled_train, sampled_test


def scale_and_window(X_df, scaler, use_gpu=True, y_col='faultNumber', window_size=20, stride=5):
    """
    Applies scaling and sliding window segmentation to the dataset.

    Args:
        X_df (pd.DataFrame): Input time-series data with labels.
        scaler (cuML StandardScaler): Fitted GPU-based scaler.
        y_col (str): Column name containing the target/fault labels.
        window_size (int): Size of each time window.
        stride (int): Step size between consecutive windows.

    Returns:
        tuple: (X_win, y_win) where
            - X_win (np.ndarray): 3D array of shape [num_windows, window_size, num_features].
            - y_win (np.ndarray): 1D array of labels aligned with each window.
    """
    y = X_df[y_col].values
    X = X_df.iloc[:, 3:].values

    if use_gpu and GPU_AVAILABLE:
        X_scaled = scaler.transform(cp.asarray(X))
        num_windows = (len(X_scaled) - window_size) // stride + 1
        X_indices = cp.arange(window_size)[None, :] + cp.arange(num_windows)[:, None] * stride
        y_indices = cp.arange(window_size - 1, len(X_scaled), stride)

        X_win = X_scaled[X_indices]
        y_win = cp.asarray(y)[y_indices]

        return X_win.get(), y_win.get()
    else:
        X_scaled = scaler.transform(X)
        num_windows = (len(X_scaled) - window_size) // stride + 1
        X_indices = np.arange(window_size)[None, :] + np.arange(num_windows)[:, None] * stride
        y_indices = np.arange(window_size - 1, len(X_scaled), stride)

        X_win = np.array([X_scaled[idx] for idx in X_indices])
        y_win = y[y_indices]

        return X_win, y_win


def load_sampled_data(window_size=20, stride=5, type_model="supervised", use_gpu=True):
    """
    Loads, scales, and windows the sampled training and test data.

    Args:
        window_size (int): Length of each sliding window.
        stride (int): Step size for windowing.
        type_model (str): Either "supervised" or "unsupervised".

    Returns:
        tuple: (X_train, X_test, y_train, y_test) as NumPy arrays.
            - X_train: [N_train, window_size, num_features]
            - X_test: [N_test, window_size, num_features]
            - y_train: [N_train]
            - y_test: [N_test]
    """
    train_ts = read_training_data()
    sampled_train, sampled_test = sample_train_and_test(train_ts, type_model)

    fault_free = sampled_train[sampled_train['faultNumber'] == 0].iloc[:, 3:].values

    if use_gpu and GPU_AVAILABLE:
        scaler = cuStandardScaler()
        scaler.fit(cp.asarray(fault_free))
        print(f"[INFO] Using GPU Scaler (cuML), fit on fault-free samples: {fault_free.shape[0]} rows")
    else:
        scaler = skStandardScaler()
        scaler.fit(fault_free)
        print(f"[INFO] Using CPU Scaler (scikit-learn), fit on fault-free samples: {fault_free.shape[0]} rows")

    print("[INFO] Scaling and windowing training data...")
    X_train, y_train = scale_and_window(sampled_train, scaler,
                                        use_gpu=use_gpu,
                                        y_col='faultNumber',
                                        window_size=window_size, stride=stride)

    print("[INFO] Scaling and windowing test data...")
    X_test, y_test = scale_and_window(sampled_test, scaler,
                                      use_gpu=use_gpu,
                                      y_col='faultNumber',
                                      window_size=window_size, stride=stride)

    return X_train, X_test, y_train, y_test
