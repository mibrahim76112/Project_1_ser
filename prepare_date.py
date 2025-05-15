import pyreadr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_and_save_data(test_size=0.2, save_path='E:\PROJECT-1\myenv\prepare_data', sample_fraction=0.4):
    print("Loading datasets...")
    df_ff = pyreadr.read_r(r'/workspace/TEP_FaultFree_Training.RData')['fault_free_training']
    df_faulty = pyreadr.read_r(r'TEP_Faulty_Training.RData')['faulty_training']

    # Combine fault-free and faulty data for splitting
    df_all = pd.concat([df_ff, df_faulty])

    # Take a 50% random sample of the data
    print(f"Sampling {int(sample_fraction * 100)}% of the data...")
    df_all = df_all.sample(frac=sample_fraction, random_state=41)

    # Extract features and labels for all data
    X_all = df_all.iloc[:, 3:].values.astype(np.float32)
    y_all = df_all['faultNumber'].values.astype(np.int32)

    # Split into train and test sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=41)

    # Scale datasets on CPU
    print("Scaling datasets on CPU...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save to NPZ
    np.savez(save_path,
             X_train=X_train_scaled,
             X_test=X_test_scaled,
             y_train=y_train,
             y_test=y_test)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    prepare_and_save_data(test_size=0.2, save_path='E:\PROJECT-1\myenv\prepare_data_100.npz', sample_fraction=1)
