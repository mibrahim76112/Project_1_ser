import pyreadr
import numpy as np
import cupy as cp
import pandas as pd
from cuml.preprocessing import StandardScaler as cuStandardScaler
from sklearn.model_selection import train_test_split

def prepare_and_save_data(test_size=0.2, save_path='/workspace/.npz'):
    print("Loading datasets...")
    df_ff = pyreadr.read_r(r'TEP_FaultFree_Training.RData')['fault_free_training']
    df_faulty = pyreadr.read_r(r'TEP_Faulty_Training.RData')['faulty_training']

    # Combine fault-free and faulty data for splitting
    df_all = pd.concat([df_ff, df_faulty])
    
    # Filter fault-free data only for training/testing (or adjust as needed)
    fault_free = df_all[df_all['faultNumber'] == 0].iloc[:, 3:]

    # Extract features and labels for fault-free data (adjust if you want full dataset split)
  #  X = fault_free.values.astype(np.float32)
   # y = df_all[df_all['faultNumber'] == 0]['faultNumber'].values.astype(np.int32)
    X_all = df_all.iloc[:, 3:].values.astype(np.float32)
    y_all = df_all['faultNumber'].values  # Labels

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=41)

    # Scale datasets on GPU
    print("Scaling datasets on GPU...")
    scaler = cuStandardScaler()
    X_train_scaled = scaler.fit_transform(cp.asarray(X_train)).get()
    X_test_scaled = scaler.transform(cp.asarray(X_test)).get()

    
    np.savez(save_path,
             X_train=X_train_scaled,
             X_test=X_test_scaled,
             y_train=y_train,
             y_test=y_test)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    prepare_and_save_data(test_size=0.2, save_path='/workspace/.npz')