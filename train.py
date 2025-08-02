import torch.nn as nn
import torch
import torch.optim as optim
from data import load_sampled_data
from utils import print_classification_metrics
# from models.transformer_model_1 import TransformerModel
from models.hierarchial_transformer import HierarchicalTransformerEncoder
from models.selfgated_hierarchial_transformer import SelfGatedHierarchicalTransformerEncoder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import GradScaler
from torch.amp import autocast

def train_model(model, X_train, y_train, X_test, y_test, num_classes, device):
    """
    Trains a Transformer-based model on time-series data using mixed precision,
    and evaluates it on a test set.

    Args:
        model (torch.nn.Module): The model to be trained.
        X_train (np.ndarray): Training input features of shape [N, T, F].
        y_train (np.ndarray): Training labels of shape [N].
        X_test (np.ndarray): Test input features of shape [M, T, F].
        y_test (np.ndarray): Test labels of shape [M].
        num_classes (int): Number of classification classes.
        device (torch.device): The device to train on ('cuda' or 'cpu').

    Returns:
        None
    """
    model = model.to(device)

    # Convert data to tensors and move to device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()  # Mixed-precision training

    num_epochs = 250
    batch_size = 64
    loss_history = []

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        epoch_loss = 0.0

        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                output = model(batch_x)
                loss = criterion(output, batch_y.squeeze())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (X_train_tensor.size(0) // batch_size)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), "transformer_model_1.pth")

    # Plot loss curve
    plt.figure()
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("loss_curve.png")
    plt.show()

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        print_classification_metrics(y_true, y_pred)


def main():
    """
    Main function to load data, instantiate the model, and initiate training.

    Loads sampled time-series data, applies preprocessing, instantiates the
    Self-Gated Hierarchical Transformer model, and trains it using `train_model`.

    Returns:
        None
    """
    # Load data
    X_train, X_test, y_train, y_test = load_sampled_data(window_size=200, stride=5)

    print("Training Data Shape:", X_train.shape, y_train.shape)
    print("Test Data Shape:", X_test.shape, y_test.shape)
    print("Unique classes in Training Set:", np.unique(y_train))
    print("Unique classes in Test Set:", np.unique(y_test))
    num_classes = 21

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model (you can swap models here)
    model = SelfGatedHierarchicalTransformerEncoder(
        input_dim=X_train.shape[2],
        d_model=128,
        nhead=4,
        num_layers_low=2,
        num_layers_high=2,
        dim_feedforward=128,
        dropout=0.05,
        pool_output_size=10,
        num_classes=num_classes
    )

    # Train and evaluate the model
    train_model(model, X_train, y_train, X_test, y_test, num_classes, device)


if __name__ == "__main__":
    main()
