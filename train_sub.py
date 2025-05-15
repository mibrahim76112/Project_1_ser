import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.transformer_autoencoder import TransformerAutoencoder
from data import load_sampled_data
import numpy as np
from models.selfgated_hierarchial_transformer import SelfGatedHierarchicalTransformerEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def train_unsupervised_model(model, X_train, X_test, y_test=None, device='cpu', num_epochs=10, batch_size=128, lr=0.001):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    return model


def evaluate_reconstruction(model, X_test, y_test, device='cpu', batch_size=128, max_test_samples=10000):
    model.eval()

    # Step 1: Randomly sample indices
    total_samples = len(X_test)
    sample_size = min(max_test_samples, total_samples)
    indices = np.random.choice(total_samples, size=sample_size, replace=False)

    X_test_sampled = X_test[indices]
    y_test_sampled = y_test[indices]

    # Step 2: Convert to tensors
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test_sampled, dtype=torch.float32),
        torch.tensor(y_test_sampled, dtype=torch.int32)
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    reconstructions = []
    labels = []

    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            recon = model(batch_x).cpu().numpy()
            reconstructions.append(recon)
            labels.append(batch_y.numpy())

    reconstructions = np.concatenate(reconstructions, axis=0)
    y_true = np.concatenate(labels, axis=0)

    recon_errors = np.mean((reconstructions - X_test_sampled) ** 2, axis=(1, 2))

    fault_free_mask = y_true == 0
    
    threshold = np.percentile(recon_errors[fault_free_mask], 10)


    print(f"Auto threshold: {threshold:.4f}")

    y_pred = (recon_errors > threshold).astype(int)
    y_binary = (y_true > 0).astype(int)

   
    accuracy = accuracy_score(y_binary, y_pred)
    f1 = f1_score(y_binary, y_pred, zero_division=0)
    precision = precision_score(y_binary, y_pred, zero_division=0)
    recall = recall_score(y_binary, y_pred, zero_division=0)

    print(f"Accuracy     : {accuracy:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")




def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    X_train, X_test, _, y_test = load_sampled_data(window_size=10, stride=3, type_model = "unsupervised")

    model = TransformerAutoencoder(input_dim=X_train.shape[2], seq_len=X_train.shape[1])
    
    trained_model = train_unsupervised_model(model, X_train, X_test, device=device)

    evaluate_reconstruction(trained_model, X_test, y_test, device=device)

if __name__ == "__main__":
    main()
