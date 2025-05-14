from models.transformer_classifier import TransformerClassifier
from data import load_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_supervised(model, X_train, y_train, device, num_epochs=20, lr=1e-3, batch_size=128):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}")

    return model

def ensemble_predict(models, X, device):
    loaders = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=128)
    preds = []
    for model in models:
        model.eval()
        all_preds = []
        with torch.no_grad():
            for x in loaders:
                x = x.to(device)
                out = model(x)
                all_preds.append(torch.softmax(out, dim=1).cpu().numpy())
        preds.append(np.concatenate(all_preds, axis=0))
    return np.mean(preds, axis=0)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train, X_test, y_train, y_test = load_data()

    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    models = []
    for i in range(3):  # Ensemble of 3
        print(f"\nTraining model {i+1}")
        model = TransformerClassifier(input_dim, seq_len, num_classes)
        trained_model = train_supervised(model, X_train, y_train, device)
        models.append(trained_model)

    probs = ensemble_predict(models, X_test, device)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\nEnsemble Accuracy: {acc:.4f}")
    print(f"Ensemble F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
