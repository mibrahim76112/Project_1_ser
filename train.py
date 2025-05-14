import torch
import torch.nn as nn
import torch.optim as optim
from data import load_data_lda,load_data
from utils import print_classification_metrics
from models.transformer_model_1 import TransformerModel

def train_model(model, X_train, y_train, X_test, y_test, num_classes, device):
    model = model.to(device)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1
    batch_size = 128
    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        torch.cuda.empty_cache()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

        print_classification_metrics(y_true, y_pred)

def main():
    X_train, X_test, y_train, y_test = load_data_lda(window_size=3, stride=1, apply_lda=True)
    num_classes = 21
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerModel(input_dim=X_train.shape[2], num_classes=num_classes)
    train_model(model, X_train, y_train, X_test, y_test, num_classes, device)

if __name__ == "__main__":
    main()
