import torch.nn as nn
import torch
import torch.optim as optim
from data_2 import load_data
from utils import print_classification_metrics
from models.transformer_model_1 import TransformerModel
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
from torch.amp import GradScaler
from torch.amp import autocast

def train_model(model, X_train, y_train, X_test, y_test, num_classes, device):
    model = model.to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    scaler = GradScaler() 
    
    num_epochs = 10
    batch_size = 128
    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        torch.cuda.empty_cache()

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
        
        torch.cuda.empty_cache()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), "transformer_model_1.pth")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

        print_classification_metrics(y_true, y_pred)

def main():
    X_train, X_test, y_train, y_test,num_classes = load_data(window_size=10, stride=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerModel(input_dim=X_train.shape[2], num_classes=num_classes, num_layers=1, num_heads=1, ff_dim=128, dropout=0.3, max_len=500 )

    train_model(model, X_train, y_train, X_test, y_test, num_classes, device)

if __name__ == "__main__":
    main()
    

